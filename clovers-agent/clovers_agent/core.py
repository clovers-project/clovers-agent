import json
import asyncio
import httpx
from datetime import datetime
from pathlib import Path
from importlib import import_module
from clovers.utils import import_name, list_modules
from clovers.logger import logger
from collections import deque
from collections.abc import Iterable, Callable, Coroutine
from typing import Any, Concatenate
from .embedding import SentenceTransformer, TopicDecoupler
from .typing import Event, Message, SystemMessage, Payload, FunctionToolInfo
from .typing.message import UserMessage, AssistantMessage, ToolMessage, ContentSegment
from .typing.json_schema import JSONSchemaType
from .config import OpenAIConfig, Config

type AgentFunction[**P] = Callable[Concatenate["CloversAgent", Any, P], Coroutine[Any, Any, str]]
type WrappedAgentFunction[**P] = Callable[Concatenate[str, "CloversAgent", Any, P], Coroutine[Any, Any, tuple[ToolMessage, str]]]


def extract_plain_text(content: str | list[ContentSegment]) -> str:
    return content if isinstance(content, str) else "\n".join(text["text"] for text in content if text["type"] == "text")


def char_count(content: str | list[ContentSegment]):
    return len(content) if isinstance(content, str) else sum(len(text["text"]) for text in content if text["type"] == "text")


def int_generator():
    i = 0
    while True:
        yield i
        i += 1


class SkillCore:
    type Parameters = dict[str, JSONSchemaType]

    def __init__(self, name: str = "") -> None:
        self.category_id = int_generator()
        self.name = name
        self.intro_tools: list[FunctionToolInfo] = []
        self.skill_init: dict[str, AgentFunction] = {}
        self.manifest: dict[str, FunctionToolInfo] = {}
        self.invoker: dict[str, WrappedAgentFunction] = {}
        self.__map_category_to_id: dict[str, int] = {}
        self.__map_id_to_tools: dict[int, list[FunctionToolInfo]] = {}

    def all_categories(self):
        return self.__map_category_to_id.keys()

    def select_tools(self, category: str) -> list[FunctionToolInfo]:
        if category not in self.__map_category_to_id:
            return []
        return self.__map_id_to_tools[self.__map_category_to_id[category]]

    def register(self, name: str, description: str, parameters: Parameters | None = None, categories: str | Iterable[str] | None = None):
        if name in self.invoker:
            raise ValueError(f"Tool {name} already exists.")
        info: FunctionToolInfo = {"type": "function", "function": {"name": name, "description": description}}
        if parameters:
            info["function"]["parameters"] = {"type": "object", "properties": parameters, "required": list(parameters.keys())}
        # info 是 OpneAI API 要求的 tools 字段中元素的格式
        if not categories:
            self.intro_tools.append(info)
        else:
            if isinstance(categories, str):
                categories = [categories]
            for category in categories:
                category_id = self.__map_category_to_id[category] if category in self.__map_category_to_id else next(self.category_id)
                self.__map_category_to_id[category] = category_id
                if category_id not in self.__map_id_to_tools:
                    self.__map_id_to_tools[category_id] = []
                self.__map_id_to_tools[category_id].append(info)
        self.manifest[info["function"]["name"]] = info

        def decorator(func: AgentFunction) -> WrappedAgentFunction:
            async def wrapper(tool_call_id, agent: CloversAgent, event, /, **kwargs):
                logger.info(f"[{agent.name}][CALL][{name}] called")
                logger.debug(f"[{agent.name}][CALL][{name}] called with {kwargs}")
                try:
                    content = await func(agent, event, **kwargs)
                except Exception as e:
                    logger.exception(e)
                    content = "Error"
                message: ToolMessage = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                return message, name

            self.invoker[name] = wrapper
            return wrapper

        return decorator

    def merge(self, others: "SkillCore"):
        conflict = (others.invoker.keys() & self.invoker.keys()) | (others.skill_init.keys() & self.skill_init.keys())
        if conflict:
            return conflict
        self.intro_tools.extend(others.intro_tools)
        self.manifest.update(others.manifest)
        self.skill_init.update(others.skill_init)
        self.invoker.update(others.invoker)
        for category, category_id in others.__map_category_to_id.items():
            if category in self.__map_category_to_id:
                self.__map_id_to_tools[self.__map_category_to_id[category]].extend(others.__map_id_to_tools[category_id])
            else:
                new_category_id = next(self.category_id)
                self.__map_id_to_tools[new_category_id] = others.__map_id_to_tools[category_id]
                self.__map_category_to_id[category] = new_category_id

    def on_skill(self, categories: str | Iterable[str]):
        def decorator(func: AgentFunction) -> AgentFunction:
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.exception(e)
                    return f"{categories} 初始化失败"

            if isinstance(categories, str):
                self.skill_init[categories] = wrapper
            else:
                for category in categories:
                    self.skill_init[category] = wrapper

            return wrapper

        return decorator


class ContextRecoder:
    """会话上下文管理器"""

    records: deque[tuple[UserMessage | SystemMessage, AssistantMessage | SystemMessage]]

    def __init__(self, size: int) -> None:
        self.records = deque(maxlen=size)
        self.lock = asyncio.Lock()

    def __iter__(self):
        for record in self.records:
            yield record[0]
            yield record[1]

    def __bool__(self):
        return bool(self.records)

    def over(self, request: UserMessage | SystemMessage, reply: AssistantMessage | SystemMessage):
        self.records.append((request, reply))

    def clear(self):
        self.records.clear()


class Session(ContextRecoder):
    records: deque[tuple[UserMessage, AssistantMessage, int | float]]
    silence: deque[tuple[str, int | float]]
    snap: ContextRecoder
    extra: dict

    def __init__(self, size: int, sentence_model: SentenceTransformer) -> None:
        super().__init__(size)
        self.silence = deque()
        self.snap = ContextRecoder(size)
        self.skill_menu: str | None = None
        self.current_input: UserMessage | None = None
        self.decoupler = TopicDecoupler(sentence_model)
        self.extra = {}

    def memory_filter(self, timeout: int | float):
        """过滤记忆"""
        while self.records and (self.records[0][2] <= timeout):
            self.records.popleft()

    def silence_filter(self, timeout: int | float):
        """过滤静默记录群聊上下文"""
        while self.silence and (self.silence[0][1] <= timeout):
            self.silence.popleft()

    def step(self, message: str):
        if sum(char_count(msg["content"]) for msg in self) < 2000:
            return False
        return self.decoupler.step(message)

    def over(self, request: UserMessage, reply: AssistantMessage, timestamp: int | float):
        """处理完成"""
        self.records.append((request, reply, timestamp))
        self.silence.clear()

    def clear(self):
        """清理记录"""
        self.records.clear()
        self.silence.clear()

    def sync_snap(self):
        """同步上下文到辅助AI"""
        self.snap.clear()
        self.snap.records.extend((x, y) for x, y, _ in self.records)


class OpenAIAPI:
    def __init__(self, async_client: httpx.AsyncClient, config: OpenAIConfig) -> None:
        self.async_client = async_client
        self.url = f"{config.url.rstrip("/")}/chat/completions"
        self.model = config.model
        self.headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        self.extra_body = config.extra_body

    @staticmethod
    def build_content(text: str, image_list: list[str] | None) -> str | list[ContentSegment]:
        if not image_list:
            return text
        else:
            content = []
            if text:
                content.append({"type": "text", "text": text})
            if image_list:
                content.extend({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_list)
            return content

    def build_payload(self, context: Iterable[Message] | None = None, system_prompt: str | None = None) -> Payload:
        payload: Payload = {"model": self.model, "messages": [], **self.extra_body}  # type: ignore 这里允许额外请求体
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        if context:
            payload["messages"].extend(context)
        return payload

    async def call_api(self, payload: Payload) -> AssistantMessage:
        resp = await self.async_client.post(self.url, headers=self.headers, json=payload)
        if resp.status_code != 200:
            logger.error(json.dumps(payload, indent=4, ensure_ascii=False))
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


class CloversAgent(SkillCore, OpenAIAPI):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, config: Config) -> None:
        SkillCore.__init__(self, name)
        OpenAIAPI.__init__(self, async_client, config.primary)
        self.auxiliary = OpenAIAPI(async_client, config.auxiliary) if config.auxiliary is not None else self
        self.style_prompt = config.style_prompt
        self.chat_prompt = config.chat_prompt
        self.call_prompt = config.call_prompt
        self.memory_size = config.memory_size
        self.memory_timeout = config.memory_timeout
        self.topic_coldown = config.topic_coldown
        self.sentence_model = SentenceTransformer(config.sentence_model, cache_folder=config.sentence_model_cache)
        self.sessions: dict[str, Session] = {}
        # 注册技能
        self.categories: list[str] = []
        self._plugins = config.plugins
        self._plugins_dirs = config.plugin_dirs
        self.register(
            "skill_menu",
            "获取更多技能，如果assistant无法单独完成用户指令，则需要调用此方法获取更多技能。",
            {"category": {"type": "string", "description": "选择需要的技能关键词", "enum": self.categories}},
        )(self.skill_menu)

    def init(self) -> None:
        self.load_plugins_from_list(self._plugins)
        self.load_plugins_from_dirs(self._plugins_dirs)
        self.categories.extend(self.all_categories())

    @staticmethod
    async def skill_menu(agent: "CloversAgent", event: Event, category: str):
        tip = f"已获取技能：{category}"
        agent.current_session(event).skill_menu = category
        hook = agent.skill_init.get(category)
        if hook:
            info = await hook(agent, event)
            if info:
                tip += f"\n{info}"
        return tip

    async def summary_context(self, session: Session):
        payload = self.auxiliary.build_payload(context=session)
        payload["messages"].append({"role": "system", "content": "对以上对话进行总结，保留核心内容和结论，禁止输出除总结外的其他内容。"})
        summary = (await self.auxiliary.call_api(payload))["content"].strip()
        logger.info(f"[{self.name}][SUMMARY]")
        logger.debug(f"[{self.name}][SUMMARY]{summary}")
        return summary

    async def function_call(self, event: Event, call_infos: list[dict]) -> list[tuple[ToolMessage, str]]:
        task_queue = []
        for call_info in call_infos:
            func = self.invoker[call_info["function"]["name"]]
            kwargs = json.loads(call_info["function"]["arguments"])
            task_queue.append(func(call_info["id"], self, event, **kwargs))
        return await asyncio.gather(*task_queue)

    @staticmethod
    def session_id(event: Event) -> str:
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, event: Event):
        session_id = self.session_id(event)
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(self.memory_size, self.sentence_model)
        return self.sessions[session_id]

    async def aux_reply(self, session: Session, message: UserMessage):
        if not session.silence:
            return
        async with session.snap.lock:
            if not session.lock.locked():
                return
            payload = self.auxiliary.build_payload((*session.snap, message), f"{self.style_prompt}\n{self.chat_prompt}\n")
            reply = (await self.auxiliary.call_api(payload))["content"].strip()
            assistant_msg: AssistantMessage = {"role": "assistant", "content": reply}
            session.snap.over(message, assistant_msg)
            return reply

    async def call_unit(self, session: Session, event: Event, payload: Payload, extra_prompt: str = ""):
        system_prompt = f"{self.style_prompt}\n{self.chat_prompt}\n{extra_prompt}"
        system_message: SystemMessage = {"role": "system", "content": system_prompt}
        payload["messages"].insert(0, system_message)
        payload["tools"] = self.intro_tools if self.categories else self.intro_tools[1:]
        resp = await self.call_api(payload)
        # 退出条件：不需要额外技能
        if not (tool_calls := resp.get("tool_calls")):
            return resp["content"].strip()
        session.skill_menu = None
        intro_prompt = "".join(msg[0]["content"] for msg in await self.function_call(event, tool_calls) if msg[1])
        # 退出条件：不需要额外技能
        if session.skill_menu:
            toolkit = {"skill_menu"}
            system_message["content"] = f"{self.call_prompt}\n{extra_prompt}\n\n{intro_prompt}"
            result = ""
            for _ in range(100):
                payload["tools"] = [self.manifest[k] for k in toolkit]
                if session.skill_menu:
                    payload["tools"].extend(x for x in self.select_tools(session.skill_menu) if x["function"]["name"] not in toolkit)
                message = await self.call_api(payload)
                if not (tool_calls := message.get("tool_calls")):
                    result = message["content"]
                    break
                payload["messages"].append(message)
                session.skill_menu = None
                for msg, key in await self.function_call(event, tool_calls):
                    toolkit.add(key)
                    payload["messages"].append(msg)
            async with session.snap.lock:
                result = f"{result or '任务执行失败'}\n\n请用你的语气复述以上结果"
                context = [system_message, *session.snap, {"role": "system", "content": result}]
                session.snap.clear()
        else:
            context = payload["messages"]
        system_message["content"] = f"{self.style_prompt}\n\n{intro_prompt}"
        payload = self.auxiliary.build_payload(context)
        return (await self.auxiliary.call_api(payload))["content"].strip()

    async def chat(self, event: Event):
        now = datetime.now()
        timestamp = int(now.timestamp())
        session = self.current_session(event)
        session.extra[event.user_id] = event.nickname
        head = f"{event.nickname}[{now.strftime("%I:%M %p")}]"
        at = "".join(f"@{name} " for user_id in event.at if (name := session.extra.get(user_id))) if event.at else ""
        if "extra_context" in event.properties:
            body = f"@me {at}{event.message}\n{"\n".join(event.extra_context)}"
        elif event.to_me:
            body = f"@me {at}{event.message}"
        else:
            session.silence.append((f"{head}{at}{event.message}", timestamp))
            return
        request = f"{head}{at}{body}"
        if session.lock.locked():
            return await self.aux_reply(session, {"role": "user", "content": self.auxiliary.build_content(request, event.image_list)})
        async with session.lock:
            async with session.snap.lock:
                session.memory_filter(timestamp - self.memory_timeout)
                session.silence_filter(timestamp - self.topic_coldown)
                session.silence.append((request, timestamp))
                message = list(x[0] for x in session.silence)
                message = "\n".join(message)
                if session.step(message) and (summary := await self.summary_context(session)):
                    session.clear()
                    session.silence.append((summary, timestamp))
                session.sync_snap()
                session.snap.over({"role": "user", "content": message}, {"role": "system", "content": "正在执行任务..."})
            content: list[ContentSegment] = []
            if (call := event.call("flat_context")) and (flat_context := await call):
                content.append({"type": "text", "text": "<引用上下文>"})
                for unit in flat_context:
                    if unit["text"]:
                        content.append({"type": "text", "text": f'{unit["nickname"]}:{unit["text"]}'})
                    if unit["images"]:
                        content.extend({"type": "image_url", "image_url": {"url": x}} for x in unit["images"])
                content.append({"type": "text", "text": "</引用上下文>"})
            pure_content: list[ContentSegment] = [{"type": "text", "text": message}]
            pure_content.extend({"type": "image_url", "image_url": {"url": x}} for x in event.image_list)
            content.extend(pure_content)
            session.current_input = {"role": "user", "content": content}  # 注入输入（可修改）
            payload: Payload = {"model": self.model, "messages": [*session, session.current_input]}
            try:
                resp = await self.call_unit(session, event, payload, f"今天的日期是:{now.strftime('%Y年%m月%d日')}")
            except Exception as e:
                logger.exception(e)
                return
            session.over({"role": "user", "content": pure_content}, {"role": "assistant", "content": resp}, timestamp)
            session.current_input = None
            return resp

    def load_plugin(self, name: str | Path, is_path=False):
        package = import_name(name, is_path)
        try:
            plugin = getattr(import_module(package), "__plugin__", None)
            if not isinstance(plugin, SkillCore):
                raise TypeError(f"{package}.__plugin__ must be a subclass of SkillCore")
        except Exception as e:
            logger.exception(f'[{self.name}][LOADING] "{package}" load failed', exc_info=e)
            return
        conflict = self.merge(plugin)
        if conflict:
            logger.error(f'[{self.name}][LOADING] "{package}" conflict with {conflict}')
        else:
            logger.info(f'[{self.name}][LOADING] "{package}" loaded')
        plugin.name = plugin.name or package

    def load_plugins_from_list(self, plugin_list: list[str]):
        for plugin in plugin_list:
            self.load_plugin(plugin)

    def load_plugins_from_dirs(self, plugin_dirs: list[str]):
        for plugin_dir in plugin_dirs:
            plugin_dir = Path(plugin_dir)
            if not plugin_dir.exists():
                plugin_dir.mkdir(parents=True, exist_ok=True)
                continue
            for plugin in list_modules(plugin_dir):
                self.load_plugin(plugin)
