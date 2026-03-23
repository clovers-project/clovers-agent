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
from typing import TypedDict, Any, Concatenate
from .typing import Event, Message, SystemMessage, UserMessage, AssistantMessage, ToolMessage, Payload, FunctionToolInfo
from .typing.json_schema import JSONSchemaType
from .config import OpenAIConfig, Config

type AgentFunction[**P] = Callable[Concatenate["CloversAgent", Any, P], Coroutine[Any, Any, str]]
type WrappedAgentFunction[**P] = Callable[Concatenate[str, "CloversAgent", Any, P], Coroutine[Any, Any, tuple[ToolMessage, str]]]


class ToolManager:
    type Parameters = dict[str, JSONSchemaType]

    class ExtraToolsType(TypedDict):
        info: FunctionToolInfo
        keywords: set[str]

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.intro_tools: list[FunctionToolInfo] = []
        self.skill_keywords: set[str] = set()
        self.skill_hooks: dict[str, AgentFunction] = {}
        self.extra_tools: list[ToolManager.ExtraToolsType] = []
        self.functions: dict[str, WrappedAgentFunction] = {}

    def tool(self, name: str, description: str, parameters: Parameters | None, keywords: Iterable[str] | None = None):
        if name in self.functions:
            raise ValueError(f"Tool {name} already exists.")
        info: FunctionToolInfo = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
            },
        }
        if parameters:
            info["function"]["parameters"] = {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys()),
            }
        if not keywords:
            self.intro_tools.append(info)
        else:
            keywords = set(keywords)
            self.skill_keywords.update(keywords)
            self.extra_tools.append({"info": info, "keywords": keywords})

        def decorator(func: AgentFunction) -> WrappedAgentFunction:
            async def wrapper(tool_call_id, agent: CloversAgent, event, /, **kwargs):
                logger.debug(f"[{agent.name}][TOOL CALL][{name}] called")
                try:
                    content = await func(agent, event, **kwargs)
                except Exception as e:
                    logger.exception(e)
                    content = "Error"
                message: ToolMessage = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                return message, name

            self.functions[name] = wrapper
            return wrapper

        return decorator

    def on_skill(self, category: str):
        def decorator(func: AgentFunction) -> AgentFunction:
            self.skill_hooks[category] = func
            return func

        return decorator

    def mixin(self, plugin: "ToolManager"):
        conflict = plugin.functions.keys() & self.functions.keys()
        if conflict:
            return conflict
        self.intro_tools.extend(plugin.intro_tools)
        self.skill_keywords.update(plugin.skill_keywords)
        self.extra_tools.extend(plugin.extra_tools)
        self.functions.update(plugin.functions)
        self.skill_hooks.update(plugin.skill_hooks)

    def load_plugin(self, name: str | Path, is_path=False):
        """加载 clovers-agent 插件

        Args:
            name (str | Path): 插件的包名或路径
            is_path (bool, optional): 是否为路径
        """
        package = import_name(name, is_path)
        try:
            plugin = getattr(import_module(package), "__plugin__", None)
            assert isinstance(plugin, ToolManager)
        except Exception as e:
            logger.exception(f'[{self.name}][loading plugin] "{package}" load failed', exc_info=e)
            return
        conflict = self.mixin(plugin)
        if conflict:
            logger.error(f'[{self.name}][loading plugin] "{package}" conflict with {conflict}')
        else:
            logger.info(f'[{self.name}][loading plugin] "{package}" loaded')
        plugin.name = plugin.name or package

    def load_plugins_from_list(self, plugin_list: list[str]):
        """从包名列表加载插件

        Args:
            plugin_list (list[str]): 插件的包名列表
        """
        for plugin in plugin_list:
            self.load_plugin(plugin)

    def load_plugins_from_dirs(self, plugin_dirs: list[str]):
        """从本地目录列表加载插件

        Args:
            plugin_dirs (list[str]): 插件的目录列表
        """
        for plugin_dir in plugin_dirs:
            plugin_dir = Path(plugin_dir)
            if not plugin_dir.exists():
                plugin_dir.mkdir(parents=True, exist_ok=True)
                continue
            for plugin in list_modules(plugin_dir):
                self.load_plugin(plugin)


class OpenAIAPI:
    def __init__(self, async_client: httpx.AsyncClient, config: OpenAIConfig) -> None:
        self.async_client = async_client
        self.url = f"{config.url.rstrip("/")}/chat/completions"
        self.model = config.model
        self.headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        self.extra_body = config.extra_body

    @staticmethod
    def build_content(text: str, image_list: list[str] | None):
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


class Session:
    type Timestamp = int | float

    records: deque[tuple[UserMessage, AssistantMessage, Timestamp]]
    silence: deque[tuple[str, Timestamp]]

    def __init__(self, size: int) -> None:
        self.records = deque(maxlen=size)
        self.silence = deque()
        self.lock = asyncio.Lock()

    def memory_filter(self, timeout: int | float):
        """过滤记忆"""
        while self.records and (self.records[0][2] <= timeout):
            self.records.popleft()

    def silence_filter(self, timeout: int | float):
        """过滤静默记录群聊上下文"""
        while self.silence and (self.silence[0][1] <= timeout):
            self.silence.popleft()

    @property
    def context(self):
        for request, reply, _ in self.records:
            yield request
            yield reply

    def over(self, request: UserMessage, reply: AssistantMessage, timestamp: int | float):
        self.records.append((request, reply, timestamp))
        self.silence.clear()

    def clear(self):
        self.records.clear()
        self.silence.clear()


class CloversAgent(ToolManager, OpenAIAPI):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, config: Config) -> None:
        ToolManager.__init__(self, name)
        OpenAIAPI.__init__(self, async_client, config.primary)
        self.auxiliary = OpenAIAPI(async_client, config.auxiliary) if config.auxiliary is not None else self
        self.style_prompt = config.style_prompt
        self.chat_prompt = config.chat_prompt
        self.call_prompt = config.call_prompt
        self.memory_size = config.memory_size
        self.memory_timeout = config.memory_timeout
        self.topic_coldown = config.topic_coldown
        self.sessions: dict[str, Session] = {}
        self.current_input: UserMessage | None = None
        self.load_plugins_from_list(config.plugins)
        self.load_plugins_from_dirs(config.plugin_dirs)
        if self.skill_keywords:
            skill_keywords = list(self.skill_keywords)
            self.tool(
                "skill_menu",
                "获取更多技能，如果assistant无法单独完成用户指令，则需要调用此方法获取更多技能。",
                {"category": {"type": "string", "description": "选择需要的技能关键词", "enum": skill_keywords}},
            )(self.skill_menu)
            logger.info(f"[{self.name}][LOAD SKILLS] {skill_keywords}")
        self.toolmap: dict[str, FunctionToolInfo] = {}
        for tool in self.intro_tools:
            self.toolmap[tool["function"]["name"]] = tool
        for tool in self.extra_tools:
            self.toolmap[tool["info"]["function"]["name"]] = tool["info"]

    @staticmethod
    async def skill_menu(agent: "CloversAgent", event: Event, category: str):
        tip = f"已获取技能：{category}"
        event.properties["skill_menu"] = category
        hook = agent.skill_hooks.get(category)
        if hook:
            info = await hook(agent, event)
            if info:
                tip += f"\n{info}"
        return tip

    async def function_call(self, event: Event, call_infos: list[dict]) -> list[tuple[ToolMessage, str]]:
        task_queue = []
        for call_info in call_infos:
            func = self.functions[call_info["function"]["name"]]
            kwargs = json.loads(call_info["function"]["arguments"])
            task_queue.append(func(call_info["id"], self, event, **kwargs))
        return await asyncio.gather(*task_queue)

    def current_session(self, event: Event):
        session_id = event.group_id or f"private-{event.user_id}"
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(self.memory_size)
        return self.sessions[session_id]

    def select_tools(self, keyword: str):
        return [d["info"] for d in self.extra_tools if keyword in d["keywords"]]

    async def summary_context(self, event: Event) -> str:
        session = self.current_session(event)
        payload = self.build_payload(context=session.context)
        payload["messages"].append(
            {"role": "user", "content": "对以上对话进行深度详细总结，保留核心内容和结论，禁止输出除总结外的其他内容。"}
        )
        return (await self.auxiliary.call_api(payload))["content"].strip()

    async def call_unit(self, event: Event, payload: Payload):
        date_prompt = f"今天的日期是:{datetime.now().strftime('%Y年%m月%d日')}"
        system_prompt = f"{self.style_prompt}\n{self.chat_prompt}\n{date_prompt}"
        system_message: SystemMessage = {"role": "system", "content": system_prompt}
        payload["messages"].insert(0, system_message)
        mark = len(payload["messages"])
        payload["tools"] = self.intro_tools
        resp = await self.call_api(payload)
        # 退出条件：不需要额外技能
        if not (tool_calls := resp.get("tool_calls")):
            return resp["content"].strip()
        event.properties["skill_menu"] = ""
        intro_prompt = "".join(msg[0]["content"] for msg in await self.function_call(event, tool_calls) if msg[1])
        # 退出条件：不需要额外技能
        if event.skill_menu:
            used_tools = {"skill_menu"}
            system_message["content"] = f"{self.call_prompt}\n{date_prompt}\n\n{intro_prompt}"
            for _ in range(100):
                payload["tools"] = [self.toolmap[k] for k in used_tools]
                if event.skill_menu:
                    select_tools = self.select_tools(event.skill_menu)
                    payload["tools"].extend(tool for tool in select_tools if tool["function"]["name"] not in used_tools)
                message = await self.call_api(payload)
                if not (tool_calls := message.get("tool_calls")):
                    payload["messages"] = payload["messages"][:mark]
                    intro_prompt = f"接下来你需要用你的语气复述如下内容：\n{message["content"]}"
                    break
                payload["messages"].append(message)
                event.properties["skill_menu"] = ""
                for msg, key in await self.function_call(event, tool_calls):
                    used_tools.add(key)
                    payload["messages"].append(msg)
        system_message["content"] = f"{self.style_prompt}\n\n{intro_prompt}"
        payload = self.auxiliary.build_payload(context=payload["messages"])
        return (await self.auxiliary.call_api(payload))["content"].strip()

    async def chat(self, event: Event):
        now = datetime.now()
        timestamp = int(now.timestamp())
        session = self.current_session(event)
        if not event.to_me:
            session.silence.append((f"{event.nickname}[{now.strftime("%I:%M %p")}]{event.message}", timestamp))
            return
        if session.lock.locked():
            if not session.silence:
                return
            payload = self.auxiliary.build_payload(
                ({"role": "user", "content": event.message},),
                f"{self.style_prompt}\n{self.chat_prompt}\n"
                f"**特别提示**\n你正在处理上一个指令：{session.silence[-1][0]}\n"
                "如果用户进行了简单的提问或不依赖上下文的聊天，请正常回复。\n"
                "如果用户提出了新任务、追问或依赖上下文的聊天，请告知用户你在忙，请稍等。",
            )
            return (await self.auxiliary.call_api(payload))["content"].strip()
        async with session.lock:
            session.memory_filter(timestamp - self.memory_timeout)
            session.silence_filter(topic_timeout := (timestamp - self.topic_coldown))
            session.silence.append((f"{event.nickname}[{now.strftime("%I:%M %p")}]@me {event.message}", timestamp))
            if len(session.records) >= 5 and session.records[-1][2] < topic_timeout:
                summary = await self.summary_context(event)
                logger.debug(f"[{self.name}][SUMMARY] {summary}")
                session.records.clear()
                session.silence.appendleft((summary, topic_timeout))
            message = list(x[0] for x in session.silence)
            if "extra_context" in event.properties and event.extra_context:
                message.extend(event.extra_context)
            content = self.build_content("\n".join(message), event.image_list)
            self.current_input = {"role": "user", "content": content}  # 注入输入（可修改）
            payload: Payload = {"model": self.model, "messages": [*session.context, self.current_input]}
            if (call := event.call("flat_context")) and (flat_context := await call):
                if isinstance(content, str):
                    self.current_input["content"] = [{"type": "text", "text": content}]
                assert isinstance(self.current_input["content"], list)
                self.current_input["content"].append({"type": "text", "text": "<引用上下文>"})
                for unit in flat_context:
                    self.current_input["content"].append({"type": "text", "text": f'{unit["nickname"]}:{unit["text"]}'})
                    if unit["images"]:
                        self.current_input["content"].extend({"type": "image_url", "image_url": {"url": x}} for x in unit["images"])
                self.current_input["content"].append({"type": "text", "text": "</引用上下文>"})
            try:
                resp = await self.call_unit(event, payload)
            except Exception as e:
                logger.exception(e)
                return
            session.over({"role": "user", "content": content}, {"role": "assistant", "content": resp}, timestamp)
            self.current_input = None
            return resp
