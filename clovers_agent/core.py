import json
import asyncio
import httpx
import time
from datetime import datetime
from pathlib import Path
from importlib import import_module
from clovers.utils import import_name, list_modules
from clovers.logger import logger
from collections import deque
from collections.abc import Iterable, Callable, Coroutine
from typing import Literal, TypedDict, Any, Concatenate
from .typing import Event, ChatMessage, ToolMessage, Payload, FunctionToolInfo
from .typing.json_schema import JSONSchemaType
from .config import Config

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
                content = await func(agent, event, **kwargs)
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


class Session:
    type UserMessage = ChatMessage[Literal["user"]]
    type AssistantMessage = ChatMessage[Literal["assistant"]]
    type Timestamp = int | float

    records: deque[tuple[UserMessage, AssistantMessage, Timestamp]]
    silence: deque[tuple[str, Timestamp]]

    def __init__(self, size: int) -> None:
        self.records = deque(maxlen=size)
        self.silence = deque()
        self.running: bool = False

    def memory_filter(self, timeout: int | float):
        """过滤记忆"""
        while self.records and (self.records[0][2] <= timeout):
            self.records.popleft()
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


class CloversAgent(ToolManager):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient) -> None:
        super().__init__(name)
        self.async_client = async_client
        config = Config.sync_config()
        self.model = config.model
        self.url = f"{config.url.rstrip("/")}/chat/completions"
        self.headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        self.style_prompt = config.style_prompt
        self.call_prompt = config.call_prompt
        self._memory_size = config.memory_size
        self.memory_timeout = config.memory_timeout
        self.topic_coldown = config.topic_coldown
        self.sessions: dict[str, Session] = {}
        self.current_input: Session.UserMessage | None = None
        self.load_plugins_from_list(config.plugins)
        self.load_plugins_from_dirs(config.plugin_dirs)
        self.toolmap: dict[str, FunctionToolInfo] = {}
        if self.skill_keywords:
            skill_keywords = list(self.skill_keywords)
            self.tool(
                "skill_menu",
                "获取更多技能，如果assistant无法单独完成用户指令，则需要调用此方法获取更多技能。",
                {"category": {"type": "string", "description": "选择需要的技能关键词", "enum": skill_keywords}},
            )(self.skill_menu)
            logger.info(f"[{self.name}][LOAD SKILLS] {skill_keywords}")
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
                tip += f"，{info}"
        return tip

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

    async def call_api(self, payload: Payload):
        resp = await self.async_client.post(self.url, headers=self.headers, json=payload)
        try:
            resp.raise_for_status()
        except:
            logger.error(json.dumps(payload, indent=4, ensure_ascii=False))
            raise
        return resp.json()["choices"][0]["message"]

    async def function_call(self, event: Event, call_infos: list[dict]) -> list[tuple[ToolMessage, str]]:
        task_queue = []
        for call_info in call_infos:
            func = self.functions[call_info["function"]["name"]]
            kwargs = json.loads(call_info["function"]["arguments"])
            task_queue.append(func(call_info["id"], self, event, **kwargs))
        return await asyncio.gather(*task_queue)

    def build_payload(self, context: Iterable[ChatMessage] | None = None, system_prompt: str | None = None) -> Payload:
        payload: Payload = {"model": self.model, "messages": []}
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        if context:
            payload["messages"].extend(context)
        return payload

    @staticmethod
    def session_id(event: Event):
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, event: Event):
        session_id = self.session_id(event)
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(self._memory_size)
        return self.sessions[session_id]

    def select_tools(self, keyword: str):
        return [d["info"] for d in self.extra_tools if keyword in d["keywords"]]

    async def summary_context(self, event: Event) -> str:
        session = self.current_session(event)
        payload = self.build_payload(context=session.context)
        payload["messages"].append({"role": "user", "content": "请对以上对话进行深度总结。"})
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "topic_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string", "description": "对话内容的精炼总结，保留核心内容和结论。"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }
        return json.loads((await self.call_api(payload))["content"])["summary"].strip()

    async def call_unit(self, event: Event, payload: Payload):
        system_prompt = f"{self.style_prompt}\nDate:{datetime.now().strftime('%m-%d')}"
        system_message: ChatMessage = {"role": "system", "content": system_prompt}
        payload["messages"].insert(0, system_message)
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
            system_message["content"] = f"{self.call_prompt}\n\n{intro_prompt}"
            for _ in range(30):
                payload["tools"] = [self.toolmap[k] for k in used_tools]
                if event.skill_menu:
                    select_tools = self.select_tools(event.skill_menu)
                    payload["tools"].extend(tool for tool in select_tools if tool["function"]["name"] not in used_tools)
                message = await self.call_api(payload)
                if not (tool_calls := message.get("tool_calls")):
                    break
                payload["messages"].append(message)
                event.properties["skill_menu"] = ""
                for msg, key in await self.function_call(event, tool_calls):
                    used_tools.add(key)
                    payload["messages"].append(msg)
        del payload["tools"]
        system_message["content"] = f"{system_prompt}\n\n{intro_prompt}"
        return (await self.call_api(payload))["content"].strip()

    async def chat(self, event: Event):
        session = self.current_session(event)
        payload: Payload = {"model": self.model, "messages": []}
        now = int(time.time())
        session.memory_filter(now - self.memory_timeout)
        if len(session.records) >= 5 and session.records[0][2] < now - self.memory_timeout:
            summary = await self.summary_context(event)
            logger.debug(f"[{self.name}][SUMMARY] {summary}")
            session.clear()
            session.silence.append((summary, now))
        payload["messages"].extend(session.context)
        content = self.build_content("\n".join(x[0] for x in session.silence), event.image_list)
        self.current_input = {"role": "user", "content": content}
        payload["messages"].append(self.current_input)
        try:
            resp = await self.call_unit(event, payload)
        except Exception as e:
            logger.exception(e, exc_info=e)
            return
        session.over(self.current_input, {"role": "assistant", "content": resp}, now)
        self.current_input = None
        return resp
