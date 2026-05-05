import time
import json
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from clovers.core import ModuleLoader
from clovers.logger import logger
from clovers_client import Event as BaseEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .api import OpenAIAPI
from .skill import SkillCore
from .session import Session
from .utils import deep_add
from .embedding import SentenceTransformer
from .constants import ON_CHAT, ON_SKILL, SKILL_MENU
from typing import Protocol
from .typing.json_schema import BaseJSONSchemaType
from .typing.message import UserMessage, ContentSegment, ToolCallInfo, ToolMessage
from .config import Config


class Event(BaseEvent, Protocol):
    extra_context: list[str] = []


class CloversAgent(SkillCore, ModuleLoader[SkillCore]):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, scheduler: AsyncIOScheduler, config: Config) -> None:
        ModuleLoader.__init__(self, ["TOOLS"], SkillCore)
        self.name = name
        # clovers 设置
        if config.whitelist:
            whitelist = set(config.whitelist)
            logger.info(f"[{self.name}] 检查规则设置为白名单模式：{whitelist}")
            self.check = lambda e: e.group_id is not None and e.group_id in whitelist
        elif config.blacklist:
            blacklist = set(config.blacklist)
            logger.info(f"[{self.name}] 检查规则设置为黑名单模式：{blacklist}")
            self.check = lambda e: e.group_id is not None and e.group_id not in blacklist
        elif config.console_mode:
            logger.info(f"[{self.name}] 启动控制台模式")
            self.check = lambda e: True
        else:
            logger.info(f"[{self.name}] 已关闭")
            self.check = lambda e: False
        # api
        self._api = OpenAIAPI(async_client, config.api)
        self._apis = {name: OpenAIAPI(async_client, api_config) for name, api_config in config.apis.items()}
        # 文件
        path = Path(config.path)
        self.usage_dir = path / "usages"
        self.payload_dir = path / "payloads"
        self.prompts_dir = path / "prompts"
        self.style_prompt = config.style_prompt
        self.base_prompt = config.base_prompt
        self.chat_prompt = config.chat_prompt
        self.call_prompt = config.call_prompt
        self.wait_prompt = config.wait_prompt
        self.router_prompt = config.router_prompt
        self.summary_prompt = config.summary_prompt
        # 配置
        self.memory_timeout = config.memory_timeout
        self.silence_timeout = config.silence_timeout
        self.memory_size = config.memory_size
        self.silence_size = config.silence_size
        self.router_size = config.router_size
        self.unimportant_size = config.unimportant_size
        self.decouple_length = config.decouple_length
        # 模型设置
        self.usage_counter = {}
        self.sessions: dict[str, Session] = {}
        self.sentence_model = SentenceTransformer(config.sentence_model, cache_folder=config.sentence_model_cache)
        self.scheduler = scheduler
        # 注册技能
        self.skills = tuple()
        self._plugins = config.plugins
        self._plugin_dirs = config.plugin_dirs
        self._skill_dirs = config.skill_dirs
        self._category_schema: BaseJSONSchemaType = {"type": "string"}
        self.scheduler.add_job(self.daily_tasks, trigger="cron", hour="*/8", misfire_grace_time=120)

    def api(self, key: str):
        return self._apis.get(key, self._api)

    @staticmethod
    def check(e: Event) -> bool:
        raise NotImplementedError

    def _load(self, package: str):
        tools = super()._load(package)
        if tools is None:
            return
        conflict = self.merge(tools)
        if conflict:
            logger.error(f'[{self.name}] "{package}" conflict with {conflict}')
            return
        logger.info(f'[{self.name}][TOOLS] "{package}" loaded')

    @staticmethod
    def load_prompt(md: Path, default_prompt: str):
        if md.exists():
            prompt = md.read_text("utf-8")
        else:
            prompt = default_prompt
            md.write_text(prompt, encoding="utf-8")
        return prompt

    def skill_init(self):
        SkillCore.__init__(self)
        self.register(
            ON_CHAT,
            "当前对话为闲聊、讨论与简单提问、或无法分配至其他工具时，调用此方法。",
        )(on_chat)
        self.register(
            ON_SKILL,
            "当用户的指令涉及外部调用时，调用此方法以进入技能执行环境。",
            {"category": self._category_schema},
        )(on_skill)
        self.register(
            SKILL_MENU,
            "获取更多技能，如果助手无法独自完成用户指令，则需要调用此方法获取更多技能。",
            {"category": self._category_schema},
            ON_SKILL,
        )(skill_menu)
        self.load_from_list(self._plugins)
        self.load_from_dirs(self._plugin_dirs)
        self.sync_menu()
        readme_md = self.prompts_dir / "README.md"
        if readme_md.exists():
            return
        if not self.prompts_dir.exists():
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            readme_md.write_text(f"删除 README.md 则会从 {self.prompts_dir.as_posix()} 读取 prompt 配置", encoding="utf-8")
        else:
            logger.info(f"[{self.name}][LOADING PROMPTS]")
        self.style_prompt = self.load_prompt(self.prompts_dir / "STYLE.md", self.style_prompt)
        self.base_prompt = self.load_prompt(self.prompts_dir / "BASE.md", self.base_prompt)
        self.chat_prompt = self.load_prompt(self.prompts_dir / "CHAT.md", self.chat_prompt)
        self.call_prompt = self.load_prompt(self.prompts_dir / "CALL.md", self.call_prompt)
        self.wait_prompt = self.load_prompt(self.prompts_dir / "WAIT.md", self.wait_prompt)
        self.router_prompt = self.load_prompt(self.prompts_dir / "ROUTER.md", self.router_prompt)
        self.summary_prompt = self.load_prompt(self.prompts_dir / "SUMMARY.md", self.summary_prompt)

    def sync_menu(self):
        for skill in self.skills:
            self.delete_skill(*skill)
        paths = (_p for _s_dir in self._skill_dirs if (_dir := Path(_s_dir)).exists() for _p in _dir.iterdir())
        category_set: set[str] = set()
        name_set: set[str] = set()
        for path in paths:
            select = self.load_skill(path)
            if not select:
                continue
            category, name = select
            if category:
                category_set.add(category)
            if name:
                name_set.add(name)
            logger.info(f'[{self.name}][SKILLS] "{category or name}" loaded')
        self.skills = *((category, None) for category in category_set), *((None, name) for name in name_set)
        self._category_schema["description"] = "\n".join(f"{category}: {desc}" for category, desc in self.categories.items())
        self._category_schema["enum"] = list(self.categories.keys())

    def daily_tasks(self):
        timeout = time.time() - self.memory_timeout
        sessions_ids = tuple(s_id for s_id, s in self.sessions.items() if s.recorder[-1][2] < timeout)
        for sessions_id in sessions_ids:
            del self.sessions[sessions_id]
        logger.info(f"[{self.name}][SESSIONS_CLEAR]")
        self.usage_counter
        usage_file = self.usage_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        usage_file.parent.mkdir(parents=True, exist_ok=True)
        with usage_file.open("w", encoding="utf-8") as f:
            json.dump(self.usage_counter, f, indent=4, ensure_ascii=False)
        usage = {k: v.get("total_tokens") for k, v in self.usage_counter.items()}
        logger.info(f"[{self.name}][USAGE_TODAY] {usage}")
        self.usage_counter.clear()

    @staticmethod
    def session_id(event: Event) -> str:
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, event: Event):
        session_id = self.session_id(event)
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(
                self.memory_size,
                self.silence_size,
                self.router_size,
                self.unimportant_size,
                self.decouple_length,
                self.sentence_model,
            )
        return self.sessions[session_id]

    async def summary_context(self, session: Session):
        payload = self._api.build_payload(context=(*session, {"role": "user", "content": self.summary_prompt}))
        try:
            summary = (await self._api.call_api(payload, session.usage_counter))["content"].strip()
            logger.info(f"[{self.name}][SUMMARY]")
            logger.debug(summary)
            return summary
        except Exception as e:
            logger.error(f"[{self.name}][SUMMARY] {e}")
            return

    async def activate_category(self, name: str, event: Event):
        hooks = self.category_hooks.get(name)
        if hooks:
            prompts = []
            for hook in hooks:
                coro = hook(self, event)
                if isinstance(coro, str):
                    prompts.append(coro)
                else:
                    prompts.append(await coro)
            return prompts

    async def activate_skill(self, event: Event, call_info: ToolCallInfo) -> ToolMessage:
        try:
            name = call_info["function"]["name"]
            kwargs = json.loads(call_info["function"]["arguments"])
            if name not in self.invoker:
                return {"role": "tool", "tool_call_id": call_info["id"], "content": f'工具 "{name}" 不存在。'}
            return await self.invoker[name](call_info["id"], self, event, **kwargs)
        except Exception as e:
            return {"role": "tool", "tool_call_id": call_info["id"], "content": f"Error {e}"}

    async def function_call(self, session: Session, event: Event, call_infos: list[ToolCallInfo]):
        messages = await asyncio.gather(*(self.activate_skill(event, x) for x in call_infos))
        session.payload["messages"].extend(messages)

    async def call_unit(self, session: Session, event: Event):
        message = await session.api.call_api(session.payload, session.usage_counter)
        if not (tool_calls := message.get("tool_calls")):
            return message["content"]
        session.payload["messages"].append(message)
        await self.function_call(session, event, tool_calls)
        return session.result

    async def execute_turn(self, session: Session, event: Event):
        for _ in range(40):
            if result := await self.call_unit(session, event):
                return result
        payload_file = self.payload_dir / self.session_id(event) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        payload_file.parent.mkdir(parents=True, exist_ok=True)
        with payload_file.open("w", encoding="utf-8") as f:
            json.dump(session.payload, f, indent=4, ensure_ascii=False)
        raise TimeoutError(f"Maximum tool call chain length exceeded, payload saved to: {payload_file.name}")

    async def router(self, session: Session, event: Event):
        router_prompt = f"<system>\n{self.router_prompt}\n{self.base_prompt}\n</system>"
        current_input = session.current_input.copy()
        current_input.append({"type": "text", "text": router_prompt})
        api = self.api("router")
        payload = api.build_payload((*session.router_context, {"role": "user", "content": current_input}))
        payload["tools"] = self.intro_tools
        try:
            message = await api.call_api(payload, session.usage_counter)
            if "tool_calls" not in message:
                raise ValueError(f"message must contain tool_calls, but got {message}")
            call_info = message["tool_calls"][0]
            category = call_info["function"]["name"]
            intro = self.intro_invoker[category]
            kwargs = json.loads(call_info["function"]["arguments"])
            logger.info(f"[{self.name}][ROUTER] {category} {kwargs}")
            coro = intro(self, event, *kwargs)
            if not isinstance(coro, str):
                await coro
        except Exception as e:
            logger.warning(f"[{self.name}][ROUTER] {ON_CHAT} {e}")
            category = ON_CHAT
            await on_chat(self, event)

        session.activate()
        if prompts := await self.activate_category(category, event):
            prompt = "\n".join(x for x in (prompts) if x)
            if prompt:
                session.unit_prompts.append(prompt)
        return await self.execute_turn(session, event)

    async def wait_chat(self, session: Session, message: str):
        unit_prompt = "\n".join(x for x in (session.unit_prompts) if x)
        prompts = (self.style_prompt, self.base_prompt, self.chat_prompt, unit_prompt)

        payload = self._api.build_payload(
            (*session, {"role": "user", "content": f"{message}\n<system>\n{self.wait_prompt}\n</system>"}),
            "\n".join(x for x in prompts if x),
        )
        try:
            return (await self._api.call_api(payload, session.usage_counter))["content"].strip()
        except Exception as e:
            logger.exception(e)

    async def execute_chat(self, session: Session, event: Event, message: str):
        quote_content: list[ContentSegment] = []
        if (call := event.call("flat_context")) and (flat_context := await call):
            quote_content.append({"type": "text", "text": "<quote>"})
            for unit in flat_context:
                if unit["text"]:
                    quote_content.append({"type": "text", "text": f'{unit["nickname"]}:{unit["text"]}'})
                if unit["images"]:
                    quote_content.extend({"type": "image_url", "image_url": {"url": x}} for x in unit["images"])
            quote_content.append({"type": "text", "text": "</quote>"})
        chat_content: list[ContentSegment] = [{"type": "text", "text": message}]
        image_list = await asyncio.gather(*map(self._api.download_url, event.image_list))
        chat_content.extend({"type": "image_url", "image_url": {"url": x}} for x in image_list if x)
        session.current_input = [*quote_content, *chat_content]
        session.usage_counter.clear()
        try:
            result = await self.router(session, event)
        except Exception as e:
            logger.exception(e)
            return
        finally:
            session.inactivate()
        session.over(
            {"role": "user", "content": chat_content},
            {"role": "assistant", "content": result},
            session.silence_recorder[-1][1],
        )
        return result

    async def chat(self, event: Event):
        timestamp = time.time()
        now = datetime.fromtimestamp(timestamp)
        session = self.current_session(event)
        if "nicknames" not in session.extra:
            session.extra["nicknames"] = {}
        session.extra["nicknames"][event.user_id] = event.nickname
        head = f"{event.nickname}[{now.strftime("%I:%M %p")}]"
        at = "".join(f"@{name} " for user_id in event.at if (name := session.extra.get(user_id))) if event.at else ""
        if "extra_context" in event.properties:
            body = f"@me {at}{event.message}\n{"\n".join(event.extra_context)}"
        elif event.to_me:
            body = f"@me {at}{event.message}"
        else:
            session.silence_recorder.append((f"{head}{at}{event.message}", timestamp))
            return
        request = f"{head}{at}{body}"
        session.memory_filter(timestamp - self.memory_timeout)
        session.silence_filter(timestamp - self.silence_timeout)
        session.silence_recorder.append((request, timestamp))
        session.unit_prompts.append(f"Today:{now.strftime('%Y-%m-%d')}")
        message = list(x[0] for x in session.silence_recorder)
        message = "\n".join(message)
        if session.lock.locked():
            async with session.wait_lock:
                result = await self.wait_chat(session, message)
        else:
            async with session.lock:
                if session.step(body) and (summary := await self.summary_context(session)):
                    session.clear()
                    session.silence_recorder.append((summary, timestamp))
                result = await self.execute_chat(session, event, message)
        deep_add(self.usage_counter, session.usage_counter)
        usage = {k: v.get("total_tokens") for k, v in session.usage_counter.items()}
        logger.info(f"[{self.name}][USAGE] {usage}")
        return result


async def skill_menu(agent: CloversAgent, event: Event, category: str):
    session = agent.current_session(event)
    if prompts := await agent.activate_category(category, event):
        prompt = "\n".join(x for x in (prompts) if x)
    else:
        prompt = ""
    if "tools" not in session.payload:
        session.payload["tools"] = [*agent.select_tools(ON_SKILL), *agent.select_tools(category)]
    else:
        used = {tool["function"]["name"] for tool in session.payload["tools"]}
        new_tools = agent.select_tools(category)
        session.payload["tools"].extend(x for x in new_tools if x["function"]["name"] not in used)
    return prompt or f"已获取技能：{category}"


async def on_skill(agent: CloversAgent, event: Event, category: str):
    session = agent.current_session(event)
    session.api = agent.api("skill")
    unit_prompt = "\n".join(x for x in (session.unit_prompts) if x)
    prompts = (agent.call_prompt, agent.base_prompt, unit_prompt)
    session.payload = session.api.build_payload(session, "\n".join(x for x in prompts if x))
    session.payload["tools"] = agent.select_tools(ON_SKILL).copy()
    skill_prompt = await skill_menu(agent, event, category)
    if skill_prompt:
        session.system_message["content"] += f"\n{skill_prompt}"
    return ""


async def on_chat(agent: CloversAgent, event: Event):
    session = agent.current_session(event)
    session.api = agent.api("chat")
    unit_prompt = "\n".join(x for x in (session.unit_prompts) if x)
    prompts = (agent.style_prompt, agent.base_prompt, agent.chat_prompt, unit_prompt)
    session.payload = session.api.build_payload(session, "\n".join(x for x in prompts if x))
    session.payload["tools"] = agent.select_tools(ON_CHAT).copy()
    return ""
