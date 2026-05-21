import time
import json
import asyncio
import httpx
from pathlib import Path
from itertools import islice
from datetime import datetime
from clovers.core import ModuleLoader
from clovers.logger import logger
from clovers_client import Event as BaseEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .api import OpenAIAPI, HybridOpenAIAPI
from .skill import SkillCore, Parameters
from .session import Session
from .embedding import SentenceTransformer
from .utils import deep_add
from typing import Protocol, Literal, override
from .typing import UserMessage, ToolMessage, ToolCallInfo
from .typing.message import MultimodalContent
from .typing.json_schema import BaseJSONSchemaType
from .config import HybridOpenAIConfig, CONFIG, PROMPTS
from .constants import (
    SYSTEM_TAG,
    USER_TAG,
    ASSISTANT_TAG,
    ON_CHAT,
    ON_CHAT_DESC,
    SKILL_MENU,
    SKILL_MENU_DESC,
    ACTIVE_REPLY,
    ACTIVE_REPLY_DESC,
    BUILTIN_CATEGORY,
    GET_IMAGE_BY_ID_INFO,
)


class Event(BaseEvent, Protocol):
    extra_context: list[str] = []


class CloversAgent(SkillCore, ModuleLoader[SkillCore]):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, scheduler: AsyncIOScheduler) -> None:
        ModuleLoader.__init__(self, ["TOOLS"], SkillCore)
        self.name = name
        self.async_client = async_client
        # 核心
        self._api = self.creat_api(CONFIG.api)
        self._apis = {name: self.creat_api(api_config) for name, api_config in CONFIG.apis.items()}
        self.sentence_model = SentenceTransformer(CONFIG.sentence_model, cache_folder=CONFIG.sentence_model_cache)
        self.scheduler = scheduler
        # 文件
        path = Path(CONFIG.path)
        self.usage_dir = path / "usages"
        self.payload_dir = path / "payloads"
        self.prompts_dir = path / "prompts"
        self.init_prompts()
        # 状态
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.sessions: dict[str, Session] = {}
        usage_file = self.usage_file
        if usage_file.exists():
            with usage_file.open("r", encoding="utf-8") as f:
                self.usage_counter = json.load(f)
        else:
            self.usage_counter = {}
        # 配置
        self.call_depth = CONFIG.call_depth
        self.wait_coldown = CONFIG.wait_coldown
        self.active_coldown = CONFIG.active_coldown
        self.dormant_timeout = CONFIG.dormant_timeout
        self.active_context_size = CONFIG.active_context_size
        # 技能
        self.skills = tuple()
        self._plugins = CONFIG.plugins
        self._plugin_dirs = CONFIG.plugin_dirs
        self._skill_dirs = CONFIG.skill_dirs
        self.skill_parameters: Parameters[Literal["category"], BaseJSONSchemaType] = {"category": {"type": "string"}}
        self.scheduler.add_job(self.daily_tasks, trigger="cron", hour=2, misfire_grace_time=3600)

    @property
    def usage_file(self):
        return self.usage_dir / f"{self.today}.json"

    @property
    def style_prompt(self) -> str:
        """Agent 人物设定核心提示"""
        return "\n".join(x for x in (self._style_prompt, self.base_prompt) if x)

    @property
    def chat_prompt(self) -> str:
        """Agent 聊天核心提示"""
        return "\n".join(x for x in (self._style_prompt, self.base_prompt, self._chat_prompt) if x)

    def creat_api(self, config: HybridOpenAIConfig):
        if config.vision:
            return HybridOpenAIAPI(self.async_client, config)
        else:
            return OpenAIAPI(self.async_client, config)

    def api(self, key: str):
        return self._apis.get(key, self._api)

    @override
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
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text(prompt, encoding="utf-8")
        return prompt

    def init_prompts(self):
        logger.info(f"[{self.name}][LOADING PROMPTS]")
        self.base_prompt = self.load_prompt(self.prompts_dir / "BASE.md", PROMPTS.base_prompt)
        self.router_prompt = self.load_prompt(self.prompts_dir / "ROUTER.md", PROMPTS.router_prompt)
        self._style_prompt = self.load_prompt(self.prompts_dir / "STYLE.md", PROMPTS.style_prompt)
        self._chat_prompt = self.load_prompt(self.prompts_dir / "CHAT.md", PROMPTS.chat_prompt)
        self.wait_prompt = self.load_prompt(self.prompts_dir / "WAIT.md", PROMPTS.wait_prompt)
        self.summary_prompt = self.load_prompt(self.prompts_dir / "SUMMARY.md", PROMPTS.summary_prompt)
        self.active_decision_prompt = self.load_prompt(self.prompts_dir / "ACTIVE_DECISION.md", PROMPTS.active_decision_prompt)
        self.active_reply_prompt = self.load_prompt(self.prompts_dir / "ACTIVE_REPLY.md", PROMPTS.active_reply_prompt)

    def skill_init(self):
        SkillCore.__init__(self)
        self.register(ON_CHAT, ON_CHAT_DESC)(on_chat)
        self.register(SKILL_MENU, SKILL_MENU_DESC, self.skill_parameters, BUILTIN_CATEGORY)(skill_menu)
        self.category_decorator(GET_IMAGE_BY_ID_INFO, BUILTIN_CATEGORY)(view_id_image)
        self.load_from_list(self._plugins)
        self.load_from_dirs(self._plugin_dirs)
        self.sync_menu()

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
        self.skill_parameters["category"]["description"] = "\n".join(f"{category}: {desc}" for category, desc in self.categories.items())
        self.skill_parameters["category"]["enum"] = list(self.categories.keys())

    def daily_tasks(self):
        timestamp = time.time()
        sessions_ids = tuple(s_id for s_id, s in self.sessions.items() if s.last_active_time < timestamp - s.memory_timeout)
        for sessions_id in sessions_ids:
            del self.sessions[sessions_id]
        logger.info(f"[{self.name}][SESSIONS_CLEAR]")
        usage = {k: v.get("total_tokens") for k, v in self.usage_counter.items()}
        logger.info(f"[{self.name}][USAGE_TODAY] {usage}")
        self.today = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        self.usage_counter.clear()

    @staticmethod
    def session_id(event: Event) -> str:
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, event: Event):
        session_id = self.session_id(event)
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(self.sentence_model)
        return self.sessions[session_id]

    async def summary_context(self, session: Session):
        api = self.api("summary")
        payload = api.build_payload((*session, {"role": "user", "content": self.summary_prompt}))
        try:
            summary = (await api.call_api(payload, session.usage_counter))["content"].strip()
            logger.info(f"[{self.name}][SUMMARY]")
            logger.debug(summary)
            return summary
        except Exception as e:
            logger.error(f"[{self.name}][SUMMARY] {e}")
            return

    async def activate_category(self, name: str, event: Event) -> list[str] | None:
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

    async def call_unit(self, session: Session, event: Event):
        message = await session.api.call_api(session.payload, session.usage_counter)
        if not (tool_calls := message.get("tool_calls")):
            return message["content"]
        session.payload["messages"].append(message)
        messages = await asyncio.gather(*(self.activate_skill(event, x) for x in tool_calls))
        session.payload["messages"].extend(messages)
        return session.result

    async def execute_turn(self, session: Session, event: Event):
        for _ in range(self.call_depth):
            if result := await self.call_unit(session, event):
                return result
        payload_file = self.payload_dir / self.session_id(event) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        payload_file.parent.mkdir(parents=True, exist_ok=True)
        with payload_file.open("w", encoding="utf-8") as f:
            json.dump(session.payload, f, indent=4, ensure_ascii=False)
        raise TimeoutError(f"Maximum tool call chain length exceeded, payload saved to: {payload_file.name}")

    async def router(self, session: Session, event: Event):
        if len(self.intro_tools) < 2:
            return await on_chat(self, event)
        try:
            api = self.api("router")
            record = "".join(f"{a}\n{ASSISTANT_TAG.format(b)}\n" for a, b in session.router_recorder)
            payload = api.build_payload(
                ({"role": "user", "content": [{"type": "text", "text": record}, *session.current_input]},),
                self.router_prompt,
            )
            payload["tools"] = self.intro_tools
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
            category = await on_chat(self, event)
        return category

    async def active_decision(self, session: Session, timestamp: float):
        silence_duration = timestamp - session.last_active_time
        if silence_duration < self.active_coldown:
            return False
        if silence_duration > self.dormant_timeout:
            return True
        contents = [x for x, _ in islice(reversed(session.silence_recorder), self.active_context_size)]
        if len(contents) < self.active_context_size:
            return False
        message: UserMessage = {"role": "user", "content": "\n".join(reversed(contents))}
        api = self.api("decision")
        payload = api.build_payload((message,), "\n".join(self.active_decision_prompt))
        payload["tools"] = [{"type": "function", "function": {"name": ACTIVE_REPLY, "description": ACTIVE_REPLY_DESC}}]
        try:
            resp = await api.call_api(payload, session.usage_counter)
            return silence_duration > self.active_coldown and "tool_calls" in resp
        except Exception as e:
            logger.exception(e)
            return False

    async def active_reply(self, session: Session, content: str):
        api = self.api("active")
        payload = api.build_payload(
            ({"role": "user", "content": content},),
            "\n".join(x for x in (self.style_prompt, self.active_reply_prompt) if x),
        )
        logger.info(f"[{self.name}][ACTIVE_REPLY]")
        resp = await api.call_api(payload, session.usage_counter)
        return resp["content"].strip()

    async def wait_chat(self, session: Session, content: str):
        api = self.api("wait")
        wait_prompt = f"{content}\n{SYSTEM_TAG.format(self.wait_prompt)}"
        payload = api.build_payload((*session, {"role": "user", "content": wait_prompt}), self.style_prompt)
        resp = await api.call_api(payload, session.usage_counter)
        return resp["content"].strip()

    async def handle_chat(self, session: Session, event: Event):
        timestamp = time.time()
        now = datetime.fromtimestamp(timestamp)
        if "nicknames" not in session.extra:
            session.extra["nicknames"] = {}
        nicknames = session.extra["nicknames"]
        nicknames[event.user_id] = event.nickname
        at = "".join(f"@{name} " for user_id in event.at if (name := nicknames.get(user_id))) if event.at else ""
        message = event.message
        if "extra_context" in event.properties:
            body = f"@assistant {at}{message}\n{"\n".join(event.extra_context)}"
        elif event.to_me:
            body = f"@assistant {at}{message}"
        else:
            body = f"{at}{message}"
            session.silence_recorder.append((USER_TAG.format(event.nickname, body), timestamp))
            if event.at or not await self.active_decision(session, timestamp):
                return
            async with session.execute_lock, session.wait_lock:
                content = "\n".join(x for x, _ in session.silence_recorder)
                try:
                    result = await self.active_reply(session, content)
                except Exception as e:
                    logger.exception(e)
                    return
                session.last_active_time = timestamp
                session.over(content, {"role": "assistant", "content": result}, timestamp)
                return result
        session.refresh(timestamp)
        session.silence_recorder.append((USER_TAG.format(event.nickname, body), timestamp))
        content = "\n".join(x for x, _ in session.silence_recorder)
        if session.execute_lock.locked():
            if session.wait_lock.locked():
                return
            silence_duration = timestamp - session.last_active_time
            if silence_duration < self.wait_coldown:
                return
            async with session.wait_lock:
                try:
                    result = await self.wait_chat(session, content)
                except Exception as e:
                    logger.exception(e)
                    return
                session.last_active_time = timestamp
                return result
        async with session.execute_lock:
            if session.step(body) and (summary := await self.summary_context(session)):
                session.clear()
                session.silence_recorder.append((summary, timestamp))
            quote_content: MultimodalContent = []
            if (call := event.call("flat_context")) and (flat_context := await call):
                quote_content.append({"type": "text", "text": "<quote>\n"})
                for unit in flat_context:
                    if unit["text"]:
                        quote_content.append({"type": "text", "text": f'{unit["nickname"]}:{unit["text"]}'})
                    if unit["images"]:
                        quote_content.extend({"type": "image_url", "image_url": {"url": x}} for x in unit["images"])
                quote_content.append({"type": "text", "text": "\n</quote>"})
            chat_content: MultimodalContent = [{"type": "text", "text": content}]
            image_list = await asyncio.gather(*map(self._api.download_url, event.image_list))
            chat_content.extend({"type": "image_url", "image_url": {"url": x}} for x in image_list if x)
            session.current_input = [*quote_content, *chat_content]
            session.unit_prompts.append(f"Now:{self.today} {now.strftime("%I:%M %p")}")
            try:
                category = await self.router(session, event)
                if category_prompts := await self.activate_category(category, event):
                    session.unit_prompts.extend(category_prompts)
                session.activate()
                result = await self.execute_turn(session, event)
            except Exception as e:
                logger.exception(e)
                return
            finally:
                session.inactivate()
            session.last_active_time = timestamp
            session.over(chat_content, {"role": "assistant", "content": result}, timestamp)
            return result

    def update_usage(self, usage_counter: dict):
        if not usage_counter:
            return
        deep_add(self.usage_counter, usage_counter)
        usage = {k: v.get("total_tokens") for k, v in usage_counter.items()}
        usage_counter.clear()
        logger.info(f"[{self.name}][USAGE] {usage}")
        usage_file = self.usage_file
        usage_file.parent.mkdir(parents=True, exist_ok=True)
        with usage_file.open("w", encoding="utf-8") as f:
            json.dump(self.usage_counter, f, indent=4, ensure_ascii=False)

    async def chat(self, event: Event):
        session = self.current_session(event)
        result = await self.handle_chat(session, event)
        self.update_usage(session.usage_counter)
        return result


async def on_chat(agent: CloversAgent, event: Event):
    session = agent.current_session(event)
    session.api = agent.api("chat")
    session.payload = session.api.build_payload(session, agent.chat_prompt)
    session.payload["tools"] = [*agent.select_tools(ON_CHAT), agent.manifest[SKILL_MENU]]
    return ON_CHAT  # intro 不需要返回字符串，但是 on_chat 作为内置 intro 返回 ON_CHAT 比较方便


async def skill_menu(agent: CloversAgent, event: Event, category: str):
    session = agent.current_session(event)
    if prompts := await agent.activate_category(category, event):
        prompt = "\n".join(x for x in (prompts) if x)
    else:
        prompt = ""
    if "tools" not in session.payload:
        session.payload["tools"] = [agent.manifest[SKILL_MENU], *agent.select_tools(category)]
    else:
        used = {tool["function"]["name"] for tool in session.payload["tools"]}
        new_tools = agent.select_tools(category)
        session.payload["tools"].extend(x for x in new_tools if x["function"]["name"] not in used)
    return prompt or f"Skills for '{category}' have been loaded."


async def view_id_image(agent: CloversAgent, event: Event, image_id: int):
    session = agent.current_session(event)
    url = session.image_url(image_id)
    if not url:
        return f"Error: [image:{image_id}] is missing."
    session.current_input.append({"type": "image_url", "image_url": {"url": url}})
    return "OK"
