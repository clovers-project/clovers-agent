import sys
import time
import json
import traceback
import contextlib
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from collections import deque
from clovers.core import ModuleLoader
from clovers.logger import logger
from clovers_client import Event as BaseEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .api import OpenAIAPI, HybridOpenAIAPI
from .skill import SkillCore, AgentCore, invoker_wrapper
from .session import Session, SessionComplete
from .utils import deep_add, save_json, decode_bytes
from .config import HybridOpenAIConfig, CONFIG, PROMPTS
from .constants import (
    SYSTEM_TAG,
    USER_TAG,
    ASSISTANT_TAG,
    ON_CHAT,
    ON_CHAT_DESC,
    SKILL_MENU,
    SKILL_MENU_DESC,
    BUILTIN_CATEGORY,
    GET_IMAGE_BY_ID_INFO,
    EXECUTE_SCRIPT,
    EXECUTE_SCRIPT_DESC,
    EXECUTE_SCRIPT_TIMEOUT,
    EXECUTE_SCRIPT_MAX_TAIL_LEN,
    READ_REFERENCE,
    READ_REFERENCE_DESC,
)
from typing import Protocol, TypedDict, override
from .typing import ToolMessage, ToolCallInfo, Payload
from .typing.message import MultimodalContent
from .typing.json_schema import BaseJSONSchemaType


class Event(BaseEvent, Protocol):
    extra_context: list[str] = []


class ToolCallError(TypedDict):
    name: str
    args: dict
    traceback: str


class CloversAgent(AgentCore, ModuleLoader[SkillCore]):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, scheduler: AsyncIOScheduler) -> None:
        AgentCore.__init__(self, CONFIG.skill_dirs)
        ModuleLoader.__init__(self, ["TOOLS"], SkillCore)
        self.name = name
        self.async_client = async_client
        # 核心
        self._api = self.creat_api(CONFIG.api)
        self._apis = {name: self.creat_api(api_config) for name, api_config in CONFIG.apis.items()}
        from .embedding import SentenceTransformer

        self.sentence_model = SentenceTransformer(CONFIG.sentence_model, cache_folder=CONFIG.sentence_model_cache)
        self.scheduler = scheduler
        # 文件
        path = Path(CONFIG.path)
        self.usage_dir = path / "usages"
        self.session_dir = path / "sessions"
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
        # 聊天配置
        self.call_depth = CONFIG.call_depth
        self.wait_cooldown = CONFIG.wait_cooldown
        self.chime_in_cooldown = CONFIG.chime_in_cooldown
        # DEBUG
        self.tool_failures: deque[ToolCallError] = deque(maxlen=5)
        self.extra = {}
        # 技能初始化
        self._category_schema: BaseJSONSchemaType = {"type": "string"}
        self.register(ON_CHAT, ON_CHAT_DESC)(on_chat)
        self.register(SKILL_MENU, SKILL_MENU_DESC, {"category": self._category_schema}, BUILTIN_CATEGORY)(skill_menu)
        self.invoker[EXECUTE_SCRIPT] = invoker_wrapper(EXECUTE_SCRIPT, execute_script)
        self.invoker[READ_REFERENCE] = invoker_wrapper(READ_REFERENCE, read_reference)
        self.category_decorator(GET_IMAGE_BY_ID_INFO, BUILTIN_CATEGORY)(view_id_image)
        self.load_from_list(CONFIG.plugins)
        self.load_from_dirs(CONFIG.plugin_dirs)
        self.reload_skills()
        self.sync_menu()
        self.scheduler.add_job(self.daily_tasks, trigger="cron", hour=2, misfire_grace_time=3600)

    def sync_menu(self):
        self._category_schema["description"] = "\n".join(f"{category}: {desc}" for category, desc in self.category_desc_map.items())
        self._category_schema["enum"] = list(self.category_desc_map.keys())

    @property
    def skill_dirs(self):
        return (_p for _s_dir in self._skill_dirs if (_dir := Path(_s_dir).expanduser()).exists() for _p in _dir.iterdir())

    def recorder_file(self, session_id: str):
        return self.session_dir / session_id / "recorder.json"

    @property
    def usage_file(self):
        return self.usage_dir / f"{self.today}.json"

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
        try:
            self.merge(tools)
        except Exception as e:
            logger.error(f'[{self.name}][LOAD TOOLS] "{package}" load failed: {e}')
            return
        logger.info(f'[{self.name}][LOAD TOOLS] "{package}" loaded')

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
        self.router_prompt = self.load_prompt(self.prompts_dir / "ROUTER.md", PROMPTS.router_prompt)
        self._style_prompt = self.load_prompt(self.prompts_dir / "STYLE.md", PROMPTS.style_prompt)
        self.base_prompt = self.load_prompt(self.prompts_dir / "BASE.md", PROMPTS.base_prompt)
        self._chat_prompt = self.load_prompt(self.prompts_dir / "CHAT.md", PROMPTS.chat_prompt)
        self._chime_in_prompt = self.load_prompt(self.prompts_dir / "CHIME_IN.md", PROMPTS.chime_in_prompt)
        self.wait_prompt = self.load_prompt(self.prompts_dir / "WAIT.md", PROMPTS.wait_prompt)
        self.summary_prompt = self.load_prompt(self.prompts_dir / "SUMMARY.md", PROMPTS.summary_prompt)

    @property
    def style_prompt(self) -> str:
        """Agent 人物设定核心提示

        需要角色连贯性的回复应含有此系统提示，并在此提示后接该调用系统提示的具体的要求。
        """
        return "\n".join(x for x in (self._style_prompt, self.base_prompt) if x)

    @property
    def chat_prompt(self) -> str:
        """Agent 预设的聊天场景。"""
        return "\n".join(x for x in (self._style_prompt, self.base_prompt, self._chat_prompt) if x)

    @property
    def chime_in_prompt(self) -> str:
        """Agent 主动回复要求"""
        return "\n".join(x for x in (self._style_prompt, self.base_prompt, self._chime_in_prompt) if x)

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

    @staticmethod
    def session_id(event: Event) -> str:
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, event: Event):
        session_id = self.session_id(event)
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id, self.sentence_model)
            record_path = self.recorder_file(session_id)
            if record_path.exists():
                try:
                    with record_path.open("r", encoding="utf-8") as f:
                        self.sessions[session_id].recorder.extend(json.load(f))
                except Exception as e:
                    logger.error(f"[{self.name}][RECORDER_LOAD] {e}")
        return self.sessions[session_id]

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

    async def invoke_tool(self, event: Event, call_info: ToolCallInfo) -> ToolMessage:
        name = call_info["function"]["name"]
        tc_id = call_info["id"]
        raw_args = call_info["function"]["arguments"]
        if name not in self.invoker:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}`: not found"}
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"{name}`: bad JSON args"}
        try:
            invoker = self.invoker[name](self, event, **args)
        except TypeError as e:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"{name}: invalid args: {e}"}
        try:
            return {"role": "tool", "tool_call_id": tc_id, "content": await invoker}
        except SessionComplete:
            raise
        except Exception as e:
            logger.exception(e)
            self.tool_failures.appendleft({"name": name, "args": args, "traceback": "".join(traceback.format_exception(e))})
            return {"role": "tool", "tool_call_id": tc_id, "content": f"{name} failed"}

    async def call_turn(self, api: OpenAIAPI, payload: Payload, usage_counter: dict, event: Event):
        for _ in range(self.call_depth):
            message = await api.call_api(payload, usage_counter)
            if not (tool_calls := message.get("tool_calls")):
                content = message["content"].strip()
                if not content:
                    continue
                return content
            messages = await asyncio.gather(*(self.invoke_tool(event, x) for x in tool_calls))
            payload["messages"].append(message)
            payload["messages"].extend(messages)
        raise RuntimeError(f"Maximum tool call chain length exceeded")

    async def router(self, text_input: str, event: Event, usage_counter: dict):
        if len(self.intro_tools) < 2:
            return await on_chat(self, event)
        api = self.api("router")
        payload = api.build_payload(({"role": "user", "content": text_input},), self.router_prompt)
        payload["tools"] = self.intro_tools
        message = await api.call_api(payload, usage_counter)
        if "tool_calls" not in message:
            raise ValueError(f"message must contain tool_calls, but got {message}")
        category = message["tool_calls"][0]["function"]["name"]
        logger.info(f"[{self.name}][ROUTER] {category}")
        await self.invoker[category](self, event)
        # 因为 router 的模型一般很轻量，这里禁止 intro_invoker 的参数防止造成不必要的 fallback
        return category

    async def chime_in(self, session: Session, content: str):
        api = self.api("chime_in")
        payload = api.build_payload(({"role": "user", "content": content},), self.chime_in_prompt)
        logger.info(f"[{self.name}][CHIME_IN]")
        resp = await api.call_api(payload, session.usage_counter)
        return resp["content"].strip()

    async def wait_reply(self, session: Session, content: str):
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
            body = f"@ME {at}{message}\n{"\n".join(event.extra_context)}"
        elif event.to_me:
            body = f"@ME {at}{message}"
        else:
            body = f"{at}{message}"
            session.silence_recorder.append((USER_TAG.format(event.nickname, body), timestamp))
            if event.at or (timestamp - session.last_active_time) < self.chime_in_cooldown:
                return
            async with session.execute_lock, session.wait_lock:
                content = "\n".join(x for x, _ in session.silence_recorder)
                try:
                    result = await self.chime_in(session, content)
                except Exception as e:
                    logger.exception(e)
                    return
                session.last_active_time = timestamp
                session.over(content, {"role": "assistant", "content": result}, timestamp)
                save_json(self.recorder_file(session.session_id), session.recorder)
                return result
        session.refresh(timestamp)
        session.silence_recorder.append((USER_TAG.format(event.nickname, body), timestamp))
        content = "\n".join(x for x, _ in session.silence_recorder)
        if session.execute_lock.locked():
            if session.wait_lock.locked():
                return
            silence_duration = timestamp - session.last_active_time
            if silence_duration < self.wait_cooldown:
                return
            async with session.wait_lock:
                try:
                    result = await self.wait_reply(session, content)
                except Exception as e:
                    logger.exception(e)
                    return
                session.last_active_time = timestamp
                return result
        async with session.execute_lock:
            if session.step(body, timestamp) and (summary := await self.summary_context(session)):
                session.clear()
                content = summary + "\n" + content
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
            text_input = [f"{a}\n{ASSISTANT_TAG.format(b)}" for a, b in session.router_recorder]
            text_input.append(content)
            try:
                intro = await self.router("\n".join(text_input), event, session.usage_counter)
                session.api
                session.payload
            except SessionComplete as e:
                return e.data
            except Exception as e:
                logger.warning(f"[{self.name}][ROUTER] {ON_CHAT} {e}")
                intro = await on_chat(self, event)
            try:
                if intro_prompts := await self.activate_category(intro, event):
                    session.unit_prompts.extend(intro_prompts)
            except SessionComplete as e:
                return e.data
            except Exception as e:
                logger.warning(f"[{self.name}][INTRO HOOK] {intro} {e}")
            session.activate()
            try:
                result = await self.call_turn(session.api, session.payload, session.usage_counter, event)
            except SessionComplete as e:
                return e.data
            except Exception as e:
                session.inactivate()
                file = f"payload_{id(session.payload)} {datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
                save_json(self.session_dir / session.session_id / file, session.payload)
                logger.exception(e)
                return
            session.inactivate()
            if result:
                session.last_active_time = timestamp
                session.over(chat_content, {"role": "assistant", "content": result}, timestamp)
                save_json(self.recorder_file(session.session_id), session.recorder)
            return result

    def update_usage(self, usage_counter: dict):
        if not usage_counter:
            return
        deep_add(self.usage_counter, usage_counter)
        usage = {k: v.get("total_tokens") for k, v in usage_counter.items()}
        usage_counter.clear()
        logger.info(f"[{self.name}][USAGE] {usage}")
        save_json(self.usage_file, self.usage_counter)

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
        prompt = "\n".join(x for x in prompts if x)
    else:
        prompt = ""
    if "tools" not in session.payload:
        session.payload["tools"] = [agent.manifest[SKILL_MENU], *agent.select_tools(category)]
    else:
        used = {tool["function"]["name"] for tool in session.payload["tools"]}
        new_tools = agent.select_tools(category)
        session.payload["tools"].extend(x for x in new_tools if x["function"]["name"] not in used)
    if scripts_map := agent.scripts_map.get(category):
        if not session.scripts_enum:
            session.payload["tools"].append(
                {
                    "type": "function",
                    "function": {
                        "name": EXECUTE_SCRIPT,
                        "description": EXECUTE_SCRIPT_DESC,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "interpreter": {"type": "string"},
                                "path": {"type": "string", "enum": session.scripts_enum},
                                "args": {"type": "array", "items": {"type": "string"}},
                                "timeout": {"type": "number"},
                            },
                            "required": ["path"],
                        },
                    },
                }
            )
        existing = set(session.scripts_enum)
        session.scripts_enum.extend(path for path in scripts_map if path not in existing)
    if references_map := agent.references_map.get(category):
        if not session.references_enum:
            session.payload["tools"].append(
                {
                    "type": "function",
                    "function": {
                        "name": READ_REFERENCE,
                        "description": READ_REFERENCE_DESC,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string", "enum": session.references_enum}},
                            "required": ["path"],
                        },
                    },
                }
            )
        existing = set(session.references_enum)
        session.references_enum.extend(path for path in references_map if path not in existing)
    return prompt or f"Skills for '{category}' have been loaded."


async def view_id_image(agent: CloversAgent, event: Event, image_id: int):
    session = agent.current_session(event)
    url = session.image_url(image_id)
    if not url:
        return f"Error: [image:{image_id}] is missing."
    session.current_input.append({"type": "image_url", "image_url": {"url": url}})
    return "OK"


async def cleanup_process(process: asyncio.subprocess.Process):
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            process.kill()
            await process.wait()


async def execute_script(
    agent: CloversAgent,
    event: Event,
    interpreter: str = "",
    path: str = "",
    args: list[str] | None = None,
    timeout: int = 120,
):
    if not path:
        return "Error: path is missing."
    if interpreter in ("python", "python3"):
        cmd = [sys.executable]
    elif not interpreter:
        cmd = []
    else:
        cmd = [interpreter]
    cmd.append(agent.scripts_map[path.split("/", 1)[0]][path])  # path 是映射路径，路径规则 "{category}/relative_path"
    if args:
        cmd.extend(args)
    try:
        process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=min(timeout, EXECUTE_SCRIPT_TIMEOUT))
    except asyncio.TimeoutError:
        asyncio.create_task(cleanup_process(process))
        return f"ERROR: Script execution timed out after {timeout} seconds."
    except FileNotFoundError:
        return f"ERROR: Interpreter '{interpreter}' not found."
    except PermissionError:
        return f"ERROR: Permission denied – cannot execute script."
    except Exception as e:
        return f"ERROR: Unexpected error – {e}"
    out_text = decode_bytes(stdout)
    MAX_TAIL_LEN = EXECUTE_SCRIPT_MAX_TAIL_LEN + 3
    if len(out_text) > MAX_TAIL_LEN:
        out_text = "..." + out_text[-EXECUTE_SCRIPT_MAX_TAIL_LEN:]
    err_text = decode_bytes(stderr)
    if len(err_text) > MAX_TAIL_LEN:
        err_text = "..." + err_text[-EXECUTE_SCRIPT_MAX_TAIL_LEN:]
    return f"RETURN CODE: {process.returncode}\nOUT: {out_text}\nERR: {err_text}"


async def read_reference(agent: CloversAgent, event: Event, path: str):
    file = Path(agent.references_map[path.split("/", 1)[0]][path])
    if not file.exists():
        return f"ERROR: File '{path}' does not exist."
    return decode_bytes(file.read_bytes())
