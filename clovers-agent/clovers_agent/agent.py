import sys
import traceback
import contextlib
import time
import json
import asyncio
import httpx
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from collections import deque
from clovers.core import ModuleLoader
from clovers.logger import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .api import OpenAIAPI, HybridOpenAIAPI
from .core import invoker_wrapper, as_invoker, AgentTools, AgentSkills, Event
from .session import iter_recoder, Session, Turn, SessionComplete
from .utils import deep_add, read_file, load_json, save_json, decode_bytes
from .config import HybridOpenAIConfig, CONFIG, PROMPTS
from .constants import (
    USER_TAG,
    ON_CHAT,
    ON_CHAT_DESC,
    SKILL_MENU,
    SKILL_MENU_DESC,
    BUILTIN_CATEGORY,
    GET_IMAGE_BY_ID_INFO,
    EXECUTE_SCRIPT,
    EXECUTE_SCRIPT_DEFAULT_TIMEOUT,
    EXECUTE_SCRIPT_MAX_TIMEOUT,
    EXECUTE_SCRIPT_DESC,
    EXECUTE_SCRIPT_MAX_TAIL_LEN,
    READ_REFERENCE,
    READ_REFERENCE_DESC,
)

from typing import override, TypedDict
from .typing import ToolMessage, ToolCallInfo
from .typing.json_schema import BaseJSONSchemaType


class ToolCallError(TypedDict):
    name: str
    args: dict
    traceback: list[str]


class CloversAgent(AgentSkills, ModuleLoader[AgentTools]):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, scheduler: AsyncIOScheduler) -> None:
        AgentSkills.__init__(self, CONFIG.skill_dirs)
        ModuleLoader.__init__(self, ["TOOLS"], AgentTools)
        self.name = name
        self.async_client = async_client
        # 核心
        self._api = self.create_api(CONFIG.api)
        self._apis = {name: self.create_api(api_config) for name, api_config in CONFIG.apis.items()}
        self.scheduler = scheduler
        # 文件
        path = Path(CONFIG.path)
        self.usage_dir = path / "usages"
        self.session_dir = path / "sessions"
        self.prompts_dir = path / "prompts"
        # 状态
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.sessions: dict[str, Session] = {}
        self.usage_counter = load_json(self.usage_file) or {}
        self.tool_failures: deque[ToolCallError] = deque(maxlen=5)
        self.extra = {}
        # 聊天配置
        self.call_depth = CONFIG.call_depth
        self.wait_cooldown = CONFIG.wait_cooldown
        self.chime_in_cooldown = CONFIG.chime_in_cooldown
        # 技能初始化
        self._category_schema: BaseJSONSchemaType = {"type": "string"}
        self.register(ON_CHAT, ON_CHAT_DESC)(on_chat)
        self.register(SKILL_MENU, SKILL_MENU_DESC, {"category": self._category_schema}, BUILTIN_CATEGORY)(skill_menu)
        self.invoker[EXECUTE_SCRIPT] = invoker_wrapper(EXECUTE_SCRIPT, execute_script)
        self.invoker[READ_REFERENCE] = invoker_wrapper(READ_REFERENCE, read_reference)
        self.category_decorator(GET_IMAGE_BY_ID_INFO, BUILTIN_CATEGORY)(view_id_image)
        self.scheduler.add_job(self.daily_tasks, trigger="cron", hour=2, misfire_grace_time=3600)

    def reload_prompts(self):
        # Prompts
        self.base_prompt = read_file(self.prompts_dir / "BASE.md", PROMPTS.base_prompt)
        style_prompt = read_file(self.prompts_dir / "STYLE.md", PROMPTS.style_prompt)
        self.style_prompt = "\n".join(x for x in (style_prompt, self.base_prompt) if x)
        chat_prompt = read_file(self.prompts_dir / "CHAT.md", PROMPTS.chat_prompt)
        self.chat_prompt = "\n".join(x for x in (style_prompt, self.base_prompt, chat_prompt) if x)
        self.chime_in_prompt = read_file(self.prompts_dir / "CHIME_IN.md", PROMPTS.chime_in_prompt)
        self.router_prompt = read_file(self.prompts_dir / "ROUTER.md", PROMPTS.router_prompt)
        self.wait_prompt = read_file(self.prompts_dir / "WAIT.md", PROMPTS.wait_prompt)
        self.summary_prompt = read_file(self.prompts_dir / "SUMMARY.md", PROMPTS.summary_prompt)

    def startup(self):
        from .embedding import SentenceTransformer

        self.sentence_model = SentenceTransformer(CONFIG.sentence_model, cache_folder=CONFIG.sentence_model_cache)
        self.reload_prompts()
        self.load_from_list(CONFIG.plugins)
        self.load_from_dirs(CONFIG.plugin_dirs)
        self.reload_skills()
        self.sync_menu()
        self.validate_manifest()

    def sync_menu(self):
        self._category_schema["description"] = "\n".join(f"{category}: {desc}" for category, desc in self.category_desc_map.items())
        self._category_schema["enum"] = list(self.category_desc_map.keys())

    def recorder_file(self, session_id: str):
        return self.session_dir / session_id / "recorder.jsonl"

    @property
    def usage_file(self):
        return self.usage_dir / f"{self.today}.json"

    def create_api(self, config: HybridOpenAIConfig):
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

    def daily_tasks(self):
        timestamp = time.time()
        sessions_ids = tuple(s_id for s_id, s in self.sessions.items() if s.last_active_time < timestamp - s.memory_timeout)
        for sessions_id in sessions_ids:
            del self.sessions[sessions_id]
        logger.info(f"[{self.name}][SESSIONS CLEAR]")
        usage = {k: v.get("total_tokens") for k, v in self.usage_counter.items()}
        logger.info(f"[{self.name}][USAGE TODAY] {usage}")
        self.today = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        self.usage_counter.clear()

    async def summary_context(self, session: Session):
        s = session.create_turn()
        s.create_payload(self.api("summary"), self.base_prompt)
        s.payload["messages"].extend(session.messages)
        s.unit_prompts.append(self.summary_prompt)
        s.activate()
        try:
            resp = await s.call_api()
            summary = resp["content"].strip()
            logger.info(f"[{self.name}][SUMMARY]")
            logger.debug(summary)
            return summary
        except Exception as e:
            logger.warning(f"[{self.name}][SUMMARY] {e}")
            return

    @staticmethod
    def session_id(event: Event) -> str:
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = Session(session_id, self.recorder_file(session_id), self.sentence_model)
            self.sessions[session_id] = session
        return session

    async def activate_category(self, name: str, session: Turn, event: Event) -> list[str] | None:
        hooks = self.category_hooks.get(name)
        if not hooks:
            return
        return [coro if isinstance((coro := hook(self, session, event)), str) else await coro for hook in hooks]

    async def invoke_tool(self, call_info: ToolCallInfo, session: Turn, event: Event) -> ToolMessage:
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
            invoker = self.invoker[name](self, session, event, **args)
        except TypeError as e:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"{name}: invalid args: {e}"}
        try:
            return {"role": "tool", "tool_call_id": tc_id, "content": await invoker}
        except SessionComplete:
            raise
        except Exception as e:
            logger.exception(e)
            if self.tool_failures is not None:
                self.tool_failures.append({"name": name, "args": args, "traceback": traceback.format_exception(e)})
            return {"role": "tool", "tool_call_id": tc_id, "content": f"{name} failed"}

    async def call_turn(self, session: Turn, event: Event):
        try:
            session.activate()
            for _ in range(self.call_depth):
                message = await session.call_api()
                session.payload["messages"].append(message)
                if content := message["content"].strip():
                    yield content
                if not (tool_calls := message.get("tool_calls")):
                    session.inactivate()
                    return
                messages = await asyncio.gather(*(self.invoke_tool(x, session, event) for x in tool_calls))
                session.payload["messages"].extend(messages)
            raise TimeoutError(f"Maximum tool call chain length exceeded")
        except Exception as e:
            logger.exception(e)
            file = f"payload_{session.session_id} {datetime.now().strftime('%Y%m%d-%H%M%S')} {id(session.payload)}.json"
            save_json(self.session_dir / session.session_id / file, session.payload)
            session.inactivate()

    async def router(self, content: str, session: Session, event: Event):
        if len(self.intro_manifest) < 2:
            return await on_chat(self, session, event)
        s = session.create_turn()
        s.create_payload(self.api("router"), self.base_prompt)
        s.unit_prompts.append(self.router_prompt)
        s.payload["messages"].extend(iter_recoder(session.router_recorder))
        s.input_text(content)
        s.payload["tools"] = list(self.intro_manifest.values())
        s.activate()
        message = await s.call_api()
        if "tool_calls" not in message:
            raise ValueError(f"message must contain tool_calls, but got {message}")
        intro = message["tool_calls"][0]["function"]["name"]
        # 因为 router 的模型一般很轻量，这里禁止 intro_invoker 的参数防止造成不必要的 fallback
        return await self.intro_invoker[intro](self, session, event) or intro

    async def handle_chime_in(self, session: Session, content: str):
        s = session.create_turn()
        s.create_payload(self.api("chime_in"), self.chat_prompt)
        s.payload["messages"].extend(session.messages)
        s.unit_prompts.append(self.chime_in_prompt)
        s.input_text(content)
        s.activate()
        try:
            message = await s.call_api()
            content = message["content"].strip()
            logger.info(f"[{self.name}][CHIME IN]")
            return content
        except Exception as e:
            logger.warning(f"[{self.name}][CHIME IN] {e}")

    async def handle_wait_replay(self, session: Session, content: str):
        if session.cursor is None:
            return
        s = session.create_turn()
        s.create_payload(session.api)
        keys = set(session.payload.keys())
        keys.remove("messages")
        for k, v in session.payload.items():
            if k == "messages":
                continue
            s.payload[k] = deepcopy(v)
        if session.cursor > 0:
            s.payload["messages"].extend(session.payload["messages"][: session.cursor - 1])
        ask = session.executing_ask
        if isinstance(ask, list):
            ask = "".join(x["text"] for x in ask if x["type"] == "text")
        if session.executing_reply:
            s.payload["messages"].append({"role": "user", "content": ask})
            s.payload["messages"].append({"role": "assistant", "content": session.executing_reply})
        else:
            s.input_text(ask)
        s.input_text(content)
        s.unit_prompts.append(self.wait_prompt)
        try:
            message = await s.call_api()
            content = message["content"].strip()
            logger.info(f"[{self.name}][WAIT REPLY]")
            return content
        except Exception as e:
            logger.warning(f"[{self.name}][WAIT REPLY] {e}")

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
            if session.execute_lock.locked() or session.wait_lock.locked():
                return
            content = "\n".join(x for x, _ in session.silence_recorder)
            session.executing_ask = content
            session.executing_reply = None
            async with session.execute_lock, session.wait_lock:
                content = "\n".join(x for x, _ in session.silence_recorder)
                if reply := await self.handle_chime_in(session, content):
                    self.send(reply, event)
                    session.executing_reply = reply
                    session.record_turn(timestamp)
                return
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
                if reply := await self.handle_wait_replay(session, content):
                    self.send(reply, event)
                return
        session.executing_ask = content
        session.executing_reply = None
        async with session.execute_lock:
            if session.step(body, timestamp) and (summary := await self.summary_context(session)):
                session.clear()
                session.input_text(summary)
            if (call := event.call("flat_context")) and (flat_context := await call):
                session.input_text("<quote>\n")
                for unit in flat_context:
                    if unit["text"]:
                        session.input_text(f'{unit["nickname"]}:{unit["text"]}')
                    if unit["images"]:
                        session.input_image(unit["images"])
                session.input_text("\n</quote>")
            session.input_text(content)
            image_list = await asyncio.gather(*map(self._api.download_url, event.image_list))
            session.input_image(x for x in image_list if x)
            try:
                intro = await self.router(content, session, event)
                session.api
                session.payload
                logger.info(f"[{self.name}][ROUTER] {intro}")
            except Exception as e:
                intro = await on_chat(self, session, event)
                logger.warning(f"[{self.name}][ROUTER] {intro} {e}")
            session.extend_tools(self.select_tools(intro))
            try:
                if intro_prompts := await self.activate_category(intro, session, event):
                    session.unit_prompts.extend(intro_prompts)
            except Exception as e:
                logger.warning(f"[{self.name}][INTRO HOOK] {intro} {e}")
            session.unit_prompts.append(f"Now:{self.today} {now.strftime("%I:%M %p")}")
            session.executing_ask = session.current_input.copy()
            async for reply in self.call_turn(session, event):
                session.executing_reply = reply
                self.send(reply, event)
            session.record_turn(timestamp)

    async def handle_wakeup(self, session: Turn, event: Event, m_idx: int, r_idx: int):
        current_session = self.current_session(session.session_id)
        recorder = current_session.recorder
        current_session.executing_ask = session.current_input.copy()
        current_session.executing_reply = None
        async with current_session.execute_lock, current_session.wait_lock:
            if len(recorder) <= r_idx:
                session.payload["messages"].extend(iter_recoder(recorder))
            else:
                session.payload["messages"][m_idx] = recorder[r_idx][0].copy()
                session.payload["messages"].extend(iter_recoder(recorder[r_idx + 1 :]))
            async for reply in self.call_turn(session, event):
                current_session.executing_reply = reply
                self.send(reply, event)
            current_session.record_turn(time.time())

    def update_usage(self, usage_counter: dict):
        if not usage_counter:
            return
        deep_add(self.usage_counter, usage_counter)
        usage = {k: v.get("total_tokens") for k, v in usage_counter.items()}
        usage_counter.clear()
        logger.info(f"[{self.name}][USAGE] {usage}")
        save_json(self.usage_file, self.usage_counter)

    def send(self, text: str, event: Event):
        asyncio.create_task(event.send("text", text))

    async def chat(self, event: Event):
        session = self.current_session(self.session_id(event))
        try:
            await self.handle_chat(session, event)
        except SessionComplete as e:
            if reply := e.data:
                self.send(reply, event)
        self.update_usage(session.usage_counter)

    async def wakeup(self, session: Turn, event: Event, m_idx: int, ridx: int):
        try:
            await self.handle_wakeup(session, event, m_idx, ridx)
        except SessionComplete as e:
            if reply := e.data:
                self.send(reply, event)
        self.update_usage(session.usage_counter)


@as_invoker
async def on_chat(agent: CloversAgent, session: Turn, event: Event):
    session.create_payload(agent.api(ON_CHAT), agent.chat_prompt)
    session.payload["tools"] = [agent.manifest[SKILL_MENU]]
    return ON_CHAT


@as_invoker
async def skill_menu(agent: CloversAgent, session: Turn, event: Event, category: str):
    if prompts := await agent.activate_category(category, session, event):
        prompt = "\n".join(x for x in prompts if x)
    else:
        prompt = ""
    payload_tools = session.extend_tools((agent.manifest[SKILL_MENU], *agent.select_tools(category)))
    if scripts_map := agent.scripts_map.get(category):
        if not session.scripts_enum:
            payload_tools.append(
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
            payload_tools.append(
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


@as_invoker
async def view_id_image(agent: CloversAgent, session: Turn, event: Event, image_id: int):
    url = agent.current_session(session.session_id).image_url(image_id)
    if not url:
        return f"Error: [image:{image_id}] is missing."
    session.current_input.append({"type": "image_url", "image_url": {"url": url}})
    return "OK"


async def cleanup_process(process: asyncio.subprocess.Process):
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=5)


async def decode_stream(reader: asyncio.StreamReader, tail_length: int):
    maxlen = tail_length + 2
    buffer = deque[str](maxlen=maxlen)
    async for line in reader:
        buffer.extend(decode_bytes(line.rstrip().rsplit(b"\r", 1)[-1]))
        buffer.append("\n")
    if len(buffer) == maxlen:
        buffer.popleft()
        buffer.popleft()
        buffer.popleft()
        buffer.extendleft("...")
    return "".join(buffer)


@as_invoker
async def execute_script(
    agent: CloversAgent,
    session: Turn,
    event: Event,
    interpreter: str = "",
    path: str = "",
    args: list[str] | None = None,
    timeout: int = EXECUTE_SCRIPT_DEFAULT_TIMEOUT,
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
    except FileNotFoundError:
        return f"ERROR: Interpreter '{interpreter}' not found."
    except PermissionError:
        return f"ERROR: Permission denied – cannot execute script."
    except Exception as e:
        return f"ERROR: Unexpected error – {e}"
    try:
        tasks = (
            decode_stream(process.stdout, EXECUTE_SCRIPT_MAX_TAIL_LEN),  # type: ignore
            decode_stream(process.stderr, EXECUTE_SCRIPT_MAX_TAIL_LEN),  # type: ignore
            process.wait(),
        )
        if not isinstance(timeout, int):
            timeout = EXECUTE_SCRIPT_DEFAULT_TIMEOUT
        else:
            timeout = min(timeout, EXECUTE_SCRIPT_MAX_TIMEOUT)
        out_text, err_text, returncode = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        return f"RETURN CODE: {returncode}\nOUT: {out_text}\nERR: {err_text}"
    except asyncio.TimeoutError:
        await cleanup_process(process)
        return f"ERROR: Script execution timed out after {timeout} seconds."
    except Exception as e:
        await cleanup_process(process)
        return f"ERROR: Unexpected error – {e}"


@as_invoker
async def read_reference(agent: CloversAgent, session: Turn, event: Event, path: str):
    file = Path(agent.references_map[path.split("/", 1)[0]][path])
    if not file.exists():
        return f"ERROR: File '{path}' does not exist."
    return decode_bytes(file.read_bytes())
