import sys
import traceback
import contextlib
import time
import json
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from collections import deque
from clovers.core import ModuleLoader
from clovers.logger import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .api import OpenAIAPI, HybridOpenAIAPI
from .core import cast_invoker, AgentTools, AgentSkills, Event
from .session import iter_recoder, Turn, Session, SessionComplete, WakeupContext
from .utils import deep_add, read_file, load_json, save_json, decode_bytes
from .config import HybridOpenAIConfig, CONFIG, PROMPTS
from .constants import (
    USER_TAG,
    ASSISTANT_TAG,
    ON_CHAT,
    ON_CHAT_DESC,
    SKILL_MENU,
    SKILL_MENU_DESC,
    BUILTIN_CATEGORY,
    INTRINSIC_CATEGORY,
    READ_IMAGE_BY_ID_INFO,
    INJECT_PROMPT_INFO,
    EXECUTE_SCRIPT,
    EXECUTE_SCRIPT_DEFAULT_TIMEOUT,
    EXECUTE_SCRIPT_MAX_TIMEOUT,
    EXECUTE_SCRIPT_MAX_TAIL_LEN,
    READ_REFERENCE,
    WAKEUP_TIP,
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
        self.max_call_depth = CONFIG.max_call_depth
        self.chime_in_cooldown = CONFIG.chime_in_cooldown
        # 技能初始化
        self._category_schema: BaseJSONSchemaType = {"type": "string"}
        self.register(ON_CHAT, ON_CHAT_DESC)(on_chat)
        self.register(SKILL_MENU, SKILL_MENU_DESC, {"category": self._category_schema}, INTRINSIC_CATEGORY)(skill_menu)
        self.invoker[EXECUTE_SCRIPT] = execute_script
        self.invoker[READ_REFERENCE] = read_reference
        self.category_decorator(READ_IMAGE_BY_ID_INFO, BUILTIN_CATEGORY)(read_image_by_id)
        self.category_decorator(INJECT_PROMPT_INFO, BUILTIN_CATEGORY)(inject_prompt)
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
        self.aside_prompt = read_file(self.prompts_dir / "ASIDE.md", PROMPTS.aside_prompt)
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
        self._category_schema["enum"] = list(self.category_desc_map)

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
            logger.error('[%s][LOAD TOOLS] "%s" load failed: %s', self.name, package, e)
            return
        logger.info('[%s][LOAD TOOLS] "%s" loaded', self.name, package)

    def daily_tasks(self):
        timestamp = time.time()
        expired_session_ids = tuple(s_id for s_id, s in self.sessions.items() if s.last_active_time < timestamp - s.memory_timeout)
        for session_id in expired_session_ids:
            self.sessions[session_id].cleanup()
            del self.sessions[session_id]
        logger.info("[%s][SESSIONS CLEAR]", self.name)
        usage = {k: v.get("total_tokens") for k, v in self.usage_counter.items()}
        logger.info("[%s][USAGE TODAY] %s", self.name, usage)
        self.today = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        self.usage_counter.clear()

    async def summarize_context(self, session: Session):
        turn = session.create_turn()
        turn.create_payload(self.api("summary"), self.base_prompt)
        turn.payload["messages"].extend(session.messages)
        turn.unit_prompts.append(self.summary_prompt)
        turn.activate()
        try:
            message = await turn.call_api()
            summary = (message["content"] or "").strip()
            logger.info("[%s][SUMMARY]", self.name)
            logger.debug(summary)
            return summary
        except Exception as e:
            logger.warning("[%s][SUMMARY] failed: %s", self.name, e)
            return

    @staticmethod
    def session_id(event: Event) -> str:
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = Session(session_id, self.session_dir / session_id, self.sentence_model)
            self.sessions[session_id] = session
        return session

    async def activate_category(self, name: str, turn: Turn, event: Event) -> list[str] | None:
        hooks = self.category_hooks.get(name)
        if not hooks:
            return
        return await asyncio.gather(*(hook(self, turn, event) for hook in hooks))

    async def invoke_tool(self, call_info: ToolCallInfo, turn: Turn, event: Event) -> ToolMessage:
        name = call_info["function"]["name"]
        tc_id = call_info["id"]
        raw_args = call_info["function"]["arguments"]
        logger.info("[%s][CALL][%s]", self.name, name)
        logger.debug("[%s][CALL][%s][IN] %s", self.name, name, raw_args)
        if name not in self.invoker:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}`: not found"}
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}`: bad JSON args: {e}"}
        try:
            invoker = self.invoker[name](self, turn, event, **args)
        except TypeError as e:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}`: invalid args: {e}"}
        try:
            content = await invoker
            logger.debug("[%s][CALL][%s][OUT] %s", self.name, name, content)
        except SessionComplete:
            raise
        except Exception as e:
            logger.exception("[%s][CALL][%s][%s] failed", self.name, name, tc_id)
            self.tool_failures.append({"name": name, "args": args, "traceback": traceback.format_exception(e)})
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}` failed"}
        return {"role": "tool", "tool_call_id": tc_id, "content": content}

    async def iter_replies(self, turn: Turn, event: Event):
        try:
            turn.activate()
            for _ in range(self.max_call_depth):
                message = await turn.call_api()
                turn.payload["messages"].append(message)
                content = (message["content"] or "").strip()
                if tool_calls := message.get("tool_calls"):
                    messages = await asyncio.gather(*(self.invoke_tool(x, turn, event) for x in tool_calls))
                    turn.payload["messages"].extend(messages)
                    yield content
                    continue
                else:
                    turn.deactivate()
                    yield content
                    return
            raise TimeoutError(f"Maximum tool call chain length exceeded")
        except Exception as e:
            logger.exception("[%s][CALL TURN] session %s failed", self.name, turn.session_id)
            file = f"payload_{turn.session_id} {datetime.now().strftime('%Y%m%d-%H%M%S')} {id(turn.payload)}.json"
            save_json(self.session_dir / turn.session_id / file, turn.payload)
            turn.deactivate()

    async def resolve_turn(self, turn: Turn, event: Event):
        last_reply = None
        async for reply in self.iter_replies(turn, event):
            last_reply = reply
        return last_reply

    async def route(self, content: str, session: Session, event: Event):
        if len(self.intro_manifest) < 2:
            return await on_chat(self, session, event)
        turn = session.create_turn()
        turn.create_payload(self.api("router"), self.base_prompt)
        turn.unit_prompts.append(self.router_prompt)
        turn.payload["messages"].extend(iter_recoder(session.router_recorder))
        turn.input_text(content)
        turn.payload["tools"] = list(self.intro_manifest.values())
        turn.activate()
        message = await turn.call_api()
        if "tool_calls" not in message:
            raise ValueError(f"message must contain tool_calls, but got {message}")
        intro = message["tool_calls"][0]["function"]["name"]
        # 因为 router 的模型一般很轻量，这里禁止 intro_invoker 的参数防止造成不必要的 fallback
        return await self.intro_invoker[intro](self, session, event) or intro

    async def call_chime_in(self, session: Session, content: str):
        turn = session.create_turn()
        turn.create_payload(self.api("chime_in"), self.chat_prompt)
        turn.payload["messages"].extend(session.messages)
        turn.unit_prompts.append(self.chime_in_prompt)
        turn.input_text(content)
        turn.activate()
        try:
            message = await turn.call_api()
            reply = (message["content"] or "").strip()
            logger.info("[%s][CHIME IN]", self.name)
            return reply
        except Exception as e:
            logger.warning("[%s][CHIME IN] session %s failed: %s: %s", self.name, turn.session_id, type(e).__name__, e)

    async def call_aside(self, session: Session, event: Event, content: str, current_output: str):
        if session.cursor is None:
            return
        turn = session.create_turn()
        turn.create_payload(session.api)
        if session.cursor > 0:
            turn.payload["messages"].extend(session.payload["messages"][: session.cursor - 1])
        turn.payload["tools"] = [INJECT_PROMPT_INFO]
        turn.input_text(content)
        turn.unit_prompts.append(self.aside_prompt.format(current_output))
        turn.activate()
        try:
            message = await turn.call_api()
            if tool_calls := message.get("tool_calls"):
                await self.invoke_tool(tool_calls[0], turn, event)
                reply = None
            else:
                reply = (message["content"] or "").strip()
            logger.info("[%s][ASIDE]", self.name)
            return reply
        except Exception as e:
            logger.warning("[%s][ASIDE] session %s failed: %s: %s", self.name, turn.session_id, type(e).__name__, e)

    def update_usage(self, usage_counter: dict):
        if not usage_counter:
            return
        deep_add(self.usage_counter, usage_counter)
        usage = {k: v.get("total_tokens") for k, v in usage_counter.items()}
        usage_counter.clear()
        logger.info("[%s][USAGE] %s", self.name, usage)
        save_json(self.usage_file, self.usage_counter)

    def send(self, text: str, event: Event):
        asyncio.create_task(event.send("text", text))

    async def execute_session(self, session: Session, turn: Turn, event: Event, timestamp: float):
        current_input = turn.current_input.copy()
        last_reply = None
        session.pending_prompt = None
        session.last_active_time = timestamp
        async for reply in self.iter_replies(turn, event):
            last_reply = reply
            if session.pending_prompt:
                async with session.aside_lock:
                    pending_prompt = session.pending_prompt
                    session.pending_prompt = None
                    aside_reply = await self.call_aside(session, event, pending_prompt, last_reply)
                    if aside_reply:
                        self.send(aside_reply, event)
                        session.record_history(ASSISTANT_TAG.format(aside_reply), timestamp)
                    elif prompt := session.pending_prompt:
                        turn.payload["messages"].append({"role": "user", "content": prompt})
                session.pending_prompt = None
            else:
                if reply:
                    self.send(reply, event)
        if reply:
            current_input[-1] = {"type": "text", "text": session.turn_input}
            session.record_turn(current_input, reply)
        return last_reply

    def submit_turn_for_wakeup(self, turn: Turn, event: Event):
        session = self.current_session(turn.session_id)
        if session.wakeup_context is None:
            if session.cursor is None:
                raise RuntimeError("Cannot create WakeupContext when the session is inactive")
            turn = turn.create_turn()
            turn.usage_counter = {}
            turn.api = session.api
            turn.payload = session.payload
            wakeup_context = session.wakeup_context = WakeupContext(turn)
            asyncio.create_task(self.run_wakeup(session, event, session.last_active_time))
        else:
            wakeup_context = session.wakeup_context
        turn.usage_counter = wakeup_context.turn.usage_counter
        wakeup_context.tasks.append(asyncio.create_task(self.resolve_turn(turn, event)))

    async def run_wakeup(self, session: Session, event: Event, action_time: float):
        if not (wakeup_context := session.wakeup_context):
            raise RuntimeError("Cannot run wakeup without a WakeupContext")
        turn = wakeup_context.turn
        tasks = wakeup_context.tasks
        seen = set()
        while True:
            # 合并多个调用链的唤醒任务，触发一次醒
            # 当 session.idle_event 重新 clear 时 session 仍保有 wakeup_context
            await session.idle_event.wait()
            unseen_tasks = [t for t in tasks if t not in seen]
            seen.update(unseen_tasks)
            if not unseen_tasks:
                break
            await asyncio.gather(*unseen_tasks)
        session.wakeup_context = None
        turn.unit_prompts.append(WAKEUP_TIP)
        turn.unit_prompts.extend("\n---\n".join(task.result() or "None" for task in tasks))
        try:
            async with session.execute_lock:
                recorder = session.recorder
                turn.payload["messages"].extend(iter_recoder(reversed(tuple(x for x in reversed(recorder) if x[2] > action_time))))
                turn.input_text(session.turn_input)
                await self.execute_session(session, turn, event, time.time())
        except SessionComplete as e:
            if reply := e.data:
                self.send(reply, event)
        self.update_usage(session.usage_counter)

    async def handle_input(self, session: Session, event: Event):
        timestamp = time.time()
        now = datetime.fromtimestamp(timestamp)
        if "nicknames" not in session.extra:
            session.extra["nicknames"] = {}
        nicknames = session.extra["nicknames"]
        nicknames[event.user_id] = event.nickname
        mention_prefix = "".join(f"@{name} " for user_id in event.at if (name := nicknames.get(user_id))) if event.at else ""
        if "extra_context" in event.properties:
            message = f"@ME {mention_prefix}{event.message}\n{"\n".join(event.extra_context)}"
        elif event.to_me:
            message = f"@ME {mention_prefix}{event.message}"
        else:
            session.record_history(USER_TAG.format(event.nickname, f"{mention_prefix}{event.message}"), timestamp)
            if event.at or (timestamp - session.last_active_time) < self.chime_in_cooldown:
                return
            if session.execute_lock.locked() or session.aside_lock.locked():
                return
            async with session.execute_lock, session.aside_lock:
                turn_input = session.turn_input
                if chime_in_reply := await self.call_chime_in(session, turn_input):
                    self.send(chime_in_reply, event)
                    session.last_active_time = timestamp
                    session.record_turn(turn_input, chime_in_reply)
                return
        session.record_history(USER_TAG.format(event.nickname, message), timestamp)
        if session.execute_lock.locked():
            if session.aside_lock.locked() or session.idle_event.is_set():
                return
            session.pending_prompt = session.turn_input
            return
        async with session.execute_lock:
            if session.should_summarize(message, timestamp) and (summary := await self.summarize_context(session)):
                session.compact_history(summary)
            turn_input = session.turn_input
            if (call := event.call("flat_context")) and (flat_context := await call):
                session.input_text("<quote>\n")
                for unit in flat_context:
                    if unit["text"]:
                        session.input_text(f'{unit["nickname"]}:{unit["text"]}')
                    if unit["images"]:
                        session.input_image(unit["images"])
                session.input_text("\n</quote>")
            images = await asyncio.gather(*(map(self._api.download_url, event.image_list)))
            session.input_image(image for image in images if image)
            session.input_text(turn_input)
            try:
                intro = await self.route(turn_input, session, event)
                session.api
                session.payload
                logger.info("[%s][ROUTER] %s", self.name, intro)
            except Exception as e:
                intro = await on_chat(self, session, event)
                logger.warning("[%s][ROUTER] session %s failed: %s: %s", self.name, session.session_id, type(e).__name__, e)
            session.extend_tools(self.select_tools(INTRINSIC_CATEGORY))
            session.extend_tools(self.select_tools(intro))
            try:
                if intro_prompts := await self.activate_category(intro, session, event):
                    session.unit_prompts.extend(intro_prompts)
            except Exception as e:
                logger.warning("[%s][INTRO HOOK][%s] session %s failed: %s: %s", self.name, intro, session.session_id, type(e).__name__, e)
            session.unit_prompts.append(f"Now:{self.today} {now.strftime("%I:%M %p")}")
            await self.execute_session(session, session, event, timestamp)

    async def on_input(self, event: Event):
        session = self.current_session(self.session_id(event))
        try:
            await self.handle_input(session, event)
        except SessionComplete as e:
            if reply := e.data:
                self.send(reply, event)
        self.update_usage(session.usage_counter)


async def on_chat(agent: CloversAgent, turn: Turn, event: Event):
    turn.create_payload(agent.api(ON_CHAT), agent.chat_prompt)
    return ON_CHAT


@cast_invoker
async def skill_menu(agent: CloversAgent, turn: Turn, event: Event, category: str):
    if prompts := await agent.activate_category(category, turn, event):
        prompt = "\n".join(x for x in prompts if x)
    else:
        prompt = ""
    turn.extend_tools(agent.select_tools(category))
    if scripts_map := agent.scripts_map.get(category):
        turn.extend_scripts(scripts_map.keys())
    if references_map := agent.references_map.get(category):
        turn.extend_references(references_map.keys())
    return prompt or f"Skills for '{category}' have been loaded."


@cast_invoker
async def read_image_by_id(agent: CloversAgent, turn: Turn, event: Event, image_id: int):
    url = agent.current_session(turn.session_id).image_url(image_id)
    if not url:
        return f"Error: [image:{image_id}] is missing."
    turn.current_input.append({"type": "image_url", "image_url": {"url": url}})
    return "OK"


@cast_invoker
async def inject_prompt(agent: CloversAgent, turn: Turn, event: Event, prompt: str):
    agent.current_session(turn.session_id).pending_prompt = prompt
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


@cast_invoker
async def execute_script(
    agent: CloversAgent,
    turn: Turn,
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
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        tasks = (
            decode_stream(process.stdout, EXECUTE_SCRIPT_MAX_TAIL_LEN),
            decode_stream(process.stderr, EXECUTE_SCRIPT_MAX_TAIL_LEN),
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


@cast_invoker
async def read_reference(agent: CloversAgent, turn: Turn, event: Event, path: str):
    file = Path(agent.references_map[path.split("/", 1)[0]][path])
    if not file.exists():
        return f"ERROR: File '{path}' does not exist."
    return decode_bytes(file.read_bytes())
