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
    LOAD_TOOLS,
    LOAD_TOOLS_DESC,
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

    def __init__(self, name: str, async_client: httpx.AsyncClient, scheduler: AsyncIOScheduler) -> None:
        """初始化 Agent。

        注册内置工具，初始化 API、会话和用量统计，并安排每日维护任务。

        Args:
            name (str): Agent 名称。
            async_client (httpx.AsyncClient): httpx 异步 HTTP 客户端。
            scheduler (AsyncIOScheduler): 异步任务调度器。
        """
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
        self.register(LOAD_TOOLS, LOAD_TOOLS_DESC, {"category": self._category_schema}, INTRINSIC_CATEGORY)(load_tools)
        self.tool_invoker[EXECUTE_SCRIPT] = execute_script
        self.tool_invoker[READ_REFERENCE] = read_reference
        self.category_decorator(READ_IMAGE_BY_ID_INFO, BUILTIN_CATEGORY)(read_image_by_id)
        self.category_decorator(INJECT_PROMPT_INFO, BUILTIN_CATEGORY)(inject_prompt)
        self.scheduler.add_job(self.daily_tasks, trigger="cron", hour=2, misfire_grace_time=3600)

    def reload_prompts(self):
        """重新加载提示词"""
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
        """执行 Agent 的启动初始化。"""

        from .embedding import SentenceTransformer

        self.sentence_model = SentenceTransformer(CONFIG.sentence_model, cache_folder=CONFIG.sentence_model_cache)
        self.reload_prompts()
        self.load_from_list(CONFIG.plugins)
        self.load_from_dirs(CONFIG.plugin_dirs)
        self.reload_skills()
        self.sync_menu()
        self.validate_manifest()

    def sync_menu(self):
        """同步工具分类菜单。

        根据已注册的工具分类更新分类参数的 JSON Schema，
        包括分类描述和可选值列表。
        """
        self._category_schema["description"] = "\n".join(f"{category}: {desc}" for category, desc in self.category_descriptions.items())
        self._category_schema["enum"] = list(self.category_descriptions)

    @property
    def usage_file(self):
        """获取当前日期对应的用量统计文件路径。

        Returns:
            Path: 当前日期的用量统计文件路径。
        """
        return self.usage_dir / f"{self.today}.json"

    def create_api(self, config: HybridOpenAIConfig):
        """根据配置创建 API 客户端。

        Args:
            config (HybridOpenAIConfig): API 配置。

        Returns:
            HybridOpenAIAPI | OpenAIAPI: 创建的 API 客户端。
        """
        if config.vision:
            return HybridOpenAIAPI(self.async_client, config)
        else:
            return OpenAIAPI(self.async_client, config)

    def api(self, key: str):
        """获取指定名称的 API 客户端。

        如果指定名称未配置，则返回默认 API 客户端。

        Args:
            key (str): API 配置名称。

        Returns:
            API 客户端: 对应的 API 客户端，或默认客户端。
        """
        return self._apis.get(key, self._api)

    def daily_tasks(self):
        """执行每日维护任务。

        清理超过记忆超时时间的会话，记录当日 Token 用量，
        更新当前日期，并重置用量统计计数器。
        """
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
        """根据会话消息生成上下文摘要。

        使用总结专用的 API 配置和提示词生成摘要，并记录执行日志。

        Args:
            session (Session): 待总结的会话。

        Returns:
            str | None: 生成的摘要文本；生成失败时返回 None。
        """
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
        """根据事件生成会话标识。

        群聊事件使用群组 ID，私聊事件使用用户 ID 构建标识。

        Args:
            event (Event): 当前事件。

        Returns:
            str: 会话标识。
        """
        return event.group_id or f"private-{event.user_id}"

    def current_session(self, session_id: str):
        """获取指定会话，不存在时创建新会话。

        新会话使用对应的会话目录和句向量模型进行初始化。

        Args:
            session_id (str): 会话标识。

        Returns:
            Session: 已存在或新创建的会话。
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = self.sessions[session_id] = Session(session_id, self.session_dir / session_id, self.sentence_model)
        return session

    async def activate_intro(self, intro: str, turn: Turn, event: Event):
        """执行指定初始工具钩子。

        并发执行该分类下注册的所有钩子，并收集执行结果。

        Args:
            intro (str): 初始工具名称。
            session (Turn): 当前轮次。
            event (Event): 当前事件。
        """
        hooks = self.intro_hooks.get(intro)
        if not hooks:
            return
        intro_prompts = await asyncio.gather(*(hook(self, turn, event) for hook in hooks))
        turn.unit_prompts.extend(x for x in intro_prompts if x)

    async def activate_category(self, category: str, turn: Turn, event: Event) -> list[str] | None:
        """执行指定工具分类的激活钩子。

        并发执行该分类下注册的所有钩子，并收集执行结果。

        Args:
            category (str): 工具分类名称。
            session (Turn): 当前轮次。
            event (Event): 当前事件。

        Returns:
            list[str] | None: 钩子返回的结果列表；没有注册钩子时返回 None。
        """
        hooks = self.category_hooks.get(category)
        if not hooks:
            return
        return await asyncio.gather(*(hook(self, turn, event) for hook in hooks))

    async def invoke_tool(self, call_info: ToolCallInfo, turn: Turn, event: Event) -> ToolMessage:
        """执行模型发起的工具调用。

        解析工具参数并调用对应的处理函数，将执行结果转换为工具消息。
        参数无效或工具执行失败时返回错误消息，并记录执行异常。

        Args:
            call_info (ToolCallInfo): 模型返回的工具调用信息。
            session (Turn): 当前轮次。
            event (Event): 当前事件。

        Returns:
            ToolMessage: 工具执行结果
        """
        name = call_info["function"]["name"]
        tc_id = call_info["id"]
        raw_args = call_info["function"]["arguments"]
        logger.info("[%s][CALL][%s]", self.name, name)
        logger.debug("[%s][CALL][%s][IN] %s", self.name, name, raw_args)
        if name not in self.tool_invoker:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}`: not found"}
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}`: bad JSON args: {e}"}
        try:
            invoker = self.tool_invoker[name](self, turn, event, **args)
        except TypeError as e:
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}`: invalid args: {e}"}
        try:
            content = await invoker
            logger.debug("[%s][CALL][%s][OUT] %s", self.name, name, content)
        except Exception as e:
            logger.exception("[%s][CALL][%s][%s] failed", self.name, name, tc_id)
            self.tool_failures.append({"name": name, "args": args, "traceback": traceback.format_exception(e)})
            return {"role": "tool", "tool_call_id": tc_id, "content": f"`{name}` failed"}
        return {"role": "tool", "tool_call_id": tc_id, "content": content}

    async def iter_replies(self, turn: Turn, event: Event):
        """迭代执行模型回复和工具调用链。

        循环请求模型并执行返回的工具调用，逐步产出模型的文本回复。
        模型返回最终回复时结束迭代。调用链异常时记录日志、
        保存请求载荷以供排查，并停用当前轮次。

        Args:
            turn (Turn): 当前调用链执行轮次。
            event (Event): 当前触发事件。

        Yields:
            str: 模型返回的文本内容。
        """
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
        """执行一个轮调用链并获取最后一条回复。

        Args:
            turn (Turn): 当前执行轮次。
            event (Event): 当前触发事件。

        Returns:
            str | None: 最后一条回复；没有产出回复时返回 None。
        """
        last_reply = None
        async for reply in self.iter_replies(turn, event):
            last_reply = reply
        return last_reply

    async def route(self, content: str, session: Session, event: Event):
        """根据用户输入选择对应的 Intro 工具。

        Args:
            content (str): 待检测的输入内容。
            session (Session): 当前会话。
            event (Event): 当前事件。

        Returns:
            str: Intro 名
        """
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
        """执行主动插话回复。

        使用插话专用的 API 配置和提示词，根据当前会话历史及输入内容生成回复。

        Args:
            session (Session): 当前会话。
            content (str): 用于生成插话回复的输入内容。

        Returns:
            str | None: 生成的回复文本；执行失败时返回 None。
        """
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
        """处理主任务执行期间的旁路交互。

        基于当前轮次的执行上下文创建独立轮次，并使用旁路提示词处理输入。
        模型请求注入提示词时执行对应工具，否则返回模型的文本回复。
        会话没有正在进行的执行任务时不执行处理。

        Args:
            session (Session): 当前会话。
            event (Event): 当前事件。
            content (str): 旁路交互的输入内容。
            current_output (str): 主任务当前已产生的回复。

        Returns:
            str | None: 模型的文本回复；没有文本回复或未执行处理时返回 None。
        """
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
        """合并并持久化 Token 用量统计。

        将传入的用量计数器累加到 Agent 的总计数器中，
        记录本次用量，并将更新后的统计数据写入文件。
        处理完成后清空传入的计数器。

        Args:
            usage_counter (dict): 待合并的用量统计。
        """
        if not usage_counter:
            return
        deep_add(self.usage_counter, usage_counter)
        usage = {k: v.get("total_tokens") for k, v in usage_counter.items()}
        usage_counter.clear()
        logger.info("[%s][USAGE] %s", self.name, usage)
        save_json(self.usage_file, self.usage_counter)

    def send(self, text: str, event: Event):
        """异步发送文本消息。

        创建异步任务，通过当前事件发送指定文本。

        Args:
            text (str): 待发送的文本。
            event (Event): 当前事件。
        """
        asyncio.create_task(event.send("text", text))

    async def execute_session(self, session: Session, turn: Turn, event: Event, timestamp: float):
        """会话执行出口

        迭代模型回复并发送给用户，处理执行期间收到的旁路输入，
        记录旁路回复，并在轮次结束后保存对话历史。

        Args:
            session (Session): 当前会话。
            turn (Turn): 待执行的轮次。
            event (Event): 当前事件。
            timestamp (float): 本轮输入对应的时间戳。

        Returns:
            str | None: 本轮最后一条回复；没有回复时返回 None。
        """
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
        """提交一轮唤醒调用链。

        Args:
            turn (Turn): 待提交的调用链执行轮次。
            event (Event): 用于执行调用链和唤醒的事件。
        """
        session = self.current_session(turn.session_id)
        if session.wakeup_context is None:
            print("创建唤醒调用链上下文")
            if session.cursor is None:
                raise RuntimeError("Cannot create WakeupContext when the session is inactive")
            wakeup_turn = turn.create_turn()
            wakeup_turn.usage_counter = {}
            wakeup_turn.api = session.api
            wakeup_turn.payload = session.payload
            wakeup_context = session.wakeup_context = WakeupContext(wakeup_turn)
            asyncio.create_task(self.run_wakeup(session, event, session.last_active_time))
        else:
            print("唤醒调用链已存在")
            wakeup_context = session.wakeup_context
        turn.usage_counter = wakeup_context.turn.usage_counter
        wakeup_context.tasks.append(asyncio.create_task(self.resolve_turn(turn, event)))

    async def run_wakeup(self, session: Session, event: Event, action_time: float):
        """执行唤醒

        Args:
            session (Session): 注册唤醒的会话。
            event (Event): 用于执行唤醒的事件。
            action_time (float): 会话注册唤醒时的活动时间戳。
        """
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
        """处理会话中的输入事件。

        Args:
            session (Session): 当前会话。
            event (Event): 当前事件。
        """
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
                await self.activate_intro(intro, session, event)
            except Exception as e:
                logger.warning("[%s][INTRO HOOK][%s] session %s failed: %s: %s", self.name, intro, session.session_id, type(e).__name__, e)
            session.unit_prompts.append(f"Now:{self.today} {now.strftime("%I:%M %p")}")
            await self.execute_session(session, session, event, timestamp)

    async def on_input(self, event: Event):
        """处理外部输入事件的入口。

        Args:
            event (Event): 当前事件。
        """
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
async def load_tools(agent: CloversAgent, turn: Turn, event: Event, category: str):
    prompts = []
    tools = agent.select_tools(category)
    if tools:
        turn.extend_tools(tools)
        prompts.append(f"Tools injected and now available: {", ".join(x["function"]["name"] for x in tools)}")
    if tips := await agent.activate_category(category, turn, event):
        prompts.extend(tips)
    if scripts_map := agent.scripts_map.get(category):
        turn.extend_scripts(scripts_map.keys())
    if references_map := agent.references_map.get(category):
        turn.extend_references(references_map.keys())
    return "\n".join(prompts)


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
        if not isinstance(timeout, float):
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
