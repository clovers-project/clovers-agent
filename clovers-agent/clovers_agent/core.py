import time
import json
import asyncio
import httpx
from enum import IntEnum
from pathlib import Path
from datetime import datetime
from clovers.core import ModuleLoader
from clovers.logger import logger
from clovers_client import Event as BaseEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .api import OpenAIAPI
from .skill import SkillCore
from .session import Session
from .embedding import SentenceTransformer
from typing import Protocol
from .typing import SystemMessage, ToolMessage, Payload
from .typing.json_schema import BaseJSONSchemaType
from .typing.message import ContentSegment
from .config import Config


class Event(BaseEvent, Protocol):
    extra_context: list[str] = []


class CloversAgent(SkillCore, OpenAIAPI, ModuleLoader[SkillCore]):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, scheduler: AsyncIOScheduler, config: Config) -> None:
        OpenAIAPI.__init__(self, async_client, config.primary)
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
        # 模型设置
        self.scheduler = scheduler
        self.auxiliary = OpenAIAPI(async_client, config.auxiliary) if config.auxiliary is not None else self
        self.style_prompt = config.style_prompt
        self.base_prompt = config.base_prompt
        self.chat_prompt = config.chat_prompt
        self.call_prompt = config.call_prompt
        self.interim_prompt = config.interim_prompt
        self.memory_size = config.memory_size
        self.memory_timeout = config.memory_timeout
        self.topic_coldown = config.topic_coldown
        self.sentence_model = SentenceTransformer(config.sentence_model, cache_folder=config.sentence_model_cache)
        self.sessions: dict[str, Session] = {}
        # 注册技能
        self.skills = tuple()
        self._plugins = config.plugins
        self._plugin_dirs = config.plugin_dirs
        self._skill_dirs = config.skill_dirs
        self._category_schema: BaseJSONSchemaType = {"type": "string"}
        self.scheduler.add_job(self.session_clear, trigger="cron", hour="*/8", misfire_grace_time=120)

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
    async def _skill_menu(agent: "CloversAgent", event: Event, category: str):
        tip = f"已获取技能：{category}"
        agent.current_session(event).skill_menu = category
        hook = agent.category_hooks.get(category)
        if hook:
            info = coro if isinstance(coro := hook(agent, event), str) else await coro
            if info:
                tip += f"\n{info}"
        return tip

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
        logger.info(f"")
        self._category_schema["description"] = "\n".join(f"{category}: {desc}" for category, desc in self.categories.items())
        self._category_schema["enum"] = list(self.categories.keys())

    def skill_init(self):
        SkillCore.__init__(self)
        self.register(
            "skill_menu",
            "获取更多技能，如果assistant无法单独完成用户指令，则需要调用此方法获取更多技能。",
            {"category": self._category_schema},
        )(self._skill_menu)

        @self.register("system_test", "系统测试，包含一组系统测试任务。")
        async def _(agent: "CloversAgent", event: Event):
            print("测试任务开始执行")
            await asyncio.sleep(20)
            print("测试任务执行完毕")
            return ""

        self.load_from_list(self._plugins)
        self.load_from_dirs(self._plugin_dirs)
        self.sync_menu()

    def session_clear(self):
        timeout = time.time() - self.memory_timeout
        sessions_ids = tuple(s_id for s_id, s in self.sessions.items() if s.records[-1][2] < timeout)
        for sessions_id in sessions_ids:
            del self.sessions[sessions_id]
        logger.info(f"[{self.name}][SESSIONS_CLEAR]")

    @staticmethod
    def check(e: Event) -> bool:
        raise NotImplementedError

    async def summary_context(self, session: Session):
        aux = self.auxiliary
        content = "对历史全部对话进行详细总结，保留核心内容和结论，禁止输出除总结外的其他内容。"
        payload = aux.build_payload(context=(*session, {"role": "user", "content": content}))
        try:
            summary = (await aux.call_api(payload))["content"].strip()
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
            self.sessions[session_id] = Session(self.memory_size, self.sentence_model)
        return self.sessions[session_id]

    async def interim_reply(self, session: Session, text: str, images: list[str] | None = None):
        async with session.snap.lock:
            message = self.build_message(text, images)
            system_prompt = f"{self.style_prompt}\n{self.base_prompt}\n{self.chat_prompt}"
            payload = self.build_payload((*session.snap, message), system_prompt)
            reply = (await self.call_api(payload))["content"].strip()
            if session.lock.locked():
                interim_message = session.snap.records[-1][1]
                print(interim_message)
                print(session.interim_message)
                if interim_message["role"] == "system" and not interim_message["content"]:
                    interim_message["content"] = self.interim_prompt
                    session.interim_message = interim_message
                print(session.interim_message)
                session.snap.over(message, {"role": "assistant", "content": reply})
                return reply

    async def function_call(self, event: Event, call_infos: list[dict]):
        results: list[tuple[ToolMessage, str]] = []
        task_queue = []
        for call_info in call_infos:
            name = call_info["function"]["name"]
            try:
                func = self.invoker[name]
                kwargs = json.loads(call_info["function"]["arguments"])
                task_queue.append(func(call_info["id"], self, event, **kwargs))
            except KeyError:
                results.append(({"role": "tool", "tool_call_id": call_info["id"], "content": f'工具 "{name}" 不存在。'}, ""))
            except Exception as e:
                results.append(({"role": "tool", "tool_call_id": call_info["id"], "content": str(e)}, ""))
        results.extend(await asyncio.gather(*task_queue))
        return results

    class PromptIndex(IntEnum):
        STYLE = 0
        BASE = 1
        CHAT = 2
        HOOKS = 3
        EXTRA = 4

    async def call_unit(self, session: Session, event: Event, payload: Payload, extra_prompt: str = ""):
        hooks = [coro if isinstance(coro := hook(self, event), str) else await coro for hook in self.chat_hooks]
        hooks_prompt = "\n".join(prompt for prompt in hooks if prompt)
        prompts = [self.style_prompt, self.base_prompt, self.chat_prompt, hooks_prompt, extra_prompt]
        system_message: SystemMessage = {"role": "system", "content": "\n".join(prompt for prompt in prompts if prompt)}
        payload["messages"].insert(0, system_message)
        if len(self.invoker) > 1:
            payload["tools"] = self.intro_tools
        resp = await self.call_api(payload)
        # 退出条件：不需要额外技能
        if not (tool_calls := resp.get("tool_calls")):
            return resp["content"].strip()
        session.skill_menu = None

        intro_prompt = "\n".join(msg[0]["content"] for msg in await self.function_call(event, tool_calls) if msg[1])
        system_prompt = "\n".join(prompt for prompt in prompts if prompt)
        # 退出条件：不需要额外技能
        if session.skill_menu:
            toolkit = {"skill_menu"}
            prompts[self.PromptIndex.STYLE] = self.call_prompt
            prompts[self.PromptIndex.CHAT] = ""
            system_message["content"] = "\n".join(prompt for prompt in prompts if prompt)
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
                    if key:
                        toolkit.add(key)
                    payload["messages"].append(msg)
        else:
            api = self.auxiliary if session.unimportant else self
            system_message["content"] = system_prompt
            result = (await api.call_api(api.build_payload(payload["messages"])))["content"].strip()
        if not session.interim_message:
            return result
        session.interim_message["content"] = "[正在执行用户任务]"
        aux = self.auxiliary
        if result:
            result = f"[用户任务执行完毕]请以你的语气风格完整复述如下内容：\n\n{result}"
        else:
            result = "[用户任务执行完毕]请告知用户任务执行失败"
        payload = aux.build_payload((*session.snap, {"role": "user", "content": result}), system_prompt)
        return (await aux.call_api(payload))["content"].strip()

    async def chat(self, event: Event):
        timestamp = time.time()
        now = datetime.fromtimestamp(timestamp)
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
        image_list = event.image_list
        if image_list:
            image_list = await asyncio.gather(*(self.download_url(image) for image in event.image_list))
            image_list = [image for image in image_list if image]
        if session.lock.locked():
            if not session.silence:
                return
            if session.snap.lock.locked():
                return
            return await self.interim_reply(session, request, image_list)
        async with session.lock:
            session.memory_filter(timestamp - self.memory_timeout)
            session.silence_filter(timestamp - self.topic_coldown)
            session.silence.append((request, timestamp))
            message = list(x[0] for x in session.silence)
            message = "\n".join(message)
            session.sync_snap()
            content: list[ContentSegment] = []
            if (call := event.call("flat_context")) and (flat_context := await call):
                content.append({"type": "text", "text": "<引用上下文>"})
                for unit in flat_context:
                    if unit["text"]:
                        content.append({"type": "text", "text": f'{unit["nickname"]}:{unit["text"]}'})
                    if unit["images"]:
                        content.extend({"type": "image_url", "image_url": {"url": x}} for x in unit["images"])
                content.append({"type": "text", "text": "</引用上下文>"})
            content.append({"type": "text", "text": message})
            content.extend({"type": "image_url", "image_url": {"url": x}} for x in image_list)
            session.current_input = {"role": "user", "content": content}  # 注入输入（可修改）
            session.snap.over(session.current_input, {"role": "system", "content": ""})
            payload: Payload = {"model": self.model, "messages": [*session, session.current_input]}
            try:
                result = await self.call_unit(session, event, payload, f"今天的日期是:{now.strftime('%Y年%m月%d日')}")
            except Exception as e:
                logger.exception(e)
                return
            finally:
                session.snap.clear()
            if session.step(body) and (summary := await self.summary_context(session)):
                session.clear()
                session.silence.append((summary, timestamp))
            session.over(session.current_input, {"role": "assistant", "content": result}, timestamp)
            session.current_input = None
            return result
