import json
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from clovers.core import ModuleLoader
from clovers.logger import logger
from clovers_client import Event as BaseEvent
from .api import OpenAIAPI
from .skill import SkillCore
from .session import Session
from .embedding import SentenceTransformer
from typing import Protocol
from .typing import UserMessage, AssistantMessage, SystemMessage, ToolMessage, Payload
from .typing.json_schema import BaseJSONSchemaType
from .typing.message import ContentSegment
from .config import Config


class Event(BaseEvent, Protocol):
    extra_context: list[str] = []


class CloversAgent(SkillCore, OpenAIAPI, ModuleLoader[SkillCore]):
    """OpenAI API"""

    def __init__(self, name: str, async_client: httpx.AsyncClient, config: Config) -> None:
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
        self.auxiliary = OpenAIAPI(async_client, config.auxiliary) if config.auxiliary is not None else self
        self.style_prompt = config.style_prompt
        self.base_prompt = config.system_prompt
        self.chat_prompt = config.chat_prompt
        self.call_prompt = config.call_prompt
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
        self.load_from_list(self._plugins)
        self.load_from_dirs(self._plugin_dirs)
        self.sync_menu()

    @staticmethod
    def check(e: Event) -> bool:
        raise NotImplementedError

    async def summary_context(self, session: Session):
        payload = self.build_payload(context=session)
        payload["messages"].append({"role": "user", "content": "对以上对话进行总结，保留核心内容和结论，禁止输出除总结外的其他内容。"})
        summary = (await self.call_api(payload))["content"].strip()
        logger.info(f"[{self.name}][SUMMARY]")
        logger.debug(summary)
        return summary

    @staticmethod
    async def default_func(tool_call_id: str, content: str) -> tuple[ToolMessage, str]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}, ""

    async def function_call(self, event: Event, call_infos: list[dict]) -> list[tuple[ToolMessage, str]]:
        task_queue = []
        for call_info in call_infos:
            name = call_info["function"]["name"]
            try:
                func = self.invoker[name]
                kwargs = json.loads(call_info["function"]["arguments"])
                task_queue.append(func(call_info["id"], self, event, **kwargs))
            except KeyError:
                task_queue.append(self.default_func(call_info["id"], f'工具 "{name}" 不存在。'))
            except Exception as e:
                task_queue.append(self.default_func(call_info["id"], str(e)))
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
        hooks = [coro if isinstance(coro := hook(self, event), str) else await coro for hook in self.chat_hooks]
        hooks_prompt = "\n".join(prompt for prompt in hooks if prompt)
        system_prompt = f"{self.style_prompt}\n{self.base_prompt}\n{self.chat_prompt}\n{hooks_prompt}\n{extra_prompt}"
        system_message: SystemMessage = {"role": "system", "content": system_prompt}
        payload["messages"].insert(0, system_message)
        if self.categories:
            payload["tools"] = self.intro_tools
        resp = await self.call_api(payload)
        # 退出条件：不需要额外技能
        if not (tool_calls := resp.get("tool_calls")):
            return resp["content"].strip()
        session.skill_menu = None
        intro_prompt = "".join(msg[0]["content"] for msg in await self.function_call(event, tool_calls) if msg[1])
        # 退出条件：不需要额外技能
        if session.skill_menu:
            toolkit = {"skill_menu"}
            system_message["content"] = f"{self.call_prompt}\n{hooks_prompt}\n{extra_prompt}\n{intro_prompt}"
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
            async with session.snap.lock:
                if result:
                    if session.records:
                        return result
                    else:
                        result = f"{result}\n\n请以你的语气风格完整复述上述内容。"
                else:
                    result = "请告知用户任务执行失败。"
                context = [system_message, *session.snap, {"role": "system", "content": result}]
        else:
            context = payload["messages"]
        system_message["content"] = f"{self.style_prompt}\n{hooks_prompt}\n{intro_prompt}"
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
        image_list = await asyncio.gather(*(self.download_url(image) for image in event.image_list))
        image_list = [image for image in image_list if image]
        if session.lock.locked():
            return await self.aux_reply(session, {"role": "user", "content": self.auxiliary.build_content(request, image_list)})
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
                session.snap.over(
                    {"role": "user", "content": message},
                    {"role": "system", "content": "任务正在执行，请稍等。"},
                )
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
            pure_content.extend({"type": "image_url", "image_url": {"url": x}} for x in image_list)
            content.extend(pure_content)
            session.current_input = {"role": "user", "content": content}  # 注入输入（可修改）
            payload: Payload = {"model": self.model, "messages": [*session, session.current_input]}
            try:
                resp = await self.call_unit(session, event, payload, f"今天的日期是:{now.strftime('%Y年%m月%d日')}")
            except Exception as e:
                logger.exception(e)
                return
            finally:
                session.snap.clear()
            session.over({"role": "user", "content": pure_content}, {"role": "assistant", "content": resp}, timestamp)
            session.current_input = None
            return resp

    def _load(self, package: str):
        tools = super()._load(package)
        if tools is None:
            return
        conflict = self.merge(tools)
        if conflict:
            logger.error(f'[{self.name}] "{package}" conflict with {conflict}')
            return
        logger.info(f'[{self.name}][TOOLS] "{package}" loaded')
