import httpx
import asyncio
from clovers import Plugin
from clovers.logger import logger
from clovers_client import Event as EventProtocol
from clovers_apscheduler import scheduler
from .agent import Event, CloversAgent
from .session import TurnSession
from .config import CHECK

from typing import override


class ChatAI(CloversAgent):
    def __init__(self) -> None:
        super().__init__("ChatAI", httpx.AsyncClient(timeout=300), scheduler)

    async def send(self, result: str, event: Event):
        lines = tuple(x for line in result.split("\n") if (x := line.strip()))
        if len(lines) > 4:
            await event.send("text", result)
            return
        session = self.current_session(self.session_id(event))
        if "sending_lock" not in session.extra:
            session.extra["sending_lock"] = asyncio.Lock()
        sending_lock = session.extra["sending_lock"]
        async with sending_lock:
            for text in lines:
                await event.send("text", text)
                await asyncio.sleep(min(0.12 * len(text), 8))

    @override
    async def chat(self, event: Event):
        result = await super().chat(event)
        if result:
            await self.send(result, event)

    @override
    async def wakeup(self, session: TurnSession, event: Event):
        result = await super().wakeup(session, event)
        if result:
            await self.send(result, event)


AGENT = ChatAI()
PLUGIN = Plugin[Event](priority=100)
PLUGIN.protocol = EventProtocol

PLUGIN.startup(AGENT.startup)
PLUGIN.shutdown(AGENT.async_client.aclose)


def check(e: Event) -> bool: ...


# clovers 设置
if CHECK.whitelist:
    whitelist = set(CHECK.whitelist)
    logger.info(f"[{AGENT.name}] 检查规则设置为白名单模式：{whitelist}")
    check = lambda e: e.group_id is not None and e.group_id in whitelist
elif CHECK.blacklist:
    blacklist = set(CHECK.blacklist)
    logger.info(f"[{AGENT.name}] 检查规则设置为黑名单模式：{blacklist}")
    check = lambda e: e.group_id is not None and e.group_id not in blacklist
elif CHECK.console_mode:
    logger.info(f"[{AGENT.name}] 启动控制台模式")
    check = lambda e: True
else:
    logger.info(f"[{AGENT.name}] 已关闭")
    check = lambda e: False

PLUGIN.handle(None, ["user_id", "group_id", "nickname", "to_me", "image_list", "at"], rule=check, priority=2, block=False)(AGENT.chat)
