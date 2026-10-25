import httpx
import asyncio
from clovers import Plugin
from clovers.logger import logger
from clovers_client import Event as EventProtocol
from clovers_apscheduler import scheduler
from .agent import Event, CloversAgent
from .config import CHECK

from typing import override

KEY_SENDING_LOCK = "sending_lock"


class ChatAgent(CloversAgent):
    @override
    def send(self, result: str, event: Event):
        session = self.current_session(self.session_id(event))
        if KEY_SENDING_LOCK not in session.extra:
            lock = session.extra[KEY_SENDING_LOCK] = asyncio.Lock()
        else:
            lock = session.extra[KEY_SENDING_LOCK]
        asyncio.create_task(chat_send(lock, result, event))


async def chat_send(lock: asyncio.Lock, result: str, event: Event):
    lines = tuple(x for line in result.split("\n") if (x := line.strip()))
    async with lock:
        if len(lines) > 4:
            await event.send("text", result)
            await asyncio.sleep(min(0.12 * len(result), 8))
            return
        for text in lines:
            await event.send("text", text)
            await asyncio.sleep(min(0.12 * len(text), 8))


AGENT = ChatAgent("CLOVERS_AGENT", httpx.AsyncClient(timeout=300), scheduler)
PLUGIN = Plugin[Event](priority=100)
PLUGIN.protocol = EventProtocol

PLUGIN.startup(AGENT.startup)
PLUGIN.shutdown(AGENT.async_client.aclose)


def check(e: Event) -> bool: ...


# clovers 设置
if CHECK.whitelist:
    whitelist = set(CHECK.whitelist)
    logger.info("[%s] 检查规则设置为白名单模式: %s", AGENT.name, whitelist)
    check = lambda e: e.group_id is not None and e.group_id in whitelist
elif CHECK.blacklist:
    blacklist = set(CHECK.blacklist)
    logger.info("[%s] 检查规则设置为黑名单模式: %s", AGENT.name, blacklist)
    check = lambda e: e.group_id is not None and e.group_id not in blacklist
elif CHECK.console_mode:
    logger.info("[%s] 启动控制台模式", AGENT.name)
    check = lambda e: True
else:
    logger.info("[%s] 已关闭", AGENT.name)
    check = lambda e: False

PLUGIN.handle(None, ["user_id", "group_id", "nickname", "to_me", "image_list", "at"], rule=check, priority=2, block=False)(AGENT.chat)
