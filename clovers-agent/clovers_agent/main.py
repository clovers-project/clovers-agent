import httpx
import asyncio
from clovers import Plugin, Result
from clovers.logger import logger
from clovers_client import Event as EventProtocol
from clovers_client.result import SegmentedResult, SegmentedMessage
from clovers_apscheduler import scheduler
from .core import Event, CloversAgent
from .config import CHECK

AGENT = CloversAgent("CloversAgent", httpx.AsyncClient(timeout=300), scheduler)
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


@PLUGIN.handle(None, ["user_id", "group_id", "nickname", "to_me", "image_list", "at"], rule=check, priority=2, block=False)
async def _(event: Event) -> SegmentedResult | None:
    result = await AGENT.chat(event)
    if not result:
        return None
    session = AGENT.current_session(event)
    if "sending_lock" not in session.extra:
        session.extra["sending_lock"] = asyncio.Lock()
    sending_lock = session.extra["sending_lock"]
    return Result("segmented", format_message(result, sending_lock))


async def format_message(result: str, lock: asyncio.Lock) -> SegmentedMessage:
    async with lock:
        lines = [x for line in result.split("\n") if (x := line.strip())]
        if len(lines) > 4:
            yield Result("text", result)
        else:
            for seg in lines:
                yield Result("text", seg)
                await asyncio.sleep(min(0.12 * len(seg), 8))
