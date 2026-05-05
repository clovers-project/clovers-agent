if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

import httpx
import asyncio
from clovers import Plugin, Result
from clovers_client import Event as EventProtocol
from clovers_client.result import SegmentedResult, SegmentedMessage
from clovers_apscheduler import scheduler
from .core import Event, CloversAgent
from .config import Config

ASYNC_CLIENT = httpx.AsyncClient(timeout=300)
CONFIG = Config.sync_config(__package__)
AGENT = CloversAgent("CloversAgent", ASYNC_CLIENT, scheduler, CONFIG)

PLUGIN = Plugin[Event](priority=100)
PLUGIN.protocol = EventProtocol


PLUGIN.startup(AGENT.skill_init)
PLUGIN.shutdown(ASYNC_CLIENT.aclose)


@PLUGIN.handle(
    None,
    ["user_id", "group_id", "nickname", "to_me", "image_list", "at"],
    rule=AGENT.check,
    priority=2,
    block=False,
)
async def _(event: Event) -> SegmentedResult | None:
    result = await AGENT.chat(event)
    if result is None:
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
            for seg in result.split("\n"):
                seg = seg.strip()
                if not seg:
                    continue
                yield Result("text", seg)
                await asyncio.sleep(min(1 + 0.12 * len(seg), 8))
