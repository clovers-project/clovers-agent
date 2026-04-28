if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

import httpx
from clovers import Plugin, Result
from clovers_client import Event as EventProtocol
from .core import Event, CloversAgent
from .config import Config

ASYNC_CLIENT = httpx.AsyncClient(timeout=300)
CONFIG = Config.sync_config(__package__)
AGENT = CloversAgent("CloversAgent", ASYNC_CLIENT, CONFIG)

PLUGIN = Plugin[Event](priority=100)
PLUGIN.protocol = EventProtocol


PLUGIN.startup(AGENT.init)
PLUGIN.shutdown(ASYNC_CLIENT.aclose)


@PLUGIN.handle(
    None,
    ["user_id", "group_id", "nickname", "to_me", "image_list", "at"],
    rule=AGENT.check,
    priority=2,
    block=False,
)
async def _(event: Event):
    result = await AGENT.chat(event)
    return Result("text", result) if result else None
