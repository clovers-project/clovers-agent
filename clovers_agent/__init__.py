import httpx
from datetime import datetime
from clovers import Plugin, Result
from .core import ToolManager, CloversAgent
from .typing import Event

__plugin__ = Plugin(build_result=lambda result: Result("text", result), priority=100)
__plugin__.set_protocol("properties", Event)

agent: CloversAgent


@__plugin__.startup
async def _():
    global agent
    agent = CloversAgent("CloversAgent", httpx.AsyncClient(timeout=300))


@__plugin__.handle(None, ["user_id", "group_id", "nickname", "image_list", "to_me"], priority=2, block=False)
async def _(event: Event):
    session = agent.current_session(event)
    now = datetime.now()
    if event.to_me:
        session.silence.append((f"{event.nickname}[{now.strftime("%I:%M %p")}]@me {event.message}", now.timestamp()))
    else:
        session.silence.append((f"{event.nickname}[{now.strftime("%I:%M %p")}]{event.message}", now.timestamp()))
        return
    if session.running:
        return
    session.running = True
    result = await agent.chat(event)
    session.running = False
    return result


__version__ = "0.1.0"
__all__ = ["ToolManager", "CloversAgent", "Event", "__plugin__"]
