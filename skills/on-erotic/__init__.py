from clovers_agent import Event
from clovers_agent.core import CloversAgent


async def on_erotic(agent: CloversAgent, event: Event, content: str):
    print("涩涩")
    return content
