from clovers_agent import cast_invoker, CloversAgent, Turn, Event
from clovers_agent.constants import ON_CHAT


@cast_invoker
async def on_erotic(agent: CloversAgent, turn: Turn, event: Event, content: str):
    turn.create_payload(agent.api("on_erotic"), f"{agent.style_prompt}\n{content}")
    return ON_CHAT
