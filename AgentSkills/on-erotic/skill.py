from clovers_agent import as_invoker, CloversAgent, Turn, Event
from clovers_agent.constants import ON_CHAT


@as_invoker
async def on_erotic(agent: CloversAgent, session: Turn, event: Event, content: str):
    session.create_payload(agent.api("erotic"), f"{agent.style_prompt}\n{content}")
    return ON_CHAT
