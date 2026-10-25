from clovers_agent import cast_invoker, CloversAgent, Turn, Event
from clovers_agent.constants import ON_CHAT


@cast_invoker
async def on_erotic(agent: CloversAgent, session: Turn, event: Event, content: str):
    session.create_payload(agent.api("on_erotic"), agent.style_prompt)
    session.unit_prompts.append(content)
    return ON_CHAT
