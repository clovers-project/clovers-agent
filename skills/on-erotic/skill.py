from clovers_agent import CloversAgent, Event
from clovers_agent.constants import ON_CHAT


async def on_erotic(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    session.api = agent.api("erotic")
    if category_prompts := await agent.activate_category(ON_CHAT, event):
        session.unit_prompts.extend(category_prompts)
    system_prompt = f"{agent.style_prompt}\n{content}"
    session.payload = session.api.build_payload(session.unimportant_context, system_prompt)
    session.payload["tools"] = agent.select_tools(ON_CHAT).copy()
    session.unimportant = True
    return ""
