from clovers_agent import CloversAgent, Event
from clovers_agent.constants import ON_CHAT


async def on_erotic(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    unit_prompt = "\n".join(x for x in (session.unit_prompts) if x)
    prompts = [agent.style_prompt, agent.base_prompt, content, unit_prompt]
    if chat_prompts := await agent.activate_category(ON_CHAT, event):
        prompts.append("\n".join(x for x in (chat_prompts) if x))
    session.payload = agent.api.build_payload(session.unimportant_context, "\n".join(x for x in prompts if x))
    session.payload["tools"] = agent.select_tools(ON_CHAT).copy()
    session.unimportant = True
    return ""
