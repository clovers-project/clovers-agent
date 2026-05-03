from clovers_agent import CloversAgent, Event


async def on_erotic(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    session.unimportant()
    session.chat_prompt = "\n".join(x for x in (agent.style_prompt, agent.base_prompt, content, session.unit_prompt) if x)
    return "OK"
