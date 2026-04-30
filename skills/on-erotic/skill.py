from clovers_agent import CloversAgent, Event


async def on_erotic(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    session.unimportant = True
    return content
