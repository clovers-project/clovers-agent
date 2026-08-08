from clovers_agent import CloversAgent, Event
from clovers_agent.constants import SKILL_MENU


async def on_skill(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    session.api = agent.api("skill")
    session.payload = session.api.build_payload(session, f"{content}\n{agent.base_prompt}")
    session.payload["tools"] = [agent.manifest[SKILL_MENU]]
    return ""
