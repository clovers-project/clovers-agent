from clovers_agent import CloversAgent, Event
from clovers_agent.constants import SKILL_MENU


async def on_skill(agent: CloversAgent, event: Event, content: str, category: str):
    session = agent.current_session(event)
    session.api = agent.api("skill")
    session.payload = session.api.build_payload(session, f"{content}\n{agent.base_prompt}")
    session.payload["tools"] = [agent.manifest[SKILL_MENU]]
    skill_prompt = (await agent.invoker[SKILL_MENU]("", agent, event, category))["content"]
    if skill_prompt:
        session.system_message["content"] += f"\n{skill_prompt}"
    return ""
