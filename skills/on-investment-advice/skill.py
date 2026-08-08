from clovers_agent import CloversAgent, Event


async def on_investment_advice(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    session.api = agent.api("investment_advice")
    session.payload = session.api.build_payload(session, f"{content}\n{agent.base_prompt}")
    session.payload["tools"] = agent.select_tools("network").copy()
    return ""


async def screen_stocks_entry(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    for _ in range(agent.call_depth):
        message = await session.api.call_api(session.payload, session.usage_counter)
