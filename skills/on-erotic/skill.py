from clovers_agent import CloversAgent, Event


async def on_erotic(agent: CloversAgent, event: Event, content: str):
    print("on_erotic 触发")
    print("接收skill.md文档：", content[:100])
    return "禁止涩涩"
