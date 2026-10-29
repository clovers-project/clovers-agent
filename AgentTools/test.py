if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from clovers_agent import CloversAgent, Turn, Event, AgentTools
from clovers_agent.constants import INTRINSIC_CATEGORY

TOOLS = AgentTools()


@TOOLS.register(
    "test_daily_tasks",
    description="测试执行每天执行一次的定时任务",
    category=INTRINSIC_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event) -> str:
    session = agent.current_session(turn.session_id)
    session.last_active_time = 0
    await agent.daily_tasks()
    return "OK"


@TOOLS.register(
    "test_compact_context",
    description="测试执行压缩上下文",
    category=INTRINSIC_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event) -> str:
    await agent.compact_context(agent.current_session(turn.session_id), event)
    return "下一轮生效"


@TOOLS.register(
    "test_raise_exception",
    description="测试抛出异常",
    category=INTRINSIC_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event) -> str:
    return "helloworld" + 1  # type: ignore[operator]
