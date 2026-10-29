if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from clovers_agent import CloversAgent, Turn, Event, AgentTools
from clovers_agent.constants import INTRINSIC_CATEGORY

TOOLS = AgentTools()


@TOOLS.register(
    "daily_tasks",
    description="测试执行每天执行一次的定时任务",
    category=INTRINSIC_CATEGORY,
)
async def daily_tasks(agent: CloversAgent, turn: Turn, event: Event) -> str:
    session = agent.current_session(turn.session_id)
    session.last_active_time = 0
    await agent.daily_tasks()
    return "OK"
