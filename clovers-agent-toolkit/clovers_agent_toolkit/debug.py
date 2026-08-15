import sys
import importlib
from clovers_agent import CloversAgent, Event, SkillCore
from clovers_agent.core import ON_CHAT, SKILL_MENU
from clovers.logger import logger

IMPORT_NAME = None
TOOLS = SkillCore()
DEBUG = "debug"


@TOOLS.register("enter_debug_mode", "当用户是开发者，并且进行调试或开发时调用此工具以进入调试模式。")
async def _(agent: CloversAgent, event: Event):
    session = agent.current_session(event)
    session.api = agent.api("chat")
    tips = f"\n\n[DEBUG MODE]当前调试目标: {IMPORT_NAME}" if IMPORT_NAME else "\n\n[DEBUG MODE]当前未指定调试目标，请向用户获取。"
    session.payload = session.api.build_payload(session, agent.chat_prompt + tips)
    session.payload["tools"] = [*agent.select_tools(DEBUG), *agent.select_tools(ON_CHAT), agent.manifest[SKILL_MENU]]
    return ""


@TOOLS.register(
    "reload_module",
    "重新加载已修改的工具，使开发者刚完成的代码修改立即生效。",
    {"module_name": {"type": "string", "description": "调试目标的 Python 模块名。只在变更调试目标时指定该参数。"}},
    category=DEBUG,
    required=[],
)
async def _(agent: CloversAgent, event: Event, import_name: str | None = None):
    global IMPORT_NAME
    if import_name is not None:
        IMPORT_NAME = import_name
        logger.info(f"导入路径已更新为 {IMPORT_NAME}")
    if IMPORT_NAME is None:
        return "修复目标的导入路径未初始化。请先向开发者获取"
    if IMPORT_NAME in sys.modules:
        agent.detach(sys.modules[IMPORT_NAME].TOOLS)
        agent.merge(importlib.reload(sys.modules[IMPORT_NAME]).TOOLS)
        agent.sync_menu()
    else:
        agent.load_from_list([IMPORT_NAME])
    return f"已重新加载{IMPORT_NAME}"


@TOOLS.register("get_last_error", "获取上一次执行失败产生的Traceback", category=DEBUG)
async def _(agent: CloversAgent, event: Event):
    if agent.tracebacks:
        return agent.tracebacks[-1]
    return "未发生错误"
