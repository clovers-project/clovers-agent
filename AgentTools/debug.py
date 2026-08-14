import sys
import importlib
from clovers_agent import CloversAgent, Event, SkillCore
from clovers_agent.core import ON_CHAT, SKILL_MENU
from clovers_agent.config import CONFIG
from clovers.logger import logger

IMPORT_NAME = None
TOOLS = SkillCore()


@TOOLS.register("debug", "进入调试环境。当用户为开发者且正在对自己进行开发时，必须调用此方法。")
async def _(agent: CloversAgent, event: Event):
    session = agent.current_session(event)
    session.api = agent.api("chat")
    session.payload = session.api.build_payload(session, agent.chat_prompt)
    session.payload["tools"] = [*agent.select_tools("debug"), *agent.select_tools(ON_CHAT), agent.manifest[SKILL_MENU]]
    return ""


@TOOLS.register(
    "hotfix",
    "当出现了自己的工具调用没有正常工作的情况后，在开发者修复问题后调用此工具",
    {"import_name": {"type": "string", "description": '修复目标的导入名，示例：`"path.to.module"`。当需要变更导入路径时才提供此参数。'}},
    category="debug",
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
        agent.load_from_list([IMPORT_NAME])
    else:
        oldtools: SkillCore = sys.modules[IMPORT_NAME].TOOLS
        for info in oldtools.intro_tools:
            agent.intro_tools.remove(info)
        for name in oldtools.intro_invoker:
            del agent.intro_invoker[name]
        for category, hooks in oldtools.category_hooks.items():
            for hook in hooks:
                agent.category_hooks[category].remove(hook)
        for category in oldtools.categories:
            tools = agent.select_tools(category)
            for info in oldtools.select_tools(category):
                tools.remove(info)
                del agent.manifest[name]
                del agent.invoker[name]
            tools = agent.select_tools(category)
            if tools:
                continue
            del agent.categories[category]
            if category in agent.category_hooks:
                del agent.category_hooks[category]
        newtools: SkillCore = importlib.reload(sys.modules[IMPORT_NAME]).TOOLS
        agent.merge(newtools)
        agent.sync_menu()
    return "已重新加载目标模块"
