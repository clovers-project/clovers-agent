import sys
import json
import inspect
import importlib
from clovers_agent import CloversAgent, Event
from clovers_agent.core import SKILL_MENU
from clovers.logger import logger
from .toolkit import TOOLS

logger.warning("Debug 模式已启用，请勿将此模式用于生产环境。")

DEBUG_TOOLS = "debug_tools"
DEBUG_PROMPT = """\
### Debug 模式
{tips}

1. 如工具未正常工作，必须明确告知用户未正常工作工具的完整名称。
2. `reload_module`不该在工具未正常工作后立即自行调用，必须在开发者要求重试此前发生错误的操作时调用。
    - 若确定调用`reload_module`，那么`reload_module`的调用必须先于其他工具，且不能并行调用其他工具。
3. 用户可以在终端查看完整的错误日志，因此除非用户主动要求分析失败信息，否则不要主动输出更多异常信息。
4. 禁止向用户做多余的说明
"""


@TOOLS.register("on_debug", "当用户进行测试、调试或开发时调用此工具以进入调试模式。")
async def _(agent: CloversAgent, event: Event):
    session = agent.current_session(event)
    session.api = agent.api("chat")
    if not (module_name := agent.extra.get("debug_module_name")):
        tips = f"当前 Debug 目标: {module_name}"
    else:
        tips = "当前未指定 Debug 目标，请向用户获取。"
    system_prompt = f"{agent.style_prompt}\n{DEBUG_PROMPT.format(tips=tips)}"
    session.payload = session.api.build_payload(session, system_prompt)
    session.payload["tools"] = [*agent.select_tools(DEBUG_TOOLS), agent.manifest[SKILL_MENU]]
    return ""


@TOOLS.create_category(DEBUG_TOOLS, "调试工具: 包含重新加载模块、查看工具源码、获取上次报错")
async def _(agent: CloversAgent, event: Event):
    if not (module_name := agent.extra.get("debug_module_name")):
        tips = f"当前 Debug 目标: {module_name}"
    else:
        tips = "当前未指定 Debug 目标，请向用户获取。"
    return DEBUG_PROMPT.format(tips=tips)


@TOOLS.register(
    "reload_module",
    "重新加载已修改的工具，使开发者刚完成的代码修改立即生效。",
    {"module_name": {"type": "string", "description": "调试目标的 Python 模块名。只在变更调试目标时指定该参数。"}},
    category=DEBUG_TOOLS,
    required=[],
)
async def _(agent: CloversAgent, event: Event, module_name: str | None = None):

    if module_name is not None:
        agent.extra["debug_module_name"] = module_name
        logger.info(f"目标模块已更新为 {module_name}")
    if not (module_name := agent.extra.get("debug_module_name")):
        return "目标模块名未设置，请先向开发者获取"
    if module_name in sys.modules:
        agent.detach(sys.modules[module_name].TOOLS)
        prefix = f"{module_name}."
        del_modules = [name for name in sys.modules if name.startswith(prefix)]
        del_modules.append(module_name)
        for name in del_modules:
            del sys.modules[name]
        agent.merge(importlib.import_module(module_name).TOOLS)
        agent.sync_menu()
    else:
        agent.load_from_list([module_name])
    return f"已重新加载{module_name}"


@TOOLS.register("get_last_error", "获取上一次工具执行失败产生的失败记录", category=DEBUG_TOOLS)
async def _(agent: CloversAgent, event: Event):
    if agent.tool_failures:
        return json.dumps(agent.tool_failures[0], ensure_ascii=False)
    return "None"


@TOOLS.register(
    "get_tool_source",
    "获取指定工具的源代码。",
    {"tool_name": {"type": "string", "description": "工具名称"}},
    category=DEBUG_TOOLS,
)
async def _(agent: CloversAgent, event: Event, tool_name) -> str:
    func = agent.invoker.get(tool_name)
    if func is None:
        return f"工具 {tool_name} 不存在"
    try:
        return inspect.getsource(inspect.unwrap(func))
    except (OSError, TypeError):
        return f"无法获取工具 {tool_name} 的源码"
