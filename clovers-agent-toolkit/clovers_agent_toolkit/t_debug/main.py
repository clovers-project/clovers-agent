import sys
import importlib

from clovers.logger import logger
from clovers_agent import CloversAgent, Turn, Event, AgentTools

from ..base import TOOLS
from .utils import f_log, list_tool_failures, get_tool_source
from .constants import DEBUG_UTILS, KEY_DEBUG_TARGET, DEBUG_PROMPT

from typing import cast

logger.warning("Debug 模式已启用，请勿将此模式用于生产环境。")


@TOOLS.create_category(DEBUG_UTILS, "调试工具：重新加载模块、查看工具源码、查看工具失败详情")
async def _(agent: CloversAgent, turn: Turn, event: Event):
    sections = ["### Debug 模式"]
    if module_name := agent.extra.get(KEY_DEBUG_TARGET):
        sections.append(f"当前 Debug 目标: {module_name}")
    else:
        sections.append("当前未指定 Debug 目标，请在第一次调用 `reload_module` 时传入 `module_name`")
    if listing := list_tool_failures(agent):
        sections.append(listing)
    sections.append(DEBUG_PROMPT)
    return "\n\n".join(sections)


@TOOLS.register(
    "list_tool_failures",
    "收集并编号尚未编号的工具调用失败，返回按发生时间排序的列表（格式为 `编号. 工具名 - 异常信息`）。",
    category=DEBUG_UTILS,
)
async def _(agent: CloversAgent, turn: Turn, event: Event):
    return list_tool_failures(agent) or "暂无工具调用失败记录。"


@TOOLS.register(
    "get_tool_source",
    "读取指定工具的源代码。",
    {"tool_name": {"type": "string", "description": "工具名称，须为已注册工具，如 `list_tool_failures`。"}},
    category=DEBUG_UTILS,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, tool_name: str) -> str:
    return get_tool_source(agent, tool_name)


@TOOLS.register(
    "get_tool_failure",
    "按编号查看某次工具失败的详情：工具名、该工具源码、调用参数与完整 traceback。",
    {"failure_id": {"type": "integer", "description": "若该失败尚未编号，请先调用 `list_tool_failures`。"}},
    category=DEBUG_UTILS,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, failure_id: int) -> str:
    log = f_log(agent)
    if not log:
        listing = list_tool_failures(agent)
        if not listing:
            return "暂无工具调用失败记录。"
        else:
            return f"该失败尚未编号，已重新收集：\n{listing}\n请用返回的编号再次调用 `get_tool_failure`。"
    failure = next((f for i, _, f in log if i == failure_id), None)
    if failure is None:
        return f"没有找到编号为 {failure_id} 的失败记录；它可能已超出保留上限被丢弃。"
    name = failure["name"]
    source = get_tool_source(agent, name)
    traceback_text = "".join(failure["traceback"]).rstrip()
    return f"""### 失败详情

工具名: {name}

工具源码:

{source}

本次调用参数: {failure["args"]}

Traceback:

{traceback_text}
"""


@TOOLS.register(
    "reload_module",
    "重新加载指定的 Python 模块，使开发者刚完成的代码修改立即生效。",
    {"module_name": {"type": "string", "description": "调试目标的模块名；只在切换调试目标时提供，省略则沿用上次的目标。"}},
    category=DEBUG_UTILS,
    required=[],
)
async def _(agent: CloversAgent, turn: Turn, event: Event, module_name: str | None = None) -> str:
    if module_name is not None:
        agent.extra[KEY_DEBUG_TARGET] = module_name.strip().replace("-", "_")
        logger.info("调试目标模块已切换为 %s", module_name)
    target = agent.extra.get(KEY_DEBUG_TARGET)
    if not target:
        return "目标模块名未设置，请先向开发者获取。"
    if target not in sys.modules:
        logger.warning("模块 %s 未加载，将尝试加载该模块。", target)
        agent.load_from_list([target])
        return f"已加载模块 {target}"
    tools = cast(AgentTools, getattr(sys.modules[target], "TOOLS"))
    agent.detach(tools)
    prefix = f"{target}."
    modules = [name for name in sys.modules if name == target or name.startswith(prefix)]
    logger.debug("reload_module: 待重载模块: %s", modules)
    deletion = [(name, sys.modules.pop(name)) for name in modules]
    try:
        tools = cast(AgentTools, getattr(importlib.import_module(target), "TOOLS"))
        info = f"已重新加载模块 {target}"
    except Exception as exc:
        info = f"重新加载模块 {target} 失败：{type(exc).__name__}: {exc}"
        logger.error(info)
        sys.modules.update(deletion)
    agent.merge(tools)
    agent.sync_menu()
    return info
