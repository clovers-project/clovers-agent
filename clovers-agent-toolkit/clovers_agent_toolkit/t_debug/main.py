import importlib
import inspect
import sys
from collections import deque
from typing import Any

from clovers.logger import logger
from clovers_agent import CloversAgent, Event
from clovers_agent.core import ToolCallError

from ..base import TOOLS

logger.warning("Debug 模式已启用，请勿将此模式用于生产环境。")

DEBUG_UTILS = "debug_utils"
FAILURE_LOG_KEY = "failure_log"
MAX_FAILURES = 10

DEBUG_PROMPT = """\
1. 如工具未正常工作，必须明确告知用户未正常工作工具的完整名称。
2. 不要在工具报错后立即调用 `reload_module`，只有在开发者确认已修复并明确要求重试时才调用。
   - 一旦决定调用 `reload_module`，它必须是本轮唯一的工具调用，不能与其他工具并行。
3. 用户可在终端查看完整错误日志，除非用户主动要求分析失败信息，否则不要主动展开异常细节。
"""

FailureRecord = tuple[int, str, ToolCallError]


@TOOLS.create_category(DEBUG_UTILS, "调试工具：重新加载模块、查看工具源码、查看工具失败详情")
async def _(agent: CloversAgent, event: Event) -> str:
    sections = ["### Debug 模式"]
    if module_name := agent.extra.get("debug_module_name"):
        sections.append(f"当前 Debug 目标: {module_name}")
    else:
        sections.append("当前未指定 Debug 目标，请向用户获取。")
    sections.append(await list_tool_failures(agent, event))
    sections.append(DEBUG_PROMPT)
    return "\n\n".join(sections)


def _failure_log(session: Any) -> deque[FailureRecord]:
    log = session.extra.get(FAILURE_LOG_KEY)
    if log is None:
        log = deque(maxlen=MAX_FAILURES)
        session.extra[FAILURE_LOG_KEY] = log
    return log


def _summarize(failure: ToolCallError) -> str:
    lines = failure.get("traceback") or []
    tail = str(lines[-1]).strip() if lines else "无 traceback"
    return f"{failure['name']} - {tail}"


def _code_block(source: str, lang: str = "python") -> str:
    """源码里若含 ``` 就用更长的围栏，避免提前闭合。"""
    fence = "````" if "```" in source else "```"
    return f"{fence}{lang}\n{source.rstrip()}\n{fence}"


@TOOLS.register(
    "list_tool_failures",
    "收集并编号尚未编号的工具调用失败，返回按发生时间排序的列表（编号、工具名、异常摘要）。",
    category=DEBUG_UTILS,
)
async def list_tool_failures(agent: CloversAgent, event: Event) -> str:
    log = _failure_log(agent.current_session(event))
    next_id = log[-1][0] + 1 if log else 1
    for i, failure in enumerate(agent.tool_failures, next_id):
        log.append((i, _summarize(failure), failure))
    agent.tool_failures.clear()

    if not log:
        return "暂无工具调用失败记录。"
    lines = ["历史报错信息（新旧排序）："]
    lines.extend(f"{i}. {summary}" for i, summary, _ in log)
    return "\n".join(lines)


@TOOLS.register(
    "get_tool_source",
    "读取指定工具的源代码。",
    {"tool_name": {"type": "string", "description": "工具名称，须为已注册工具，如 `list_tool_failures`。"}},
    category=DEBUG_UTILS,
)
async def get_tool_source(agent: CloversAgent, event: Event, tool_name: str) -> str:
    func = agent.invoker.get(tool_name)
    if func is None:
        return f"未找到名为 {tool_name} 的工具"
    try:
        return inspect.getsource(inspect.unwrap(func))
    except (OSError, TypeError):
        return f"获取工具 {tool_name} 的源码时发生错误。"


@TOOLS.register(
    "get_tool_failure",
    "按编号查看某次工具失败的详情：工具名、该工具源码、调用参数与完整 traceback。",
    {"failure_id": {"type": "integer", "description": "失败编号；若该失败尚未编号，请先调用 `list_tool_failures`。"}},
    category=DEBUG_UTILS,
)
async def get_tool_failure(agent: CloversAgent, event: Event, failure_id: int) -> str:
    session = agent.current_session(event)
    if FAILURE_LOG_KEY not in session.extra:
        listing = await list_tool_failures(agent, event)
        return f"该失败尚未编号，已重新收集：\n{listing}\n请用返回的编号再次调用 `get_tool_failure`。"

    log = _failure_log(session)
    failure = next((f for i, _, f in log if i == failure_id), None)
    if failure is None:
        return f"没有找到编号为 {failure_id} 的失败记录；它可能已超出保留上限被丢弃。"

    name = failure["name"]
    source = await get_tool_source(agent, event, name)
    traceback_text = "".join(failure["traceback"]).rstrip()
    return f"""### 失败详情

工具名: {name}

工具源码:

{_code_block(source)}

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
async def _(agent: CloversAgent, event: Event, module_name: str | None = None) -> str:
    if module_name is not None:
        agent.extra["debug_module_name"] = module_name
        logger.info("调试目标模块已切换为 %s", module_name)

    target = agent.extra.get("debug_module_name")
    if not target:
        return "目标模块名未设置，请先向开发者获取。"

    if target not in sys.modules:
        agent.load_from_list([target])
        return f"已加载模块 {target}"

    previous_tools = getattr(sys.modules[target], "TOOLS", None)
    if previous_tools is not None:
        agent.detach(previous_tools)

    prefix = f"{target}."
    for name in [n for n in sys.modules if n == target or n.startswith(prefix)]:
        del sys.modules[name]

    try:
        module = importlib.import_module(target)
        new_tools = module.TOOLS
    except Exception as exc:
        # 导入失败时回滚，否则工具集会整个消失，且 sys.modules 里也不再有目标模块
        logger.exception("重新加载模块 %s 失败", target)
        if previous_tools is not None:
            agent.merge(previous_tools)
        agent.sync_menu()
        return f"重新加载模块 {target} 失败：{type(exc).__name__}: {exc}"

    agent.merge(new_tools)
    agent.sync_menu()
    return f"已重新加载模块 {target}"
