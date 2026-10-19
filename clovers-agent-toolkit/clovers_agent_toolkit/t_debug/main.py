import sys
import json
import inspect
import importlib
from collections import deque
from clovers_agent import CloversAgent, Event
from clovers_agent.core import ToolCallError
from clovers.logger import logger
from ..base import TOOLS
from typing import cast

logger.warning("Debug 模式已启用，请勿将此模式用于生产环境。")

DEBUG_UTILS = "debug_utils"
DEBUG_PROMPT = """\
1. 如工具未正常工作，必须明确告知用户未正常工作工具的完整名称。
2. `reload_module`不该在工具未正常工作后立即调用，必须在开发者要求重试此前发生错误的操作时调用。
    - 若确定调用`reload_module`，那么`reload_module`的调用必须先于其他工具，且不能并行调用其他工具。
3. 用户可以在终端查看完整的错误日志，因此除非用户主动要求分析失败信息，否则不要主动输出更多异常信息。
"""


@TOOLS.create_category(DEBUG_UTILS, "调试工具: 包含重新加载模块、查看工具源码、获取历史报错")
async def _(agent: CloversAgent, event: Event):
    tips = ["### Debug 模式"]
    if module_name := agent.extra.get("debug_module_name"):
        tips.append(f"当前 Debug 目标: {module_name}\n")
    else:
        tips.append(f"当前未指定 Debug 目标，请向用户获取。\n")
    tips.append(await list_tool_failures(agent, event))
    tips.append(DEBUG_PROMPT)
    return "\n".join(tips)


type Failure = tuple[int, str, ToolCallError]


@TOOLS.register(
    "list_tool_failures",
    "收集并编号已发生的工具调用失败。",
    category=DEBUG_UTILS,
)
async def list_tool_failures(agent: CloversAgent, event: Event) -> str:
    session = agent.current_session(event)
    if "failure_log" not in session.extra:
        session.extra["failure_log"] = deque(maxlen=10)
    failures = cast(deque[Failure], session.extra["failure_log"])

    next_id = failures[-1][0] + 1 if failures else 1
    failures.extend((i, f"{e['name']} - {e['traceback'][-1]}", e) for i, e in enumerate(agent.tool_failures, next_id))
    agent.tool_failures.clear()

    lines = ["历史报错信息（新旧排序）："]
    lines.extend(f"{i}. {summary}" for i, summary, _ in failures)
    return "\n".join(lines)


@TOOLS.register(
    "get_tool_source",
    "读取指定工具的源代码，用于分析工具失败原因或确认工具实际行为。",
    {"tool_name": {"type": "string", "description": "工具名称，如 `get_tool_source`。"}},
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
    {"failure_id": {"type": "integer", "description": "失败编号；若该错误尚未编号，请先调用 `list_tool_failures` 取得编号。"}},
    category=DEBUG_UTILS,
)
async def get_tool_failure(agent: CloversAgent, event: Event, failure_id: int) -> str:
    session = agent.current_session(event)
    if "failure_log" not in session.extra:
        listing = await list_tool_failures(agent, event)
        return f"""\
记录不存在。
请根据下方最新错误信息重新调用此工具。
{listing}"""
    failures = cast(deque[Failure], session.extra["failure_log"])
    failure = next((e for i, _, e in failures if i == failure_id), None)
    if failure is None:
        return f"没有找到编号为 {failure_id} 的错误记录。"
    name = failure["name"]
    source = await get_tool_source(agent, event, name)
    return f"""\
### 错误信息

工具名: {name}

工具源码:

```python
{source}
```

本次调用参数: {failure["args"]}

Traceback:

{"".join(failure["traceback"])} """


@TOOLS.register(
    "reload_module",
    "重新加载已修改的工具，使开发者刚完成的代码修改立即生效。",
    {"module_name": {"type": "string", "description": "调试目标的 Python 模块名。只在变更调试目标时指定该参数。"}},
    category=DEBUG_UTILS,
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
