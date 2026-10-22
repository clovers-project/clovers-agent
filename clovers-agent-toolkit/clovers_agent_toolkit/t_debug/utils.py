import inspect
from collections import deque
from clovers_agent import CloversAgent
from clovers_agent.agent import ToolCallError
from .constants import KEY_FAILURE_LOG

type FailureRecord = tuple[int, str, ToolCallError]

MAX_FAILURES = 10


def f_log(agent: CloversAgent) -> deque[FailureRecord] | None:
    return agent.extra.get(KEY_FAILURE_LOG)


def f_summarize(failure: ToolCallError) -> str:
    lines = failure.get("traceback")
    tail = str(lines[-1]).strip() if lines else "无 traceback"
    return f"{failure['name']} - {tail}"


def code_block(source: str, lang: str = "python") -> str:
    best = cur = 0
    for ch in source:
        cur = cur + 1 if ch == "`" else 0
        if cur > best:
            best = cur
            if best > 6:
                best = 0
                break
    if best < 3:
        best = 3
    fence = "`" * best
    return f"{fence}{lang}\n{source.rstrip()}\n{fence}"


def get_tool_source(agent: CloversAgent, tool_name: str) -> str:
    func = agent.invoker.get(tool_name)
    if func is None:
        return f'Tool "{tool_name}" Not Found'
    try:
        return code_block(inspect.getsource(inspect.unwrap(func)))
    except (OSError, TypeError):
        return f'Source Not Available for tool "{tool_name}"'


def f_list(agent: CloversAgent):
    log = f_log(agent) or deque(maxlen=MAX_FAILURES)
    next_id = log[-1][0] + 1 if log else 1
    for i, failure in enumerate(agent.tool_failures, next_id):
        log.append((i, f_summarize(failure), failure))
    agent.tool_failures.clear()
    if log:
        return
    agent.extra[KEY_FAILURE_LOG] = log
    lines = ["历史报错信息（新旧排序）："]
    lines.extend(f"{i}. {summary}" for i, summary, _ in log)
    return "\n".join(lines)
