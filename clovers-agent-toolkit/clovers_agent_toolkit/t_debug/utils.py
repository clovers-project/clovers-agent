import inspect
from collections import deque
from clovers_agent import CloversAgent
from clovers_agent.agent import ToolCallError
from .constants import KEY_FAILURE_LOG

type FailureRecord = tuple[int, str, ToolCallError]

MAX_FAILURES = 10


def f_log(agent: CloversAgent) -> deque[FailureRecord]:
    if KEY_FAILURE_LOG in agent.extra:
        log = agent.extra[KEY_FAILURE_LOG]
    else:
        log = agent.extra[KEY_FAILURE_LOG] = deque(maxlen=MAX_FAILURES)
    return log


def f_summarize(failure: ToolCallError) -> str:
    lines = failure.get("traceback")
    tail = str(lines[-1]).strip() if lines else "无 traceback"
    return f"{failure['name']} - {tail}"


def list_tool_failures(agent: CloversAgent):
    log = f_log(agent)
    next_id = log[-1][0] + 1 if log else 1
    log.extend((i, f_summarize(failure), failure) for i, failure in enumerate(agent.tool_failures, next_id))
    print(log)
    print(agent.tool_failures)
    agent.tool_failures.clear()
    if not log:
        return
    lines = ["历史报错信息（按发生时间从新到旧排列，编号越大表示越新）"]
    lines.extend(f"{i}. {summary}" for i, summary, _ in reversed(log))
    return "\n".join(lines)


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
