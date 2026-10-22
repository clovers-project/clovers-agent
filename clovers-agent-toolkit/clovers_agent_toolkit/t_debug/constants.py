from ..config import CATEGORIES

DEBUG_UTILS = CATEGORIES.DEBUG_UTILS
KEY_DEBUG_TARGET = f"{DEBUG_UTILS}.debug_target"
KEY_FAILURE_LOG = f"{DEBUG_UTILS}.failure_log"

DEBUG_PROMPT = """\
1. 如工具未正常工作，必须明确告知用户未正常工作工具的完整名称。
2. 不要在工具报错后立即调用 `reload_module`，只有在开发者确认已修复并明确要求重试时才调用。
   - 一旦决定调用 `reload_module`，它必须是本轮唯一的工具调用，不能与其他工具并行。
3. 用户可在终端查看完整错误日志，除非用户主动要求分析失败信息，否则不要主动展开异常细节。
"""
