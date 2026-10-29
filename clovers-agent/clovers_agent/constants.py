from .typing import FunctionToolInfo
from .config import CONSTANT

# constant
SYSTEM_TAG = CONSTANT.SYSTEM_TAG
USER_TAG = CONSTANT.USER_TAG
ASSISTANT_TAG = CONSTANT.ASSISTANT_TAG
HISTORY_TAG = CONSTANT.HISTORY_TAG
TOOLCALL_PROMPT = CONSTANT.TOOLCALL_PROMPT
ON_CHAT = CONSTANT.ON_CHAT
ON_CHAT_DESC = CONSTANT.ON_CHAT_DESC
HIDDEN_CATEGORY = CONSTANT.HIDDEN_CATEGORY
BUILTIN_CATEGORY = CONSTANT.BUILTIN_CATEGORY
INTRINSIC_CATEGORY = CONSTANT.INTRINSIC_CATEGORY
LOAD_TOOLS = CONSTANT.LOAD_TOOLS
LOAD_TOOLS_DESC = CONSTANT.LOAD_TOOLS_DESC
READ_IMAGE_BY_ID = CONSTANT.READ_IMAGE_BY_ID
READ_IMAGE_BY_ID_INFO: FunctionToolInfo = {
    "type": "function",
    "function": {
        "name": READ_IMAGE_BY_ID,
        "description": CONSTANT.READ_IMAGE_BY_ID_DESC,
        "parameters": {
            "type": "object",
            "properties": {"image_id": {"type": "integer", "description": CONSTANT.READ_IMAGE_BY_ID_IMAGE_ID_DESC}},
            "required": ["image_id"],
        },
    },
}
INJECT_PROMPT = CONSTANT.INJECT_PROMPT
INJECT_PROMPT_INFO: FunctionToolInfo = {
    "type": "function",
    "function": {
        "name": INJECT_PROMPT,
        "description": CONSTANT.INJECT_PROMPT_DESC,
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": CONSTANT.INJECT_PROMPT_PROMPT_DESC}},
            "required": ["prompt"],
        },
    },
}
COMPACT_CONTEXT = CONSTANT.COMPACT_CONTEXT
COMPACT_CONTEXT_INFO: FunctionToolInfo = {
    "type": "function",
    "function": {
        "name": COMPACT_CONTEXT,
        "description": CONSTANT.COMPACT_CONTEXT_DESC,
        "parameters": {
            "type": "object",
            "properties": {
                "overview": {"type": "string", "description": CONSTANT.COMPACT_CONTEXT_OVERVIEW_DESC},
                "keywords": {"type": "array", "description": CONSTANT.COMPACT_CONTEXT_KEYWORDS_DESC, "items": {"type": "string"}},
                "summary": {"type": "string", "description": CONSTANT.COMPACT_CONTEXT_SUMMARY_DESC},
            },
            "required": ["overview", "keywords", "summary"],
        },
    },
}
VISION_TAG = CONSTANT.VISION_TAG
VISION_PROMPT = CONSTANT.VISION_PROMPT
EXECUTE_SCRIPT = CONSTANT.EXECUTE_SCRIPT
EXECUTE_SCRIPT_DEFAULT_TIMEOUT = CONSTANT.EXECUTE_SCRIPT_DEFAULT_TIMEOUT
EXECUTE_SCRIPT_MAX_TIMEOUT = CONSTANT.EXECUTE_SCRIPT_MAX_TIMEOUT
EXECUTE_SCRIPT_MAX_TAIL_LEN = CONSTANT.EXECUTE_SCRIPT_MAX_TAIL_LEN
EXECUTE_SCRIPT_DESC = CONSTANT.EXECUTE_SCRIPT_DESC
READ_REFERENCE = CONSTANT.READ_REFERENCE
READ_REFERENCE_DESC = CONSTANT.READ_REFERENCE_DESC
WAKEUP_TIP = CONSTANT.WAKEUP_TIP.strip() + "\n"
