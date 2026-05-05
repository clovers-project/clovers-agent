from .typing import FunctionToolInfo

ON_CHAT = "on_chat"
ON_SKILL = "on_skill"
SKILL_MENU = "skill_menu"
DECISION_TOOL: list[FunctionToolInfo] = [
    {"type": "function", "function": {"name": "active_reply", "description": "如决策主动回复则调用此方法以进入回复环境"}}
]
