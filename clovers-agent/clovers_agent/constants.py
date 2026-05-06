from .typing import FunctionToolInfo

SYSTEM_TAG = "<system>\n{}\n</system>"
BUILTIN_CATEGORY = "builtin"
ON_CHAT = "on_chat"
ON_CHAT_DESC = "当前对话为闲聊、讨论与简单提问、或无法分配至其他工具时，调用此方法"
ON_SKILL = "on_skill"
ON_SKILL_DESC = "当用户的指令涉及外部调用时，调用此方法以进入技能执行环境。"
SKILL_MENU = "skill_menu"
SKILL_MENU_DESC = "如果助手无法独自完成用户指令，则需要调用此方法获取更多技能。"
ACTIVE_REPLY = "active_reply"
ACTIVE_REPLY_DESC = "如决策主动回复则调用此方法以进入回复环境"
VIEW_ID_IMAGE_INFO: FunctionToolInfo = {
    "type": "function",
    "function": {
        "name": "view_id_image",
        "description": "当助手认为自己需要查看上下文中格式为 [image:image_id] 的图片时调用此方法。",
        "parameters": {"type": "object", "properties": {"image_id": {"type": "integer"}}, "required": ["image_id"]},
    },
}
