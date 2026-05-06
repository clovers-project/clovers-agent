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
        "parameters": {
            "type": "object",
            "properties": {"image_id": {"type": "integer"}},
            "required": ["image_id"],
        },
    },
}
VISION_PROMPT = """\
你是一位专业的图像分析专家。你的任务是观察并描述提供给你的图像。你的描述将作为后续的纯文本模型提供关于这些图像的上下文。

对于每一张图片，请考虑以下维度：
1. 主体与背景，各元素的颜色、材质、形状、光影效果、元素之间的空间位置关系
2. 如果图像中包含任何文字，请准确识别并记录。
3. 图片类型：如照片、插画、图表、截屏等。

你的输出请遵循以下格式要求：
- 如果有多张图片，请在描述中进行区分。
- 描述应当详尽且具有逻辑性，确保可以依据描述
- 保持客观中立。

请开始你的分析和描述。
"""
