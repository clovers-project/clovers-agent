from .typing import FunctionToolInfo

SYSTEM_TAG = "<system>\n{}\n</system>"
BUILTIN_CATEGORY = "builtin"
ON_CHAT = "on_chat"
ON_CHAT_DESC = "当前对话为闲聊、讨论与简单提问、或无法分配至其他工具时，调用此方法"
ON_SKILL = "on_skill"
ON_SKILL_DESC = "当用户的指令为执行具体任务时调用此方法以进入技能执行环境。"
SKILL_MENU = "skill_menu"
SKILL_MENU_DESC = "如果助手无法独自完成用户指令，则需要调用此方法获取更多技能。"
ACTIVE_REPLY = "active_reply"
ACTIVE_REPLY_DESC = "如决策主动回复则调用此方法以进入回复环境"
GET_IMAGE_BY_ID_INFO: FunctionToolInfo = {
    "type": "function",
    "function": {
        "name": "get_image_by_id",
        "description": "上下文中的图片已替换成格式为 [image:image_id] 的标签，"
        "当用户的话题引用上述图片或助手认为自己需要查看该图片时调用此方法。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "注意不要向用户透露有关图片标签的事实，image_id 只能在上下文中获取。",
                },
            },
            "required": ["image_id"],
        },
    },
}
VISION_PROMPT = """\
你是一位专业的视觉分析专家。
你的任务是观察用户提供的一个或多个图像，并将其内容转化为详尽、准确且具有逻辑性的文字描述。
这些描述将被用作后续纯文本模型的上下文参考，因此你的描述必须足够清晰。

对于每一张图片，请考虑以下维度：
- 用户的重点需求。
- 图片的主体与背景，各元素的颜色、材质、形状、光影效果、元素之间的空间位置关系
- 如果图像中包含任何文字，请准确识别并记录。
- 图片类型：如油画、照片、图表、截屏等。
- 如果图片是摄影、插画、艺术作品等，则需要进行解读，如结构、风格、主题、情绪等

你的输出需要遵循以下要求：
- 如果有多张图片，则需要在描述中进行区分。
- 描述应当足够具体。
- **不要直接回复用户**。你的输出仅包含对图片的描述。

请现在开始分析图像并生成描述。
"""
