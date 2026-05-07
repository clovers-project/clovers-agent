from .typing import FunctionToolInfo
from .config import CONSTANT

# constant
SYSTEM_TAG = CONSTANT.system_tag
ON_CHAT = CONSTANT.on_chat
ON_CHAT_DESC = CONSTANT.on_chat_desc
ON_SKILL = CONSTANT.on_skill
ON_SKILL_DESC = CONSTANT.on_chat_desc
ACTIVE_REPLY = CONSTANT.active_reply
ACTIVE_REPLY_DESC = CONSTANT.active_reply_desc
BUILTIN_CATEGORY = CONSTANT.builtin_category
SKILL_MENU = CONSTANT.skill_menu
SKILL_MENU_DESC = CONSTANT.skill_menu_desc
GET_IMAGE_BY_ID = CONSTANT.get_image_by_id
GET_IMAGE_BY_ID_INFO: FunctionToolInfo = {
    "type": "function",
    "function": {
        "name": GET_IMAGE_BY_ID,
        "description": CONSTANT.get_image_by_id_desc,
        "parameters": {
            "type": "object",
            "properties": {"image_id": {"type": "integer", "description": CONSTANT.get_image_by_id_image_id}},
            "required": ["image_id"],
        },
    },
}
VISION_TAG = CONSTANT.vision_tag
VISION_PROMPT = CONSTANT.vision_prompt
