from typing import TypedDict, Literal, NotRequired


class TextSegment(TypedDict):
    type: Literal["text"]
    text: str


class ImageSegment(TypedDict):
    type: Literal["image_url"]
    image_url: dict[Literal["url"], str]


type ContentSegment = TextSegment | ImageSegment


class ChatMessage[Role: Literal["system", "user", "assistant"]](TypedDict):
    """对话消息"""

    role: Role
    content: str | list[ContentSegment]
    tools: NotRequired[list[dict]]


class ToolMessage(TypedDict):
    """工具消息"""

    role: Literal["tool"]
    content: str
    tool_call_id: str


type Message = ChatMessage | ToolMessage
