from typing import TypedDict, Literal, NotRequired


class TextSegment(TypedDict):
    type: Literal["text"]
    text: str


class ImageSegment(TypedDict):
    type: Literal["image_url"]
    image_url: dict[Literal["url"], str]


type ContentSegment = TextSegment | ImageSegment


class SystemMessage(TypedDict):
    """系统消息"""

    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    """用户消息"""

    role: Literal["user"]
    content: str | list[ContentSegment]


class AssistantMessage(TypedDict):
    """助手消息"""

    role: Literal["assistant"]
    content: str
    tools: NotRequired[list[dict]]


class ToolMessage(TypedDict):
    """工具消息"""

    role: Literal["tool"]
    content: str
    tool_call_id: str


type Message = UserMessage | AssistantMessage | SystemMessage | ToolMessage
