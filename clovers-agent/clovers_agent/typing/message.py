from typing import TypedDict, Literal, NotRequired

type MultimodalContentSegment = TextSegment | ImageSegment
type MultimodalContent = list[MultimodalContentSegment]
type UserMessage = TextUserMessage | MultimodalUserMessage


class TextUserMessage(TypedDict):
    """用户消息"""

    role: Literal["user"]
    content: str


class TextSegment(TypedDict):
    type: Literal["text"]
    text: str


class ImageSegment(TypedDict):
    type: Literal["image_url"]
    image_url: dict[Literal["url"], str]


class MultimodalUserMessage(TypedDict):
    """用户消息"""

    role: Literal["user"]
    content: MultimodalContent


class SystemMessage(TypedDict):
    """系统消息"""

    role: Literal["system"]
    content: str


class ToolCallFunction(TypedDict):
    """工具调用函数"""

    name: str
    arguments: str


class ToolCallInfo(TypedDict):
    """工具调用"""

    id: str
    type: Literal["function"]
    function: ToolCallFunction


class AssistantMessage(TypedDict):
    """助手消息"""

    role: Literal["assistant"]
    content: str
    tool_calls: NotRequired[list[ToolCallInfo]]


class ToolMessage(TypedDict):
    """工具消息"""

    role: Literal["tool"]
    content: str
    tool_call_id: str


type Message = UserMessage | AssistantMessage | SystemMessage | ToolMessage
