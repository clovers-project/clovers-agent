from typing import Literal, TypedDict, NotRequired
from .message import Message
from .json_schema import JSONSchemaType


class FunctionToolDefinition(TypedDict):
    name: str
    """工具名称`"""
    description: str
    """工具描述`"""
    parameters: NotRequired[JSONSchemaType]
    """工具参数`"""


class FunctionToolInfo(TypedDict):
    type: Literal["function"]
    function: FunctionToolDefinition


class JsonSchemaFormat(TypedDict):
    name: str
    strict: bool
    schema: JSONSchemaType


class ResponseFormatJsonSchema(TypedDict):
    type: Literal["json_schema"]
    json_schema: JsonSchemaFormat


type ResponseFormatJsonObject = dict[Literal["type"], Literal["json_object"]]

type ResponseFormat = ResponseFormatJsonSchema | ResponseFormatJsonObject


class Payload(TypedDict):
    model: str
    messages: list[Message]
    tools: NotRequired[list[FunctionToolInfo]]
    response_format: NotRequired[ResponseFormat]
