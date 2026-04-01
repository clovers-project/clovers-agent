from typing import TypedDict, Literal, NotRequired


type BaseType = Literal["string", "boolean", "integer", "number"]


class BaseJSONSchemaType[T: BaseType | list[BaseType]](TypedDict):
    type: T
    description: NotRequired[str]
    enum: NotRequired[list[T]]


class ArrayJSONSchemaType(TypedDict):
    type: Literal["array"]
    description: NotRequired[str]
    minItems: NotRequired[int]
    maxItems: NotRequired[int]
    items: NotRequired[BaseJSONSchemaType | list[BaseJSONSchemaType]]


class ObjectJSONSchemaType(TypedDict):
    type: Literal["object"]
    properties: NotRequired[dict[str, "JSONSchemaType"]]
    description: NotRequired[str]
    required: NotRequired[list[str]]
    additionalProperties: NotRequired[bool]


type JSONSchemaType = BaseJSONSchemaType | ArrayJSONSchemaType | ObjectJSONSchemaType
