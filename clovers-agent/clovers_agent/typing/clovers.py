from clovers import EventProtocol
from typing import TypedDict, Protocol, Literal, Any, overload
from collections.abc import Coroutine


class FlatContextUnit(TypedDict):
    nickname: str
    user_id: str
    text: str
    images: list[str]


class Event(EventProtocol, Protocol):
    user_id: str
    group_id: str | None
    nickname: str
    to_me: bool
    at: list[str]
    image_list: list[str]
    permission: int
    extra_context: list[str]

    @overload
    async def call(self, key: Literal["text"], message: str): ...

    @overload
    async def call(self, key: Literal["console"], message: list[str]): ...

    @overload
    def call(self, key: Literal["flat_context"]) -> Coroutine[Any, Any, list[FlatContextUnit]] | None: ...
