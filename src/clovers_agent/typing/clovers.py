from clovers import EventProtocol
from typing import TypedDict, Protocol, Literal, overload


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
    image_list: list[str]
    permission: int
    skill_menu: str
    extra_context: list[str] | None

    @overload
    async def call(self, key: Literal["text"], message: str): ...

    @overload
    async def call(self, key: Literal["console"], message: list[str]): ...

    @overload
    async def call(self, key: Literal["flat_context"]) -> list[FlatContextUnit] | None: ...
