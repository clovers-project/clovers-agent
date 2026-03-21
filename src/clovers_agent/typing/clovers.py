from clovers import EventProtocol
from typing import Protocol, Literal, overload


class Event(EventProtocol, Protocol):
    user_id: str
    group_id: str | None
    nickname: str
    to_me: bool
    image_list: list[str]
    permission: int
    skill_menu: str

    @overload
    async def call(self, key: Literal["text"], message: str): ...

    @overload
    async def call(self, key: Literal["console"], message: list[str]): ...
