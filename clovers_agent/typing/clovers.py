from clovers import EventProtocol
from typing import Protocol


class Event(EventProtocol, Protocol):
    user_id: str
    group_id: str | None
    nickname: str
    to_me: bool
    image_list: list[str]
    skill_menu: str
