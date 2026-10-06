from typing import TypedDict
from pydantic import BaseModel


class MemoryItem(TypedDict):
    id: int
    content: str
    star: int


type MemoryItemTuple = tuple[str, int]


class UserProfile(BaseModel):
    nickname: str
    address_as: str = ""
    tags: str = ""
    preferences: str = ""
    impression: str = ""
    memories: list[MemoryItem] = []

    def to_markdown(self):
        self.memories.sort(key=lambda x: int(x["id"]))
        memories = "\n".join(f"{x["id"]}. {x["content"]}" for x in self.memories)
        return f"""\
# 用户档案：{self.nickname}

- 称呼: {self.address_as}
- 标签: {self.tags}
- 偏好: {self.preferences}
- 印象: {self.impression}

# 记忆碎片

{memories}
"""
