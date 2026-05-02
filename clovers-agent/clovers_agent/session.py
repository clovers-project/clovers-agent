import asyncio
from collections import deque
from .embedding import SentenceTransformer, TopicDecoupler
from .typing import Payload
from .typing.message import UserMessage, AssistantMessage, ContentSegment


def extract_plain_text(content: str | list[ContentSegment]) -> str:
    return content if isinstance(content, str) else "\n".join(text["text"] for text in content if text["type"] == "text")


def char_count(content: str | list[ContentSegment]):
    return len(content) if isinstance(content, str) else sum(len(text["text"]) for text in content if text["type"] == "text")


class ContextRecoder:
    """会话上下文管理器"""

    records: deque[tuple[UserMessage, AssistantMessage]]

    def __init__(self, size: int) -> None:
        self.records = deque(maxlen=size)
        self.lock = asyncio.Lock()

    def __iter__(self):
        for record in self.records:
            yield record[0]
            yield record[1]

    def __bool__(self):
        return bool(self.records)

    def over(self, request: UserMessage, reply: AssistantMessage):
        self.records.append((request, reply))

    def clear(self):
        self.records.clear()


class Session(ContextRecoder):
    type Storge = deque[tuple[UserMessage, AssistantMessage, int | float]]
    records: Storge
    silence: deque[tuple[str, float]]
    storage: Storge
    unimp_storage: Storge
    snap: ContextRecoder
    extra: dict

    def __init__(self, size: int, sentence_model: SentenceTransformer) -> None:
        # 标准记录
        super().__init__(size)
        self.silence = deque()
        # 临时记录
        self.snap = ContextRecoder(size)
        # 状态
        self.extra = {}
        # 不重要信息
        self.unimportant = False
        self.storage = self.records
        self.unimp_storage = deque(maxlen=2)
        # 主题分离
        self.decoupler = TopicDecoupler(sentence_model)

    def memory_filter(self, timeout: int | float):
        """过滤记忆"""
        while self.records and (self.records[0][2] <= timeout):
            self.records.popleft()

    def silence_filter(self, timeout: int | float):
        """过滤静默记录群聊上下文"""
        while self.silence and (self.silence[0][1] <= timeout):
            self.silence.popleft()

    def step(self, message: str):
        if not self.decoupler.step(message):
            return False
        count = sum(char_count(msg["content"]) for msg in self)
        return count > 800

    def activate(self, model: str, content: list[ContentSegment]):
        self.current_input = content  # 注入输入（可修改）
        self.is_first_wait: bool = True
        self.used: set[str] = set()
        self.payload: Payload = {"model": model, "messages": [{"role": "system"}, *self, {"role": "user", "content": self.current_input}]}  # type: ignore
        self.skill_menu: str | None = None

    def inactivate(self):
        self.snap.clear()
        del self.current_input
        del self.is_first_wait
        del self.payload
        del self.used
        del self.skill_menu

    def over(self, request: UserMessage, reply: AssistantMessage, timestamp: float):
        """处理完成"""
        if self.unimportant:
            self.records = self.unimp_storage
            self.unimportant = False
        elif self.unimp_storage:
            self.storage.extend(self.unimp_storage)
            self.unimp_storage.clear()
            self.records = self.storage
        self.records.append((request, reply, timestamp))
        self.silence.clear()

    def clear(self):
        """清理记录"""
        self.storage.clear()
        self.unimp_storage.clear()
        self.silence.clear()

    def sync_snap(self):
        """同步上下文到辅助AI"""
        self.snap.clear()
        self.snap.records.extend((x, y) for x, y, _ in self.records)
