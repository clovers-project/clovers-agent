import asyncio
from collections import deque, Counter
from .embedding import SentenceTransformer, TopicDecoupler
from typing import Iterable
from .typing import Payload, Message, UserMessage, AssistantMessage
from .typing.message import ContentSegment


def extract_plain_text(content: str | list[ContentSegment]) -> str:
    return content if isinstance(content, str) else "\n".join(text["text"] for text in content if text["type"] == "text")


def char_count(content: str | list[ContentSegment]):
    return len(content) if isinstance(content, str) else sum(len(text["text"]) for text in content if text["type"] == "text")


class ContextRecoder[A, B, *Args]:
    """会话上下文管理器"""

    records: deque[tuple[A, B, *Args]]

    def __init__(self, size: int) -> None:
        self.records = deque(maxlen=size)
        self.lock = asyncio.Lock()

    def __iter__(self):
        for record in self.records:
            yield record[0]
            yield record[1]

    def __bool__(self):
        return bool(self.records)

    def over(self, *args):
        self.records.append(args)

    def clear(self):
        self.records.clear()


class Session(ContextRecoder[UserMessage, AssistantMessage, float]):
    silence: deque[tuple[str, float]]
    unimp_rec: tuple[UserMessage, AssistantMessage, float] | None
    snap: ContextRecoder[UserMessage, AssistantMessage]
    extra: dict

    def __init__(self, size: int, sentence_model: SentenceTransformer) -> None:
        # 标准记录
        super().__init__(size)
        self.silence = deque()
        # 临时记录
        self.snap = ContextRecoder(size)
        # 状态
        self.extra = {}
        self.usage_counter = Counter[str]()
        # 不重要信息
        self._unimp = False
        self.unimp_rec = None
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
        self.payload: Payload = {"model": model, "messages": [{"role": "system"}, *self]}  # type: ignore
        if self.unimp_rec:
            self.payload["messages"].append(self.unimp_rec[0])
            self.payload["messages"].append(self.unimp_rec[1])
        self.payload["messages"].append({"role": "user", "content": self.current_input})
        self.skill_menu: str | None = None
        self.usage_counter.clear()
        self.unit_prompt: str
        self.mark = 0

    def inactivate(self):
        self.snap.clear()
        del self.current_input
        del self.is_first_wait
        del self.payload
        del self.used
        del self.skill_menu
        del self.unit_prompt
        del self.mark

    @property
    def system_message(self):
        return self.payload["messages"][0]

    def unimportant(self):
        if self._unimp:
            return
        self._unimp = True
        if self.unimp_rec:
            messages = self.unimp_rec[:2]
        elif self.records:
            messages = self.records[-1][:2]
        else:
            messages = ()
        self.replace_contaxt(messages)

    def replace_contaxt(self, messags: Iterable[Message]):
        messages = [self.system_message, *messags]
        mark = self.mark or (len(self.records) * 2 + 1)
        self.mark = len(messages)
        messages.extend(self.payload["messages"][mark:])
        self.payload["messages"] = messages

    def over(self, request: UserMessage, reply: AssistantMessage, timestamp: float):
        """处理完成"""
        if self._unimp:
            self.unimp_rec = (request, reply, timestamp)
            self._unimp = False
        else:
            if self.unimp_rec:
                self.records.append(self.unimp_rec)
                self.unimp_rec = None
            self.records.append((request, reply, timestamp))
        self.silence.clear()

    def clear(self):
        """清理记录"""
        self.unimp_rec = None
        self.silence.clear()

    def sync_snap(self):
        """同步上下文到辅助AI"""
        self.snap.clear()
        self.snap.records.extend((x, y) for x, y, _ in self.records)
