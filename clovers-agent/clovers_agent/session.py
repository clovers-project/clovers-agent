import asyncio
from collections import deque
from .api import OpenAIAPI
from .embedding import SentenceTransformer, TopicDecoupler
from typing import Iterable
from .typing import Payload, Message, UserMessage, AssistantMessage, SystemMessage
from .typing.message import ContentSegment


def extract_plain_text(content: str | list[ContentSegment]) -> str:
    return content if isinstance(content, str) else "\n".join(text["text"] for text in content if text["type"] == "text")


def char_count(content: str | list[ContentSegment]):
    return len(content) if isinstance(content, str) else sum(len(text["text"]) for text in content if text["type"] == "text")


type Record = tuple[UserMessage, AssistantMessage, float]


class ContextRecoder:
    """会话上下文管理器"""

    recorder: deque[Record]

    def __init__(self, size: int) -> None:
        self.recorder = deque(maxlen=size)
        self.lock = asyncio.Lock()

    def __iter__(self):
        for record in self.recorder:
            yield record[0]
            yield record[1]

    def __bool__(self):
        return bool(self.recorder)

    def over(self, request: UserMessage, reply: AssistantMessage, timestamp: float):
        self.recorder.append((request, reply, timestamp))

    def clear(self):
        self.recorder.clear()


class Session(ContextRecoder):
    api: OpenAIAPI
    payload: Payload
    current_input: list[ContentSegment]
    unit_prompts: list[str]

    def __init__(
        self,
        memory_size: int,
        silence_size: int,
        router_size: int,
        unimportant_size: int,
        decouple_size: int,
        sentence_model: SentenceTransformer,
    ) -> None:
        # 标准记录
        super().__init__(memory_size)
        self.silence_recorder: deque[tuple[str, float]] = deque(maxlen=silence_size)
        self.router_recorder: deque[Record] = deque(maxlen=router_size)
        # 临时记录
        self.snap: ContextRecoder = ContextRecoder(memory_size)
        # 状态
        self.extra: dict = {}
        self.usage_counter = {}
        # 不重要信息
        self.unimportant = False
        self.unimportant_recorder: deque[Record] = deque(maxlen=unimportant_size)
        # 主题分离
        self.decoupler = TopicDecoupler(sentence_model)
        self.decouple_size = decouple_size

    def over(self, request: UserMessage, reply: AssistantMessage, timestamp: float):
        """处理完成"""
        record = (request, reply, timestamp)
        self.router_recorder.append(record)
        if self.unimportant:
            self.unimportant_recorder.append(record)
            self.unimportant = False
        else:
            if self.unimportant_recorder:
                self.recorder.extend(self.unimportant_recorder)
                self.unimportant_recorder.clear()
            self.recorder.append(record)
        self.silence_recorder.clear()

    def clear(self):
        """清理记录"""
        self.router_recorder.clear()
        self.unimportant_recorder.clear()
        self.silence_recorder.clear()

    def memory_filter(self, timeout: int | float):
        """过滤记忆"""
        while self.recorder and (self.recorder[0][2] <= timeout):
            self.recorder.popleft()

    def silence_filter(self, timeout: int | float):
        """过滤静默记录群聊上下文"""
        while self.silence_recorder and (self.silence_recorder[0][1] <= timeout):
            self.silence_recorder.popleft()

    def sync_snap(self):
        """同步上下文到辅助AI"""
        self.snap.clear()
        self.snap.recorder.extend(self.recorder)

    def step(self, message: str):
        if sum(char_count(msg["content"]) for msg in self) < self.decouple_size:
            return False
        return self.decoupler.step(message)

    def activate(self):
        self.is_first_wait = True
        for rec in self.unimportant_recorder:
            self.payload["messages"].extend(rec[:2])
        self.cursor = len(self.payload["messages"])
        self.payload["messages"].append({"role": "user", "content": self.current_input})
        self.result = None

    def inactivate(self):
        self.snap.clear()
        # 按注入顺序删除
        del self.current_input
        del self.unit_prompts
        del self.api
        del self.payload
        del self.is_first_wait
        del self.cursor
        del self.result

    @property
    def system_message(self) -> SystemMessage:
        message = self.payload["messages"][0]
        if message["role"] == "system":
            return message
        raise ValueError(f"The first message must have the role 'system', but found '{message}' instead.")

    @property
    def router_context(self):
        for a, b, _ in self.router_recorder:
            yield a
            yield b

    @property
    def unimportant_context(self):
        if not self.unimportant_recorder and self.recorder:
            yield from self.recorder[-1][:2]

    def update_context(self, messags: Iterable[Message]):
        messages = [self.system_message, *messags]
        cursor = len(messages)
        messages.extend(self.payload["messages"][self.cursor :])
        self.cursor = cursor
        self.payload["messages"] = messages

    def complete(self, result: str):
        self.result = result
