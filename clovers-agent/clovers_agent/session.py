from __future__ import annotations
import asyncio
from itertools import count
from collections import deque
from .config import SESSION as SESSION_CONFIG
from .constants import SYSTEM_TAG, GET_IMAGE_BY_ID_INFO
from collections.abc import Iterable
from typing import TYPE_CHECKING
from .typing import Payload, Message, AssistantMessage, SystemMessage
from .typing.message import MultimodalContent, TextUserMessage

if TYPE_CHECKING:
    from .api import OpenAIAPI
    from .embedding import SentenceTransformer


type Record = tuple[TextUserMessage, AssistantMessage, float]


class SessionComplete(Exception):
    def __init__(self, data: str | None) -> None:
        super().__init__()
        self.data = data


class Session:
    api: OpenAIAPI
    payload: Payload
    current_input: MultimodalContent

    def __init__(self, session_id: str, sentence_model: SentenceTransformer) -> None:
        # 会话 id
        self.session_id = session_id
        # 标准记录
        self.recorder: list[Record] = []
        self.silence_recorder: deque[tuple[str, float]] = deque(maxlen=SESSION_CONFIG.silence_size)
        self.image_recorder: deque[tuple[int, str, float]] = deque()
        self.image_id = count()
        self.router_recorder: deque[tuple[str, str]] = deque(maxlen=SESSION_CONFIG.router_size)
        self.memory_timeout = SESSION_CONFIG.memory_timeout
        self.memory_size = SESSION_CONFIG.memory_size
        self.silence_timeout = SESSION_CONFIG.silence_timeout
        # 状态
        self.execute_lock = asyncio.Lock()
        self.wait_lock = asyncio.Lock()
        self.last_active_time = 0.0
        self.extra = {}
        self.usage_counter = {}
        self.unit_prompts: list[str] = []
        self.scripts_enum: list[str] = []
        self.references_enum: list[str] = []
        # 不重要信息
        self.unimportant = False
        self.unimportant_recorder: deque[Record] = deque(maxlen=SESSION_CONFIG.unimportant_size)
        # 主题分离
        from .embedding import TopicDecoupler

        self.decoupler = TopicDecoupler(sentence_model, SESSION_CONFIG.decoupler_alpha)
        self.decouple_length = SESSION_CONFIG.decouple_length
        self.decouple_count = 0

    def __iter__(self):
        for a, b, _ in self.recorder:
            yield a
            yield b

    def over(self, content: str | MultimodalContent, reply: AssistantMessage, timestamp: float):
        """处理完成"""
        if isinstance(content, list):
            contents: list[str] = []
            for seg in content:
                match seg["type"]:
                    case "text":
                        contents.append(seg["text"])
                    case "image_url":
                        url = seg["image_url"]["url"]
                        image_id = next(self.image_id)
                        contents.append(f" [image:{image_id}] ")
                        self.image_recorder.append((image_id, url, timestamp))
            content = "".join(contents)
        record: Record = ({"role": "user", "content": content}, reply, timestamp)
        pending_records = list[Record]()
        self.router_recorder.append((content, reply["content"]))
        if self.unimportant:
            self.unimportant_recorder.append(record)
            self.unimportant = False
        else:
            if self.unimportant_recorder:
                pending_records.extend(self.unimportant_recorder)
                self.unimportant_recorder.clear()
            pending_records.append(record)
            self.recorder.extend(pending_records)
            self.decouple_count += sum(len(u["content"]) + len(a["content"]) for u, a, _ in pending_records)
        self.silence_recorder.clear()
        return pending_records

    def clear(self):
        """清理记录"""
        self.recorder.clear()
        self.image_recorder.clear()
        self.router_recorder.clear()
        self.unimportant_recorder.clear()
        self.silence_recorder.clear()
        self.decouple_count = 0

    def image_url(self, image_id: int) -> str | None:
        return next((url for i, url, _ in self.image_recorder if i == image_id), None)

    def refresh(self, timestamp: float):
        """刷新记忆"""
        # timeout = timestamp - self.memory_timeout
        # while self.recorder and (self.recorder[0][2] <= timeout):
        #     self.recorder.popleft()
        timeout = timestamp - self.memory_timeout
        while self.image_recorder and (self.image_recorder[0][2] <= timeout):
            self.image_recorder.popleft()
        timeout = timestamp - self.silence_timeout
        while self.silence_recorder and (self.silence_recorder[0][1] <= timeout):
            self.silence_recorder.popleft()

    def step(self, message: str, timestamp: float):
        if not self.recorder:
            return False
        if len(self.recorder) > self.memory_size:
            return True
        timeout = timestamp - self.memory_timeout
        if self.recorder[0][2] < timeout:
            return True
        if self.decouple_count < self.decouple_length:
            return False
        return self.decoupler.step(message)

    def activate(self):
        payload = self.payload
        for rec in self.unimportant_recorder:
            payload["messages"].extend(rec[:2])
        self.cursor = len(payload["messages"])
        unit_prompt = SYSTEM_TAG.format("\n".join(x for x in self.unit_prompts if x)) + "\n"
        self.current_input.insert(0, {"type": "text", "text": unit_prompt})
        payload["messages"].append({"role": "user", "content": self.current_input})
        if not self.image_recorder:
            return
        if "tools" not in payload:
            payload["tools"] = []
        payload["tools"].append(GET_IMAGE_BY_ID_INFO)

    def complete(self, message: str | None = None):
        self.inactivate()
        raise SessionComplete(message)

    def inactivate(self):
        # 按注入顺序删除
        self.unit_prompts.clear()
        self.current_input = None  # type: ignore
        self.api = None  # type: ignore
        self.payload = None  # type: ignore
        self.cursor = None  # type: ignore
        del self.current_input
        del self.api
        del self.payload
        del self.cursor
        self.scripts_enum.clear()
        self.references_enum.clear()

    @property
    def system_message(self) -> SystemMessage:
        message = self.payload["messages"][0]
        if message["role"] == "system":
            return message
        raise ValueError(f"The first message must have the role 'system', but found '{message}' instead.")

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
