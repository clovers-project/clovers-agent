import asyncio
from itertools import count
from collections import deque
from .api import OpenAIAPI
from .embedding import SentenceTransformer, TopicDecoupler
from collections.abc import Iterable
from .typing import Payload, Message, AssistantMessage, SystemMessage
from .typing.message import MultimodalContent, TextUserMessage
from .config import SESSION as SESSION_CONFIG
from .constants import SYSTEM_TAG, GET_IMAGE_BY_ID_INFO


def extract_plain_text(content: str | MultimodalContent) -> str:
    return content if isinstance(content, str) else "\n".join(text["text"] for text in content if text["type"] == "text")


def char_count(content: str | MultimodalContent):
    return len(content) if isinstance(content, str) else sum(len(text["text"]) for text in content if text["type"] == "text")


type Record = tuple[TextUserMessage, AssistantMessage, float]


class Session:
    api: OpenAIAPI
    payload: Payload
    current_input: MultimodalContent

    def __init__(self, sentence_model: SentenceTransformer) -> None:
        # 标准记录
        self.recorder: deque[Record] = deque(maxlen=SESSION_CONFIG.memory_size)
        self.silence_recorder: deque[tuple[str, float]] = deque(maxlen=SESSION_CONFIG.silence_size)
        self.image_recorder: deque[tuple[int, str, float]] = deque()
        self.image_id = count()
        self.router_recorder: deque[tuple[str, str]] = deque(maxlen=SESSION_CONFIG.router_size)
        self.memory_timeout = SESSION_CONFIG.memory_timeout
        self.silence_timeout = SESSION_CONFIG.silence_timeout
        # 状态
        self.execute_lock = asyncio.Lock()
        self.wait_lock = asyncio.Lock()
        self.last_active_time = 0.0
        self.extra = {}
        self.usage_counter = {}
        self.unit_prompts: list[str] = []
        # 不重要信息
        self.unimportant = False
        self.unimportant_recorder: deque[Record] = deque(maxlen=SESSION_CONFIG.unimportant_size)
        # 主题分离
        self.decoupler = TopicDecoupler(sentence_model)
        self.decouple_length = SESSION_CONFIG.decouple_length

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
        self.router_recorder.append((content, reply["content"]))
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
        self.recorder.clear()
        self.image_recorder.clear()
        self.router_recorder.clear()
        self.unimportant_recorder.clear()
        self.silence_recorder.clear()

    def image_url(self, image_id: int) -> str | None:
        return next((url for i, url, _ in self.image_recorder if i == image_id), None)

    def refresh(self, timestamp: float):
        """刷新记忆"""
        timeout = timestamp - self.memory_timeout
        while self.recorder and (self.recorder[0][2] <= timeout):
            self.recorder.popleft()
        while self.image_recorder and (self.image_recorder[0][2] <= timeout):
            self.image_recorder.popleft()
        timeout = timestamp - self.silence_timeout
        while self.silence_recorder and (self.silence_recorder[0][1] <= timeout):
            self.silence_recorder.popleft()

    def step(self, message: str):
        if sum(char_count(msg["content"]) for msg in self) < self.decouple_length:
            return False
        return self.decoupler.step(message)

    def activate(self):
        for rec in self.unimportant_recorder:
            self.payload["messages"].extend(rec[:2])
        self.cursor = len(self.payload["messages"])
        unit_prompt = SYSTEM_TAG.format("\n".join(x for x in self.unit_prompts if x)) + "\n"
        self.current_input.insert(0, {"type": "text", "text": unit_prompt})
        self.payload["messages"].append({"role": "user", "content": self.current_input})
        self.result = None
        if not self.image_recorder:
            return
        if "tools" not in self.payload:
            self.payload["tools"] = []
        self.payload["tools"].append(GET_IMAGE_BY_ID_INFO)

    def inactivate(self):
        # 按注入顺序删除
        self.unit_prompts.clear()
        del self.current_input
        del self.api
        del self.payload
        del self.cursor
        del self.result

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

    def complete(self, result: str):
        self.result = result
