from __future__ import annotations
import asyncio
from itertools import count
from collections import deque
from .config import SESSION as SESSION_CONFIG
from .constants import SYSTEM_TAG, GET_IMAGE_BY_ID_INFO
from collections.abc import Iterable
from typing import TYPE_CHECKING, override
from .typing import Payload, Message, AssistantMessage, SystemMessage, FunctionToolInfo
from .typing.message import MultimodalContent, TextUserMessage

if TYPE_CHECKING:
    from .api import OpenAIAPI
    from .embedding import SentenceTransformer


type Record = tuple[TextUserMessage, AssistantMessage, float]


class SessionComplete(Exception):
    def __init__(self, data: str | None) -> None:
        super().__init__()
        self.data = data


class TurnSession:
    api: OpenAIAPI
    payload: Payload

    def __init__(self, session_id: str, usage_counter: dict) -> None:
        self.session_id = session_id
        self.usage_counter = usage_counter
        self.unit_prompts: list[str] = []
        self.current_input: MultimodalContent = []
        self.scripts_enum: list[str] = []
        self.references_enum: list[str] = []

    def build_payload(self, api: OpenAIAPI, system_prompt: str | None = None):
        if self.cursor is not None:
            raise RuntimeError("Cannot build payload when session is active")
        self.api = api
        self.payload = self.api.build_payload(system_prompt)

    def input_text(self, text: str):
        self.current_input.append({"type": "text", "text": text})

    def input_image(self, image: str):
        self.current_input.append({"type": "image_url", "image_url": {"url": image}})

    def input_images(self, images: Iterable[str]):
        self.current_input.extend({"type": "image_url", "image_url": {"url": image}} for image in images)

    def activate(self):
        self.cursor = len(self.payload["messages"])
        unit_prompt = SYSTEM_TAG.format("\n".join(x for x in self.unit_prompts if x)) + "\n"
        self.current_input.insert(0, {"type": "text", "text": unit_prompt})
        self.payload["messages"].append({"role": "user", "content": self.current_input})

    def complete(self, message: str | None = None):
        self.inactivate()
        raise SessionComplete(message)

    def inactivate(self):
        # 按注入顺序删除
        self.unit_prompts.clear()
        self.current_input.clear()
        self.api = None  # type: ignore
        self.payload = None  # type: ignore
        self.cursor = None
        del self.api
        del self.payload
        self.scripts_enum.clear()
        self.references_enum.clear()

    def extend_tools(self, extends: Iterable[FunctionToolInfo]):
        return extend_tools(self.payload, extends)

    def call_api(self):
        return self.api.call_api(self.payload, self.usage_counter)


def iter_recoder(recorder: Iterable[Record]):
    for a, b, _ in recorder:
        yield a
        yield b


class Session(TurnSession):
    def __init__(
        self,
        session_id: str,
        sentence_model: SentenceTransformer,
        router_size: int = SESSION_CONFIG.router_size,
        memory_size: int = SESSION_CONFIG.memory_size,
        memory_timeout: int = SESSION_CONFIG.memory_timeout,
        silence_size: int = SESSION_CONFIG.silence_size,
        silence_timeout: int = SESSION_CONFIG.silence_timeout,
        unimportant_size: int = SESSION_CONFIG.unimportant_size,
        decoupler_alpha: float = SESSION_CONFIG.decoupler_alpha,
    ) -> None:

        super().__init__(session_id, {})
        # 标准记录
        self.recorder: list[Record] = []
        self.router_recorder: deque[Record] = deque(maxlen=router_size)
        self.memory_size = memory_size
        self.memory_timeout = memory_timeout
        self.image_recorder: deque[tuple[int, str, float]] = deque()
        self.image_id = count()
        self.silence_recorder: deque[tuple[str, float]] = deque(maxlen=silence_size)
        self.silence_timeout = silence_timeout
        # 不重要信息
        self.unimportant = False
        self.unimportant_recorder: deque[Record] = deque(maxlen=unimportant_size)
        # 主题分离
        from .embedding import TopicDecoupler

        self.decoupler = TopicDecoupler(sentence_model, decoupler_alpha)
        self.decouple_length = SESSION_CONFIG.decouple_length
        self.decouple_count = 0
        # 状态
        self.execute_lock = asyncio.Lock()
        self.wait_lock = asyncio.Lock()
        self.last_active_time = 0.0
        self.extra = {}

    def snapshot(self):
        if self.cursor is None:
            return
        s = TurnSession(self.session_id, self.usage_counter)
        s.build_payload(self.api)
        s.payload["messages"].extend(self.payload["messages"][: self.cursor])
        return s

    async def snap_call(self, unit_prompt: str, content: str):
        if (s := self.snapshot()) is None:
            return
        s.unit_prompts.append(unit_prompt)
        s.input_text(content)
        s.activate()
        resp = await s.call_api()
        return resp["content"].strip()

    @property
    def messages(self):
        return *iter_recoder(self.unimportant_recorder), *iter_recoder(self.recorder)

    @override
    def activate(self):
        self.payload["messages"].extend(self.messages)
        if self.image_recorder:
            self.extend_tools((GET_IMAGE_BY_ID_INFO,))
        super().activate()

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
        self.router_recorder.append(record)
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

    def update_context(self, messages: Iterable[Message]):
        _m = [self.system_message, *messages]
        cursor = len(_m)
        _m.extend(self.payload["messages"][self.cursor :])
        self.cursor = cursor
        self.payload["messages"] = _m


def extend_tools(payload, extends: Iterable[FunctionToolInfo]) -> list[FunctionToolInfo]:
    if "tools" not in payload:
        payload["tools"] = []
        payload["tools"].extend(extends)
    else:
        used = {tool["function"]["name"] for tool in payload["tools"]}
        payload["tools"].extend(x for x in extends if x["function"]["name"] not in used)
    return payload["tools"]
