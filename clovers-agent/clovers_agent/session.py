from __future__ import annotations
import time
import asyncio
import contextlib
from pathlib import Path
from itertools import count
from collections import deque
from .utils import load_jsonl, save_jsonl
from .config import SESSION as SESSION_CONFIG
from .constants import (
    SYSTEM_TAG,
    SUMMARY_TAG,
    READ_IMAGE_BY_ID_INFO,
    EXECUTE_SCRIPT,
    EXECUTE_SCRIPT_DESC,
    READ_REFERENCE,
    READ_REFERENCE_DESC,
)

from typing import TYPE_CHECKING, cast, override, overload, Never, Any
from collections.abc import Iterable
from .typing import Payload, AssistantMessage, SystemMessage, FunctionToolInfo
from .typing.message import MultimodalContent, TextSegment, ImageSegment, TextUserMessage

if TYPE_CHECKING:
    from .api import OpenAIAPI
    from .embedding import SentenceTransformer

type Record = tuple[TextUserMessage, AssistantMessage, float]


class SessionComplete(BaseException):
    def __init__(self, data: str | None) -> None:
        super().__init__()
        self.data = data


class Turn:
    api: OpenAIAPI
    payload: Payload

    def __init__(self, session_id: str, usage_counter: dict, extra: dict) -> None:
        self.session_id = session_id
        self.usage_counter = usage_counter
        self.extra = extra
        self.unit_prompts: list[str] = []
        self.current_input: MultimodalContent = []
        self.scripts_enum: list[str] = []
        self.references_enum: list[str] = []
        self.cursor: int | None = None

    def create_payload(self, api: OpenAIAPI, system_prompt: str | None = None):
        if self.cursor is not None:
            raise RuntimeError("Cannot build payload when session is active")
        self.api = api
        self.payload = self.api.build_payload(system_prompt)

    def input_text(self, text: str):
        x = cast(TextSegment, {"type": "text", "text": text})
        self.current_input.append(x)
        return x

    @overload
    def input_image(self, image: str) -> ImageSegment: ...
    @overload
    def input_image(self, image: Iterable[str]) -> tuple[ImageSegment]: ...

    def input_image(self, image: str | Iterable[str]):
        if isinstance(image, str):
            x = cast(ImageSegment, {"type": "image_url", "image_url": {"url": image}})
            self.current_input.append(x)
        else:
            x = cast(tuple[ImageSegment], tuple({"type": "image_url", "image_url": {"url": img}} for img in image))
            self.current_input.extend(x)
        return x

    def activate(self):
        self.cursor = len(self.payload["messages"])
        if self.unit_prompts:
            unit_prompt = SYSTEM_TAG.format("\n".join(x for x in self.unit_prompts if x)) + "\n"
            self.current_input.insert(0, {"type": "text", "text": unit_prompt})
        self.payload["messages"].append({"role": "user", "content": self.current_input})

    def complete_session(self, message: str | None = None) -> Never:
        self.deactivate()
        raise SessionComplete(message)

    def deactivate(self):
        # 按注入顺序删除
        if self.unit_prompts:
            self.unit_prompts.clear()
            del self.current_input[0]
        self.current_input = []
        self.api = None  # type: ignore
        self.payload = None  # type: ignore
        del self.api
        del self.payload
        self.cursor = None
        self.scripts_enum.clear()
        self.references_enum.clear()

    def create_turn(self):
        """创建一个子会话"""
        return Turn(self.session_id, self.usage_counter, self.extra)

    def call_api(self):
        return self.api.call_api(self.payload, self.usage_counter)

    @property
    def system_message(self) -> SystemMessage | None:
        message = self.payload["messages"][0]
        if message["role"] == "system":
            return message

    def extend_tools(self, extends: Iterable[FunctionToolInfo]):
        payload = self.payload
        if "tools" not in payload:
            payload["tools"] = []
            payload["tools"].extend(extends)
        else:
            used = {tool["function"]["name"] for tool in payload["tools"]}
            payload["tools"].extend(x for x in extends if x["function"]["name"] not in used)

    def upsert_tool(self, info: FunctionToolInfo):
        payload = self.payload
        if "tools" not in payload:
            payload["tools"] = [info]
            return
        name = info["function"]["name"]
        tool = next((x for x in payload["tools"] if x["function"]["name"] == name), None)
        if tool:
            tool.update(info)
        else:
            payload["tools"].append(info)

    def extend_scripts(self, scripts: Iterable[str]):
        existing = set(self.scripts_enum)
        self.scripts_enum.extend(path for path in scripts if path not in existing)
        self.upsert_tool(
            {
                "type": "function",
                "function": {
                    "name": EXECUTE_SCRIPT,
                    "description": EXECUTE_SCRIPT_DESC,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interpreter": {"type": "string"},
                            "path": {"type": "string", "enum": self.scripts_enum},
                            "args": {"type": "array", "items": {"type": "string"}},
                            "timeout": {"type": "number"},
                        },
                        "required": ["path"],
                    },
                },
            }
        )

    def extend_references(self, references: Iterable[str]):
        existing = set(self.references_enum)
        self.references_enum.extend(path for path in references if path not in existing)
        self.upsert_tool(
            {
                "type": "function",
                "function": {
                    "name": READ_REFERENCE,
                    "description": READ_REFERENCE_DESC,
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "enum": self.references_enum}},
                        "required": ["path"],
                    },
                },
            }
        )


def iter_recoder(recorder: Iterable[Record]):
    for a, b, _ in recorder:
        yield a.copy()
        yield b.copy()


class Session(Turn):
    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        sentence_model: SentenceTransformer,
        router_size: int = SESSION_CONFIG.router_size,
        memory_size: int = SESSION_CONFIG.memory_size,
        memory_timeout: int = SESSION_CONFIG.memory_timeout,
        silence_size: int = SESSION_CONFIG.silence_size,
        silence_timeout: int = SESSION_CONFIG.silence_timeout,
        decoupler_alpha: float = SESSION_CONFIG.decoupler_alpha,
    ) -> None:

        super().__init__(session_id, {}, {})
        self.recorder_file = session_dir / "recorder.jsonl"
        # 主题分离
        from .embedding import TopicDecoupler

        self.decoupler = TopicDecoupler(sentence_model, decoupler_alpha)
        self.decouple_length = SESSION_CONFIG.decouple_length

        # 标准记录
        if recorder := load_jsonl(self.recorder_file):
            self.recorder = cast(list[Record], recorder)
            self.decouple_count = sum(len(m["content"]) for m in iter_recoder(self.recorder))
        else:
            self.recorder = cast(list[Record], [])
            self.decouple_count = 0
        self.router_recorder: deque[Record] = deque(maxlen=router_size)
        self.memory_size = memory_size
        self.memory_timeout = memory_timeout
        self.image_recorder: deque[tuple[int, str, float]] = deque()
        self.image_id = count()
        self.message_history: deque[tuple[str, float]] = deque(maxlen=silence_size)
        self.silence_timeout = silence_timeout
        # 状态
        self.last_active_time = 0.0
        self.execute_lock = asyncio.Lock()
        self.aside_lock = asyncio.Lock()
        self.idle_event = asyncio.Event()
        self.idle_event.set()
        self.pending_prompt: str | None
        self.wakeup_context: WakeupContext | None = None

    @override
    def activate(self):
        self.idle_event.clear()
        self.payload["messages"].extend(self.messages)
        if self.image_recorder:
            self.extend_tools((READ_IMAGE_BY_ID_INFO,))
        super().activate()

    @override
    def deactivate(self):
        self.idle_event.set()
        super().deactivate()
        self.outbox = None  # type: ignore

    @property
    def messages(self):
        return tuple(iter_recoder(self.recorder))

    def record_history(self, content: str, timestamp: float):
        timeout = timestamp - self.memory_timeout
        while self.image_recorder and (self.image_recorder[0][2] <= timeout):
            self.image_recorder.popleft()
        timeout = timestamp - self.silence_timeout
        while self.message_history and (self.message_history[0][1] <= timeout):
            self.message_history.popleft()
        self.message_history.append((content, timestamp))

    def should_summarize(self, message: str, timestamp: float):
        if not self.recorder:
            return False
        if self.wakeup_context is not None:
            return False
        if len(self.recorder) > self.memory_size:
            return True
        timeout = timestamp - self.memory_timeout
        if self.recorder[0][2] < timeout:
            return True
        if self.decouple_count < self.decouple_length:
            return False
        return self.decoupler.step(message)

    def compact_history(self, content: str):
        self.recorder.clear()
        self.image_recorder.clear()
        self.router_recorder.clear()
        self.decouple_count = 0
        self.recorder_file.unlink(missing_ok=True)
        summary = SUMMARY_TAG.format(content)
        if self.message_history:
            timestamp = self.message_history[0][1]
        else:
            timestamp = time.time()
        self.message_history.appendleft((summary, timestamp))

    @property
    def turn_input(self):
        return "\n".join(x for x, _ in self.message_history)

    def record_turn(self, turn_input: str | MultimodalContent, turn_output: str):
        timestamp = self.last_active_time
        if isinstance(turn_input, list):
            contents: list[str] = []
            for seg in turn_input:
                match seg["type"]:
                    case "text":
                        contents.append(seg["text"])
                    case "image_url":
                        url = seg["image_url"]["url"]
                        image_id = next(self.image_id)
                        contents.append(f" [image:{image_id}] ")
                        self.image_recorder.append((image_id, url, timestamp))
            turn_input = "".join(contents)
        record: Record = ({"role": "user", "content": turn_input}, {"role": "assistant", "content": turn_output}, timestamp)
        self.router_recorder.append(record)
        self.recorder.append(record)
        self.decouple_count += len(record[0]["content"]) + len(record[1]["content"])
        self.message_history.clear()
        save_jsonl(self.recorder_file, record)

    def image_url(self, image_id: int) -> str | None:
        return next((url for i, url, _ in self.image_recorder if i == image_id), None)

    def cleanup(self):
        with contextlib.suppress(Exception):
            if self.wakeup_context:
                for task in self.wakeup_context.tasks:
                    task.cancel()
            self.recorder_file.unlink(missing_ok=True)


class WakeupContext:
    def __init__(self, turn: Turn) -> None:
        self.turn = turn
        self.tasks: list[asyncio.Task[str | None]] = []
