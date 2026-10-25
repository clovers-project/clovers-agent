import httpx
from pathlib import Path
from clovers.logger import logger
from .utils import data_url, deep_add, retry_httpx, httpx_request
from .config import OpenAIConfig, HybridOpenAIConfig
from .constants import VISION_TAG, VISION_PROMPT

from typing import override
from .typing import AssistantMessage, Payload
from .typing.message import MultimodalContent

retry_request = retry_httpx(3)(httpx_request)
ALLOWED_IMAGE_SUBTYPES = ("png", "jpeg", "webp", "gif")


class OpenAIAPI:
    def __init__(self, async_client: httpx.AsyncClient, config: OpenAIConfig) -> None:
        self.async_client = async_client
        self.url = f"{config.url.rstrip("/")}/chat/completions"
        self.model = config.model
        self.headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        self.extra_body = config.extra_body

    def build_payload(self, system_prompt: str | None = None) -> Payload:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        return {"model": self.model, "messages": messages, **self.extra_body}  # type: ignore 这里允许额外请求体

    async def call_api(self, payload: Payload, usage_counter: dict) -> AssistantMessage:
        resp = await retry_request(self.async_client.post, self.url, headers=self.headers, json=payload)
        data = resp.json()
        deep_add(usage_counter, {payload["model"]: data.get("usage")})
        message = data["choices"][0]["message"]
        if "content" not in message and "tool_calls" not in message:
            raise ValueError(f"API returned an invalid response: {resp.text}")
        return message

    async def download_url(
        self,
        url: str,
        allowed_subtype: tuple[str, ...] = ALLOWED_IMAGE_SUBTYPES,
        max_size: int = 1024 * 1024 * 10,
    ):
        try:
            if url.startswith("data:"):
                _, subtype, size = ...
                if subtype in allowed_subtype and size <= max_size:
                    return url
            else:
                if url.startswith(("https://", "http://")):
                    resp = await retry_request(self.async_client.get, url, timeout=60)
                    data = resp.content
                elif url.startswith("file://"):
                    file = Path(url[7:])
                    if not file.exists() or not file.is_file():
                        return
                    data = file.read_bytes()
                if len(data) > max_size:
                    return
                return data_url(data, allowed_subtype)
        except Exception:
            logger.exception("download_url failed")


class HybridOpenAIAPI(OpenAIAPI):
    def __init__(self, async_client: httpx.AsyncClient, config: HybridOpenAIConfig) -> None:
        super().__init__(async_client, config)
        if not config.vision:
            raise ValueError("Vision API config is required")
        self.vision = OpenAIAPI(async_client, config.vision)

    @override
    async def call_api(self, payload: Payload, usage_counter: dict) -> AssistantMessage:
        message = next((x for x in reversed(payload["messages"]) if x["role"] == "user"))
        # 最后一条 UserMessage 是 current_input
        if isinstance(content := message["content"], str):
            return await super().call_api(payload, usage_counter)
        texts = []
        images = []
        for seg in content:
            match seg["type"]:
                case "text":
                    texts.append(seg)
                case "image_url":
                    images.append(seg)
        if images and (desc := await self.call_vision(images, usage_counter)):
            content.clear()
            content.extend([{"type": "text", "text": VISION_TAG.format(desc)}])
            content.extend(texts)
        result = await super().call_api(payload, usage_counter)
        return result

    async def call_vision(self, content: MultimodalContent, usage_counter: dict):
        try:
            payload = self.vision.build_payload(VISION_PROMPT)
            payload["messages"].append({"role": "user", "content": content})
            resp = await self.vision.call_api(payload, usage_counter)
            return resp["content"].strip()
        except Exception:
            logger.exception("call_vision failed")


def pretty_payload(payload: Payload):
    if "tools" in payload:
        yield f"[TOOLS]:{[x["function"]["name"] for x in  payload["tools"]]}"
    for i, message in enumerate(payload["messages"]):
        prefix = f"[MESSAGE:{i}][{message['role'].upper()}]\n"
        match message["role"]:
            case "system":
                yield prefix + message["content"]
            case "user":
                if isinstance(message["content"], str):
                    yield prefix + message["content"]
                else:
                    yield prefix + "".join(x["text"] if x["type"] == "text" else f"[IMAGE]" for x in message["content"])
            case "assistant":
                content = prefix + message.get("content") or ""
                if "tool_calls" in message:
                    content += "".join(f"\n[CALL][{x["function"]["name"]}]{x["function"]["arguments"]}" for x in message["tool_calls"])
                yield content
            case "tool":
                yield prefix + message.get("content") or ""
