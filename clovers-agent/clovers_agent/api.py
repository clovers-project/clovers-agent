import random
import contextlib
import httpx
import asyncio
from clovers.logger import logger
from collections.abc import Iterable
from .utils import data_url, deep_add
from typing import override
from .typing import Message, AssistantMessage, Payload
from .typing.message import MultimodalContent
from .config import OpenAIConfig, HybridOpenAIConfig
from .constants import VISION_TAG, VISION_PROMPT

RETRY_DELAYS = (0.5, 1.0, None)


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
        for delay in RETRY_DELAYS:
            retry_after = None
            try:
                resp = await self.async_client.post(self.url, headers=self.headers, json=payload)
                if x := resp.headers.get("retry-after"):
                    with contextlib.suppress(ValueError):
                        retry_after = min(float(x), 3)
                resp.raise_for_status()
                data = resp.json()
                deep_add(usage_counter, {payload["model"]: data.get("usage")})
                message = data["choices"][0]["message"]
                if "content" not in message and "tool_calls" not in message:
                    raise ValueError(f"API returned an invalid response: {resp.text}")
                return message
            except httpx.RequestError as e:
                logger.exception(e)
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code < 500 and status_code != 429:
                    logger.error(f"Payload:\n{'\n'.join(pretty_payload(payload))}\nResponse:\n{e.response.text}")
                    raise RuntimeError(f"API client error (HTTP {status_code})") from e
                logger.exception(e)
            except Exception as e:
                raise RuntimeError(f"Response validation failed: {e}") from e
            if delay:
                await asyncio.sleep(retry_after or delay * random.uniform(0.75, 1.25))
        raise RuntimeError("API retries exhausted")

    async def download_url(
        self,
        url: str,
        allowed_subtype: tuple[str, ...] = ("png", "jpeg", "webp", "gif"),
        max_size: int = 1024 * 1024 * 10,
    ):
        if not url.startswith("http"):
            return url
        try:
            resp = await self.async_client.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            logger.exception(e)
            return None
        data = resp.content
        if len(data) > max_size:
            return None
        return data_url(resp.content, allowed_subtype)


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
            payload = self.vision.build_payload(({"role": "user", "content": content},), VISION_PROMPT)
            resp = await self.vision.call_api(payload, usage_counter)
            return resp["content"].strip()
        except Exception as e:
            logger.exception(e)


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
