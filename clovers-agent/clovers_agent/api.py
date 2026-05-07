import httpx
from clovers.logger import logger
from collections.abc import Iterable
from .utils import data_url, deep_add
from typing import override
from .typing import Message, UserMessage, AssistantMessage, Payload
from .typing.message import MultimodalContent
from .config import OpenAIConfig, HybridOpenAIConfig
from .constants import SYSTEM_TAG, VISION_PROMPT


class OpenAIAPI:
    def __init__(self, async_client: httpx.AsyncClient, config: OpenAIConfig) -> None:
        self.async_client = async_client
        self.url = f"{config.url.rstrip("/")}/chat/completions"
        self.model = config.model
        self.headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        self.extra_body = config.extra_body

    @staticmethod
    def build_message(text: str, image_list: list[str] | None) -> UserMessage:
        if not image_list:
            return {"role": "user", "content": text}
        else:
            content = []
            if text:
                content.append({"type": "text", "text": text})
            if image_list:
                content.extend({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_list)
            return {"role": "user", "content": content}

    def build_payload(self, context: Iterable[Message] | None = None, system_prompt: str | None = None) -> Payload:
        payload: Payload = {"model": self.model, "messages": [], **self.extra_body}  # type: ignore 这里允许额外请求体
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        if context:
            payload["messages"].extend(context)
        return payload

    async def call_api(self, payload: Payload, usage_counter: dict) -> AssistantMessage:
        resp = await self.async_client.post(self.url, headers=self.headers, json=payload)
        from .session import extract_plain_text

        print("\n".join(f"[{i}{x["role"]}]:\n{extract_plain_text(x["content"])}" for i, x in enumerate(payload["messages"])))
        if resp.status_code != 200:
            logger.error(resp.text)
            resp.raise_for_status()
        try:
            data = resp.json()
            deep_add(usage_counter, {payload["model"]: data.get("usage")})
            message = data["choices"][0]["message"]
        except Exception as e:
            raise RuntimeError(f"Failed to parse API response {resp.text}") from e
        if "content" not in message and "tool_calls" not in message:
            raise ValueError(f"API returned an invalid response: {resp.text}")
        return message

    async def download_url(self, url: str):
        if not url.startswith("http"):
            return url
        try:
            resp = await self.async_client.get(url, timeout=60)
            resp.raise_for_status()
            return data_url(resp.content)
        except Exception as e:
            logger.exception(e)
            return None


class HybridOpenAIAPI(OpenAIAPI):
    def __init__(self, async_client: httpx.AsyncClient, config: HybridOpenAIConfig) -> None:
        super().__init__(async_client, config)
        if not config.vision:
            raise ValueError("Vision API config is required")
        self.vision = OpenAIAPI(async_client, config.vision)

    @override
    async def call_api(self, payload: Payload, usage_counter: dict) -> AssistantMessage:
        index = next((i for i, x in enumerate(reversed(payload["messages"])) if x["role"] == "user"), -1)
        index = -1 - index
        if index is not None and isinstance(content := payload["messages"][index]["content"], list):
            texts = []
            has_image = False
            for seg in content:
                match seg["type"]:
                    case "text":
                        texts.append(seg["text"])
                    case "image_url":
                        has_image = True
            contents = []
            if has_image and (desc := await self.call_vision(content, usage_counter)):
                print(desc)
                contents.append(SYSTEM_TAG.format(desc))
                contents.append("\n")
            contents.append("".join(texts))
            payload["messages"][index] = {"role": "user", "content": "".join(contents)}
        return await super().call_api(payload, usage_counter)

    async def call_vision(self, content: MultimodalContent, usage_counter: dict):
        try:
            payload = self.vision.build_payload(({"role": "user", "content": content},), VISION_PROMPT)
            resp = await self.vision.call_api(payload, usage_counter)
            return resp["content"].strip()
        except Exception as e:
            logger.exception(e)
