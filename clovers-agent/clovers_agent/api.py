import json
import httpx
from clovers.logger import logger
from collections.abc import Iterable
from .utils import data_url
from .typing import Message, UserMessage, AssistantMessage, Payload
from .config import OpenAIConfig


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

    async def call_api(self, payload: Payload) -> AssistantMessage:
        resp = await self.async_client.post(self.url, headers=self.headers, json=payload)
        if resp.status_code != 200:
            logger.error(json.dumps(payload, indent=4, ensure_ascii=False))
            logger.error(resp.text)
            resp.raise_for_status()
        try:
            message = resp.json()["choices"][0]["message"]
        except Exception as e:
            logger.error(resp.text)
            raise e
        if "content" not in message and "tool_calls" not in message:
            raise ValueError(f"API returned an invalid response: {message}")
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
