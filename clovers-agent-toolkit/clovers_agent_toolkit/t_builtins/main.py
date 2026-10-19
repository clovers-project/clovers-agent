from clovers_agent import CloversAgent, Event
from clovers.logger import logger
from ..base import TOOLS


@TOOLS.register(
    "get_image_by_url",
    "当助手需要查看一个URL图片时调用此工具。",
    {"image_url": {"type": "string", "description": "图片的完整 URL 地址"}},
    "network",
)
async def _(agent: CloversAgent, event: Event, image_url: str):
    if not image_url.startswith("http"):
        image_url = f"https://{image_url}"
    session = agent.current_session(event)
    b64url = await session.api.download_url(image_url)
    if not b64url:
        return "资源不存在或非图片格式"
    session.current_input.append({"type": "image_url", "image_url": {"url": b64url}})
    return "OK"
