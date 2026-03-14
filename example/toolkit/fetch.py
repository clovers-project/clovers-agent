from clovers_agent import Event, CloversAgent
from .toolkit import toolkit


@toolkit.tool(
    "read_webpage",
    "读取指定 URL 的网页纯文本内容。当需要从特定网页获取文本信息时使用。",
    {"webpage_url": {"type": "string", "description": "网页的完整 URL 地址"}},
    ["从URL获取资源"],
)
async def _(agent: CloversAgent, event: Event, webpage_url: str):
    if not webpage_url.startswith("http"):
        webpage_url = f"https://{webpage_url}"
    resp = await agent.async_client.get(f"https://r.jina.ai/{webpage_url}")
    if resp.status_code != 200:
        return "获取网页失败"
    return resp.text


@toolkit.tool(
    "view_image_url",
    "查看网络图片。当你需要查看用户提供的图片链接时，请调用此工具",
    {"image_url": {"type": "string", "description": "图片的完整 URL 地址"}},
    ["从URL获取资源"],
)
async def _(agent: CloversAgent, event: Event, image_url: str):
    if not image_url.startswith("http"):
        image_url = f"https://{image_url}"
    assert agent.current_input
    if isinstance(agent.current_input["content"], str):
        agent.current_input["content"] = [{"type": "text", "text": agent.current_input["content"]}]
    agent.current_input["content"].append({"type": "image_url", "image_url": {"url": image_url}})
    return "图片已放入用户上下文"
