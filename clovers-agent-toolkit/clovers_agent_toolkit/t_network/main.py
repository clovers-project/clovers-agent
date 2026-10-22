import json
import trafilatura
from clovers_agent import CloversAgent, TurnSession, Event
from ..base import TOOLS, CONFIG, retry_request
from ..config import CATEGORIES

NETWORK = CATEGORIES.NETWORK
BRAVE_API_KEY = CONFIG.BRAVE_API_KEY
BRAVE_URL = CONFIG.BRAVE_URL


TOOLS.create_category("network", "包含各种联网功能，用于从互联网获取信息和资源。")


@TOOLS.register(
    "web_search",
    "联网搜索",
    {"query": {"type": "string", "description": "搜索关键词"}},
    NETWORK,
)
async def _(agent: CloversAgent, session: TurnSession, event: Event, query: str):
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": 8}
    resp = await retry_request(agent.async_client.get, BRAVE_URL, headers=headers, params=params, timeout=30.0)
    try:
        results = resp.json()["web"]["results"]
    except (json.JSONDecodeError, KeyError):
        return "服务器未返回合法数据"
    if not results:
        return f"未找到关于 '{query}' 的相关搜索结果。"
    md_output = [f"### 关于 '{query}' 的搜索结果：\n"]
    for idx, item in enumerate(results, 1):
        title = item.get("title", "无标题")
        link = item.get("url", "#")
        snippet = item.get("description", "无摘要")
        md_output.append(f"{idx}. **[{title}]({link})**\n   摘要: {snippet}\n")
    return "\n".join(md_output)


@TOOLS.register(
    "web_extractor",
    "获取指定网页 url 的纯文本内容。",
    {"url": {"type": "string", "description": "网页的完整 URL 地址"}},
    NETWORK,
)
async def _(agent: CloversAgent, session: TurnSession, event: Event, url: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = await retry_request(agent.async_client.get, url, headers=headers)
    text = trafilatura.extract(
        resp.content,
        url=url,
        include_comments=True,
        include_images=True,
        include_links=True,
        include_formatting=False,
        favor_precision=True,
        output_format="markdown",
    )
    return text or "none"


@TOOLS.register(
    "get_image_by_url",
    "当助手需要查看一个URL图片时调用此工具。",
    {"url": {"type": "string", "description": "图片的完整 URL 地址"}},
    NETWORK,
)
async def _(agent: CloversAgent, session: TurnSession, event: Event, image_url: str):
    if not image_url.startswith("http"):
        image_url = f"https://{image_url}"
    b64url = await session.api.download_url(image_url)
    if not b64url:
        return "资源不存在或非图片格式"
    session.input_image(b64url)
    return "已将图片注入到当前用户输入"
