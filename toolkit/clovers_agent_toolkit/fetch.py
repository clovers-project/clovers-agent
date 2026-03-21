from clovers_agent import Event, CloversAgent
from .toolkit import toolkit
from .config import __config__

BRAVE_API_KEY = __config__.BRAVE_API_KEY


@toolkit.tool(
    "web_search",
    "联网搜索",
    {"query": {"type": "string", "description": "搜索关键词"}},
    ["从URL获取资源"],
)
async def _(agent: CloversAgent, event: Event, query: list[str]):
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": BRAVE_API_KEY}
    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": 8}
    resp = await agent.async_client.get(url, headers=headers, params=params, timeout=20.0)
    if resp.status_code != 200:
        return f"搜索失败，状态码：{resp.status_code}"
    try:
        results = resp.json()["web"]["results"]
    except KeyError:
        return "服务器错误，请稍后再试。"
    if not results:
        return f"未找到关于 '{query}' 的相关搜索结果。"
    md_output = [f"### 关于 '{query}' 的搜索结果：\n"]
    for idx, item in enumerate(results, 1):
        title = item.get("title", "无标题")
        link = item.get("url", "#")
        snippet = item.get("description", "无摘要")
        md_output.append(f"{idx}. **[{title}]({link})**\n   摘要: {snippet}\n")
    print("\n".join(md_output))
    return "\n".join(md_output)


@toolkit.tool(
    "web_extractor",
    "读取指定 URL 的网页纯文本内容。当需要从特定网页获取文本信息时使用。",
    {"webpage_url": {"type": "string", "description": "网页的完整 URL 地址"}},
    ["从URL获取资源"],
)
async def _(agent: CloversAgent, event: Event, webpage_url: str):
    if not webpage_url.startswith("http"):
        webpage_url = f"https://{webpage_url}"
    try:
        resp = await agent.async_client.get(webpage_url)
        if resp.status_code != 200:
            return f"获取网页失败，状态码：{resp.status_code}"
        return resp.text
    except Exception:
        return "获取网页失败"


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
