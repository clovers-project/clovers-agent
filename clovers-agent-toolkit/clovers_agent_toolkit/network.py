from clovers_agent import CloversAgent, Event
from clovers_agent.utils import is_base64
from .toolkit import TOOLS, CONFIG

BRAVE_API_KEY = CONFIG.BRAVE_API_KEY
BRAVE_URL = CONFIG.BRAVE_URL

TOOLS.create_category("network", "包含各种联网功能，用于从互联网获取信息和资源。")


@TOOLS.register(
    "web_search",
    "联网搜索",
    {"query": {"type": "string", "description": "搜索关键词"}},
    "network",
)
async def _(agent: CloversAgent, event: Event, query: list[str]):
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": 8}
    resp = await agent.async_client.get(BRAVE_URL, headers=headers, params=params, timeout=30.0)
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
    return "\n".join(md_output)


@TOOLS.register(
    "web_extractor",
    "获取指定网页 url 的纯文本内容。",
    {"webpage_url": {"type": "string", "description": "网页的完整 URL 地址"}},
    "network",
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


ALLOWED_TYPES = ("application/json", "text/", "application/xml")


@TOOLS.register(
    "http_request",
    "使用 get/post 发起网络请求",
    {
        "method": {"type": "string", "description": "请求方法", "enum": ["get", "post"]},
        "url": {"type": "string", "description": "请求的 url 地址"},
        "headers": {"type": "object", "description": "请求头 json"},
        "data": {"type": "object", "description": "当 method 为 get 时此参数为请求参数，当 method 为 post 时此参数为请求体 json"},
    },
    "network",
    required=["method", "url"],
)
async def _(agent: CloversAgent, event: Event, method: str, url: str, headers: dict = {}, data: dict = {}):
    method = method.lower()
    if method == "get":
        resp = await agent.async_client.get(url, params=data, headers=headers)
    elif method == "post":
        resp = await agent.async_client.post(url, json=data, headers=headers)
    else:
        return f"Invalid method: {method}"
    content = resp.text
    if not content:
        return "返回结果为空"
    content_type = resp.headers.get("Content-Type", "").lower()
    if not any(t in content_type for t in ALLOWED_TYPES):
        if len(content) >= 1000:
            return "返回结果长度过长"
        if is_base64(content):
            return "返回结果为 base64 编码, 已拦截。"
    return content


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
    session.current_input.append({"type": "image_url", "image_url": {"url": image_url}})
    return "OK"
