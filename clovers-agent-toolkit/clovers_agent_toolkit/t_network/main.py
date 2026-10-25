import json
import trafilatura
from clovers_agent import CloversAgent, Turn, Event
from ..utils import retry_request
from ..base import TOOLS, CONFIG
from ..config import CATEGORIES

NETWORK = CATEGORIES.NETWORK
network_config = CONFIG.network
deepseek_search_key = network_config.deepseek_search_key
deepseek_search_url = network_config.deepseek_search_url

TOOLS.create_category("network", "包含各种联网功能，用于从互联网获取信息和资源。")


@TOOLS.register(
    "web_search",
    "联网搜索",
    {"query": {"type": "string", "description": "搜索关键词"}},
    NETWORK,
)
async def _(agent: CloversAgent, session: Turn, event: Event, query: str):
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
    "web_search",
    "联网搜索",
    {
        "query": {"type": "string", "description": "关键词，多个关键词用空格分隔"},
        "question": {"type": "string", "description": "触发搜索的问题，提供该字段有助于更好地理解搜索意图"},
        "sites": {"type": "string", "description": "返回结果限制在指定站点，指定多个时用空格分隔 e.g. github.com huggingface.co."},
        "exclude_sites": {"type": "string", "description": "返回结果排除指定站点，指定多个时用空格分隔。"},
        "exclude_terms": {"type": "array", "description": "返回结果排除指定关键词，指定多个时用空格分隔。"},
        "exact_phrases": {"type": "string", "description": "返回结果精准匹配目标短语"},
        "exclude_phrases": {"type": "string", "description": "排除包含指定短语的网页"},
        "after": {"type": "string", "description": "将结果限制在指定日期之后发布的网页 e.g. 2025-01-01"},
        "before": {"type": "string", "description": "将结果限制在指定日期之前发布的网页"},
    },
    NETWORK,
)
async def _(
    agent: CloversAgent,
    session: Turn,
    event: Event,
):
    headers = {"Authorization": f"Bearer {deepseek_search_key}"}
    payload = {
        "query": query,
        "count": max(1, min(int(count or 5), 15)),
        # 需要正文时再打开(正文很长，慎用)：
        # "need_content": True,
        # 若能把用户原问题拿到，传进去能提升召回：
        # "ori_question": session.user_message[:128],
    }

    resp = await retry_request(
        agent.async_client.post,
        DEEPSEEK_SEARCH_URL,
        headers=headers,
        json=payload,
        timeout=30.0,
    )

    if resp.status_code != 200:
        try:
            return f"搜索请求失败（HTTP {resp.status_code}）：{resp.json()["error"]["message"]}"
        except Exception:
            return f"搜索请求失败（HTTP {resp.status_code}）"

    try:
        results = resp.json()["results"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "服务器未返回合法数据"

    if not results:
        return f"未找到关于 '{query}' 的相关搜索结果。"

    md_output = [f"### 关于 '{query}' 的搜索结果：\n"]
    for idx, item in enumerate(results, 1):
        title = item.get("title") or "无标题"
        link = item.get("url") or "#"
        site = item.get("site_name") or "未知来源"
        date = _fmt_date(item.get("date"))
        summary = (item.get("summary") or item.get("snippet") or "无摘要").strip()

        meta = f"来源: {site}" + (f" | 发布: {date}" if date else "")
        md_output.append(f"{idx}. **[{title}]({link})**\n   {meta}\n   摘要: {summary}\n")

    return "\n".join(md_output)


@TOOLS.register(
    "web_extractor",
    "获取指定网页 url 的纯文本内容。",
    {"url": {"type": "string", "description": "网页的完整 URL 地址"}},
    NETWORK,
)
async def _(agent: CloversAgent, session: Turn, event: Event, url: str):
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
