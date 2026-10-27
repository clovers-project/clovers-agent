import json
import httpx
import trafilatura
from clovers_agent import CloversAgent, Turn, Event
from ..utils import retry_request
from ..base import TOOLS, CONFIG
from ..config import CATEGORIES

from typing import cast
from .typing import SearchResult

NETWORK = CATEGORIES.NETWORK
network_config = CONFIG.network
deepseek_search_key = network_config.deepseek_search_key
deepseek_search_url = network_config.deepseek_search_url

TOOLS.create_category("network", "包含搜索和网页提取工具")


@TOOLS.register(
    "web_search",
    "",
    {
        "query": {
            "type": "string",
            "description": "Space-separated search keywords used for matching",
        },
        "question": {
            "type": "string",
            "description": "Describe the question to be answered to help the search service understand the search intent.",
        },
        "sites": {
            "type": "array",
            "items": {"type": "string"},
            "description": 'Restrict results to the specified domains. e.g. ["github.com", "huggingface.co"]',
        },
        "exclude_sites": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exclude results from the specified domains. Same format as `sites`.",
        },
        "exclude_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exclude webpages that contain any of the specified keywords.",
        },
        "exact_phrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Return only webpages that contain the specified phrases.",
        },
        "exclude_phrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Return only webpages that do not contain the specified phrases.",
        },
        "after": {
            "type": "string",
            "description": "Only include webpages published after this date. Accepts YYYY or YYYY-MM-DD. e.g. `2025-01-01`.",
        },
        "before": {
            "type": "string",
            "description": "Only include webpages published before this date. Same format as `after`.",
        },
    },
    NETWORK,
    required=["query"],
)
async def _(
    agent: CloversAgent,
    turn: Turn,
    event: Event,
    query: str,
    question: str | None = None,
    sites: list[str] | None = None,
    exclude_sites: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    exact_phrases: list[str] | None = None,
    exclude_phrases: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
):
    headers = {"Authorization": f"Bearer {deepseek_search_key}"}
    query_list = [query]
    if sites and isinstance(sites, list):
        query_list.append(" OR ".join(f"site:{site}" for site in sites))
    if exclude_sites and isinstance(exclude_sites, list):
        query_list.append(" ".join(f"-site:{site}" for site in exclude_sites))
    if exclude_terms and isinstance(exclude_terms, list):
        query_list.append(" ".join(f"-{term}" for term in exclude_terms))
    if exact_phrases and isinstance(exact_phrases, list):
        query_list.append(" ".join(f'"{phrase}"' for phrase in exact_phrases))
    if exclude_phrases and isinstance(exclude_phrases, list):
        query_list.append(" ".join(f'-"{phrase}"' for phrase in exclude_phrases))
    if after:
        query_list.append(f"after:{after}")
    if before:
        query_list.append(f"before:{before}")
    payload = {
        "query": " ".join(query_list),
        "count": 10,
        "ori_question": question or event.message,
        "need_content": True,
    }
    try:
        resp = await retry_request(agent.async_client.post, deepseek_search_url, headers=headers, json=payload, timeout=30.0)
    except httpx.HTTPStatusError as e:
        resp = e.response
        try:
            return f"搜索请求失败（HTTP {resp.status_code}）：{resp.json()["error"]["message"]}"
        except Exception:
            return f"搜索请求失败（HTTP {resp.status_code}）"
    except Exception:
        return "搜索请求失败"
    try:
        results = cast(list[SearchResult], resp.json()["results"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return "服务器未返回合法数据"
    if not results:
        return f"未找到关于 '{query}' 的相关搜索结果。"
    results.sort(key=lambda x: x.setdefault("rerank_score", 0.0), reverse=True)
    avg_score = sum(x["rerank_score"] for x in results) / len(results)
    md_output = [f"### 关于 '{query}' 的搜索结果：\n"]
    for idx, item in enumerate(results[:6], 1):
        if idx > 3 and item["rerank_score"] < avg_score:
            break
        title = item.get("title") or "无标题"
        site = item.get("site_name") or "未知来源"
        date = item.get("date")
        summary = (item.get("summary") or item.get("snippet") or "无").strip()
        if date:
            meta = f"来源: {site}  | 发布: {date}"
        else:
            meta = f"来源: {site}"
        md_output.append(f"### {title}\n{meta}\n{summary}")
    return "\n---\n".join(md_output)


@TOOLS.register(
    "web_extractor",
    "获取指定网页 url 的纯文本内容。",
    {"url": {"type": "string", "description": "网页的完整 URL 地址"}},
    NETWORK,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, url: str):
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
