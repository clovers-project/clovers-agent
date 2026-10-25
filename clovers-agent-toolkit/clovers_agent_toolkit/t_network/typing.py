from typing import TypedDict


class SearchResult(TypedDict):
    rerank_score: float
    title: str
    url: str
    snippet: str
    summary: str
    content: str
    site_name: str
    site_icon: str
    date: str
    authority: int
