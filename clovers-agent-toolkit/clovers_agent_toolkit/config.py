# from typing import Literal
from clovers_client import Config as BaseConfig


class Config(BaseConfig):
    BRAVE_API_KEY: str
    BRAVE_URL: str = "https://api.search.brave.com/res/v1/web/search"
    # use_shell: Literal["docker", "local"] = "docker"
    use_shell: bool = True
    """是否使用shell"""
    note_similarity_threshold: float = 0.6
    """笔记内容相似度阈值"""
    reminder_threshold: int = 5
    strong_reminder_threshold: int = 10
