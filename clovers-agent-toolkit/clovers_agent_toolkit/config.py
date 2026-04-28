# from typing import Literal
from clovers_client import Config as BaseConfig


class Config(BaseConfig):
    BRAVE_API_KEY: str
    # use_shell: Literal["docker", "local"] = "docker"
    use_shell: bool = True
    """是否使用shell"""
    session_workspace: bool = True
    """为每个会话创建一个工作空间"""
    note_similarity_threshold: float = 0.78
    """笔记内容相似度阈值"""
