# from typing import Literal
from clovers_client import Config as BaseConfig


class Config(BaseConfig):
    BRAVE_API_KEY: str
    BRAVE_URL: str = "https://api.search.brave.com/res/v1/web/search"
    # use_shell: Literal["docker", "local"] = "docker"
    use_shell: bool = True
    """是否使用shell"""
    docker_image: str = "nikolaik/python-nodejs:python3.12-nodejs20"
    """docker镜像名称"""
    reminder_threshold: int = 5
    """提醒个人档案更新对话轮数"""
    strong_reminder_threshold: int = 10
    """强提醒个人档案更新对话轮数"""
    debug_mode: bool = False
    """是否加载调试工具"""
