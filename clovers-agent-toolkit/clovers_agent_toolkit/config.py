# from typing import Literal
from enum import Enum
from pydantic import BaseModel
from clovers_client import Config as BaseConfig


class CATEGORIES(str, Enum):
    MEMORY = "memory"
    WORKSPACE = "workspace"
    NETWORK = "network"
    DEBUG_UTILS = "debug_utils"


class MemoryConfig(BaseModel):
    profile_memory_size: int = 30
    """个人档案保留记忆条数"""
    strong_reminder_threshold: int = 16
    """强提醒个人档案更新对话轮数"""
    reminder_threshold: int = 8
    """提醒个人档案更新对话轮数"""


class NetworkConfig(BaseModel):
    deepseek_search_key: str = "sk-xxx"
    """DeepSeek API Key"""
    deepseek_search_url: str = "https://api.deepseek.com/v1/tools/search"
    """DeepSeek Search API URL"""


class WorkspaceConfig(BaseModel):
    use_shell: bool = True
    """是否使用shell"""
    max_subagent_count: int = 8
    """最大子代理数量"""
    max_download_return_size: int = 4096
    """最大下载作为工具返回文件大小"""
    docker_image: str = "nikolaik/python-nodejs:python3.12-nodejs20"
    """docker镜像名称"""


class Config(BaseConfig):
    use_categories: list[str] = [
        CATEGORIES.MEMORY,
        CATEGORIES.WORKSPACE,
        CATEGORIES.NETWORK,
    ]
    memory: MemoryConfig = MemoryConfig()
    network: NetworkConfig = NetworkConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
