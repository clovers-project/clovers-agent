# from typing import Literal
from enum import Enum
from pydantic import BaseModel
from clovers_client import Config as BaseConfig


class CATEGORIES(str, Enum):
    WORKSPACE = "workspace"
    NETWORK = "network"
    DEBUG_UTILS = "debug_utils"


class NetworkConfig(BaseModel):
    deepseek_search_key: str = "sk-xxx"
    """DeepSeek Search API Key"""
    deepseek_search_url: str = "https://api.deepseek.com/v1/tools/search"
    """DeepSeek Search API URL"""
    deepseek_fetch_key: str = ""
    """DeepSeek Fetch API Key"""
    deepseek_fetch_url: str = "https://api.deepseek.com/v1/tools/fetch"
    """DeepSeek Fetch API URL"""


class WorkspaceConfig(BaseModel):
    use_shell: bool = False
    """是否使用shell"""
    max_subagent_count: int = 8
    """最大子代理数量"""
    max_download_return_size: int = 4096
    """最大下载作为工具返回文件大小"""
    docker_image: str = "nikolaik/python-nodejs:python3.12-nodejs20"
    """docker镜像名称"""
    shared_workspace: bool = False
    """是否使用共享工作空间"""


class Config(BaseConfig):
    use_categories: list[str] = [
        CATEGORIES.WORKSPACE,
        CATEGORIES.NETWORK,
    ]
    network: NetworkConfig = NetworkConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
