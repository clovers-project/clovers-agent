from pydantic import BaseModel
from clovers.config import Config as CloversConfig


class Config(BaseModel):
    BRAVE_API_KEY: str
    use_shell: bool = True
    """是否使用shell"""
    session_workspace: bool = True
    """为每个会话创建一个工作空间"""

    @classmethod
    def sync_config(cls):
        """获取 `CloversConfig.environ()[__package__]` 配置并将默认配置同步到全局配置中。"""
        __config_dict__: dict = CloversConfig.environ().setdefault(__package__, {})
        __config_dict__.update((__config__ := cls.model_validate(__config_dict__)).model_dump())
        return __config__


__config__ = Config.sync_config()
