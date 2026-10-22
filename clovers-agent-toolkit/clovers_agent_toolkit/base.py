if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from clovers_agent import AgentTools
from clovers_agent.utils import retry_httpx, httpx_request
from .config import Config, CATEGORIES

retry_request = retry_httpx(5)(httpx_request)

TOOLS = AgentTools()
CONFIG = Config.sync_config(__package__)


__all__ = ["TOOLS", "CONFIG", "CATEGORIES"]
