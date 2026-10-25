if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from clovers_agent import AgentTools
from .config import Config, CATEGORIES

TOOLS = AgentTools()
CONFIG = Config.sync_config(__package__)


__all__ = ["TOOLS", "CONFIG", "CATEGORIES"]
