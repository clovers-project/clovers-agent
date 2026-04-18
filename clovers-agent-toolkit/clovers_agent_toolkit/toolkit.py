if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from clovers_agent import SkillCore
from .config import Config

TOOLS = SkillCore("Agent Toolkit")
CONFIG = Config.sync_config(__package__)
