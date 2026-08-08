__version__ = "0.1.0"
__all__ = ["TOOLS"]

if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from clovers_agent import SkillCore

TOOLS = SkillCore()
