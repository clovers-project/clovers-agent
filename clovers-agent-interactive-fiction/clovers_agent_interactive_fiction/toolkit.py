if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from clovers_agent import SkillCore

TOOLS = SkillCore()
