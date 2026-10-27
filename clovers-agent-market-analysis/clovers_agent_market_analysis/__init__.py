__version__ = "0.3.0"
__all__ = ["TOOLS"]

if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from .main import TOOLS
