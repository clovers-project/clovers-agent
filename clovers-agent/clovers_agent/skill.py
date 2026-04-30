import frontmatter
from pathlib import Path
from itertools import count
from importlib import import_module
from clovers.core.loader import import_path
from clovers.logger import logger
from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, TYPE_CHECKING
from .typing import ToolMessage, FunctionToolInfo
from .typing.json_schema import JSONSchemaType

if TYPE_CHECKING:
    from .core import CloversAgent

type AgentFunction[**P] = Callable[Concatenate[CloversAgent, Any, P], Coroutine[Any, Any, str] | str]
type WrappedAgentFunction[**P] = Callable[Concatenate[str, CloversAgent, Any, P], Coroutine[Any, Any, tuple[ToolMessage, str]]]
type SkillMD = tuple[str, str, SkillCore.Parameters | None, str]


class SkillCore:
    type Parameters = dict[str, JSONSchemaType]

    def __init__(self) -> None:
        self.category_id = count()
        self.intro_tools: list[FunctionToolInfo] = []
        self.manifest: dict[str, FunctionToolInfo] = {}
        self.chat_hooks: list[AgentFunction] = []
        self.category_hooks: dict[str, AgentFunction] = {}
        self.invoker: dict[str, WrappedAgentFunction] = {}
        self.__map_category_to_id: dict[str, int] = {}
        self.__map_id_to_tools: dict[int, list[FunctionToolInfo]] = {}
        self.categories: dict[str, str] = {}

    def select_tools(self, category: str) -> list[FunctionToolInfo]:
        if category not in self.__map_category_to_id:
            return []
        return self.__map_id_to_tools[self.__map_category_to_id[category]]

    def create_category(self, category: str, description: str):
        if category in self.categories:
            raise ValueError(f"Category {category} already exists")
        self.categories[category] = description

        def decorator(func: AgentFunction) -> AgentFunction:
            self.category_hooks[category] = func
            return func

        return decorator

    def hook(self, func: AgentFunction):
        self.chat_hooks.append(func)
        return func

    def register(
        self,
        name: str,
        description: str,
        parameters: Parameters | None = None,
        category: str | None = None,
    ):
        if name in self.invoker:
            raise ValueError(f"Tool {name} already exists.")
        info: FunctionToolInfo = {"type": "function", "function": {"name": name, "description": description}}
        if parameters:
            info["function"]["parameters"] = {"type": "object", "properties": parameters, "required": list(parameters.keys())}
        # info 是 OpneAI API 要求的 tools 字段中元素的格式
        if not category:
            self.intro_tools.append(info)
        else:
            category_id = self.__map_category_to_id[category] if category in self.__map_category_to_id else next(self.category_id)
            self.__map_category_to_id[category] = category_id
            if category_id not in self.__map_id_to_tools:
                self.__map_id_to_tools[category_id] = []
            self.__map_id_to_tools[category_id].append(info)
        self.manifest[name] = info

        def decorator(func: AgentFunction) -> WrappedAgentFunction:
            async def wrapper(tool_call_id, agent: CloversAgent, event, /, **kwargs):
                logger.info(f"[{agent.name}][CALL][{name}] called")
                logger.debug(kwargs)
                try:
                    content = coro if isinstance(coro := func(agent, event, **kwargs), str) else await coro
                except Exception as e:
                    logger.exception(e)
                    content = "Error"
                logger.debug(f"[{name}][RETURNED] {content}")
                message: ToolMessage = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                return message, name

            self.invoker[name] = wrapper
            return wrapper

        return decorator

    def merge(self, others: "SkillCore"):
        conflict = (others.invoker.keys() & self.invoker.keys()) | (others.category_hooks.keys() & self.category_hooks.keys())
        if conflict:
            return conflict
        self.intro_tools.extend(others.intro_tools)
        self.manifest.update(others.manifest)
        self.category_hooks.update(others.category_hooks)
        self.invoker.update(others.invoker)
        for category, category_id in others.__map_category_to_id.items():
            if category in self.__map_category_to_id:
                self.__map_id_to_tools[self.__map_category_to_id[category]].extend(others.__map_id_to_tools[category_id])
            else:
                new_category_id = next(self.category_id)
                self.__map_id_to_tools[new_category_id] = []
                self.__map_id_to_tools[new_category_id].extend(others.__map_id_to_tools[category_id])
                self.__map_category_to_id[category] = new_category_id
        self.categories.update(others.categories)
        self.chat_hooks.extend(others.chat_hooks)

    def delete_skill(self, category: str | None, name: str | None):
        if name is None:
            tools = self.select_tools(category)  # type: ignore
            if not tools:
                return
            for info in tools:
                _name = info["function"]["name"]
                del self.manifest[_name]
                del self.invoker[_name]
            tools.clear()
        elif name not in self.invoker:
            return
        elif category is None:
            self.intro_tools.remove(self.manifest[name])
            del self.manifest[name]
            del self.invoker[name]
        else:
            tools = self.select_tools(category)
            tools.remove(self.manifest[name])
            del self.manifest[name]
            del self.invoker[name]
            # 不清除 category_id

    def load_skill_md(self, skill: SkillMD, category: str | None = None, func: AgentFunction | None = None):
        name, desc, parameters, content = skill
        register = self.register(name, desc, parameters)
        if skill_func := skill_wrapper(content, func):
            register(skill_func)
        return category, name

    def load_skill(self, skill_path: Path):
        if skill_path.is_file() and skill_path.suffix == ".md" and (skill_md := parse_skill(skill_path)):
            self.delete_skill(None, skill_md[0])
            return self.load_skill_md(skill_md, None)
        elif skill_path.is_dir() and (skill_file := skill_path / "SKILL.md").exists() and (skill_md := parse_skill(skill_file)):
            module = import_module_from_path(skill_path)
            other_mds = [md for file in skill_path.glob("*.md") if not file.samefile(skill_file) if (md := parse_skill(file))]
            if not other_mds:
                self.delete_skill(None, skill_md[0])
                return self.load_skill_md(skill_md, None, getattr(module, skill_md[0], None))
            else:
                category, desc, _, content = skill_md
                self.delete_skill(category, None)
                register = self.create_category(category, desc)
                if skill_func := skill_wrapper(content, getattr(module, category, None)):
                    register(skill_func)
                for md in other_mds:
                    self.load_skill_md(md, category, getattr(module, md[0], None))
                return category, None


def import_module_from_path(module_path: Path):
    if not module_path.exists():
        return
    if not (module_path / "__init__.py").exists():
        return
    try:
        return import_module(import_path(module_path))
    except Exception:
        return


def skill_wrapper(content: str, func: AgentFunction | None = None) -> AgentFunction | None:
    if not content:
        return func
    if func:
        return lambda agent, event, **kwargs: func(agent, event, content=content, **kwargs)
    return lambda agent, event: content


def parse_skill(skill_path: Path) -> SkillMD | None:
    try:
        skill = frontmatter.load(skill_path.read_text())
        name = skill["name"]
        desc = skill["description"]
        parameters: SkillCore.Parameters | None = skill.get("parameters")  # type: ignore
        content = skill.content
    except Exception as e:
        logger.exception(e)
        return
    if not (isinstance(name, str) and isinstance(desc, str)):
        return
    return name, desc, parameters, content
