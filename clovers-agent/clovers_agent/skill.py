from __future__ import annotations
import importlib.util
import frontmatter
from pathlib import Path
from itertools import count
from functools import wraps
from clovers.logger import logger
from collections.abc import Callable
from typing import Concatenate, TYPE_CHECKING
from clovers.base import Coro
from .typing import FunctionToolInfo
from .typing.json_schema import JSONSchemaType

if TYPE_CHECKING:
    from .core import CloversAgent, Event

type ToolFunction[**P] = Callable[Concatenate[CloversAgent, Event, P], str | Coro[str]]
type WrappedToolFunction[**P] = Callable[Concatenate[CloversAgent, Event, P], Coro[str]]
type Parameters[K: str, V: JSONSchemaType] = dict[K, V]
type SkillMD = tuple[str, str, Parameters | None, list[str] | None, str]


class SkillCore:
    def __init__(self) -> None:
        self.category_id = count()
        self.intro_tools: list[FunctionToolInfo] = []
        self.intro_invoker: dict[str, WrappedToolFunction] = {}
        self.manifest: dict[str, FunctionToolInfo] = {}
        self.invoker: dict[str, WrappedToolFunction] = {}
        self.__map_category_to_id: dict[str, int] = {}
        self.__map_id_to_tools: dict[int, list[FunctionToolInfo]] = {}
        self.categories: dict[str, str] = {}
        self.category_hooks: dict[str, list[ToolFunction]] = {}

    def select_tools(self, category: str) -> list[FunctionToolInfo]:
        """选择指定工具组中的所有工具。
        Args:
            category (str): 选择的工具组。

        Returns:
            list[FunctionToolInfo]: 工具列表。
        """

        if category not in self.__map_category_to_id:
            return []
        return self.__map_id_to_tools[self.__map_category_to_id[category]]

    def on_category(self, category: str):
        """添加工具组触发时钩子

        Args:
            category (str): 要绑定钩子的工具组名称。

        Returns:
            Callable: 用于注册分类钩子的装饰器。
        """

        def decorator(func: ToolFunction) -> ToolFunction:
            if category not in self.category_hooks:
                self.category_hooks[category] = []
            self.category_hooks[category].append(func)
            return func

        return decorator

    def create_category(self, category: str, description: str):
        """创建一个新的工具组。

        Args:
            category (str): 工具组名称。
            description (str): 工具组的描述信息。

        Returns:
            Callable: 用于注册分类钩子的装饰器。
        """

        if category in self.categories:
            raise ValueError(f"Category {category} already exists")
        self.categories[category] = description
        return self.on_category(category)

    def intro_decorator(self, info: FunctionToolInfo):
        """注册初始化装饰器

        Args:
            info (FunctionToolInfo): 工具的 OpenAI Function Tool 信息。

        Returns:
            Callable: 用于注册工具函数的装饰器。
        """

        def decorator(func):
            name = info["function"]["name"]
            self.intro_tools.append(info)
            self.manifest[name] = info
            self.intro_invoker[name] = invoker_wrapper(name, func)
            return func

        return decorator

    def category_decorator(self, info: FunctionToolInfo, category: str):
        """注册可分类工具装饰器

        Args:
            info (FunctionToolInfo): 工具的 OpenAI Function Tool 信息。
            category (str): 工具所属的工具组。

        Returns:
            Callable: 用于注册工具函数的装饰器。
        """

        def decorator(func: ToolFunction):
            name = info["function"]["name"]
            category_id = self.__map_category_to_id[category] if category in self.__map_category_to_id else next(self.category_id)
            self.__map_category_to_id[category] = category_id
            if category_id not in self.__map_id_to_tools:
                self.__map_id_to_tools[category_id] = []
            self.__map_id_to_tools[category_id].append(info)
            self.manifest[name] = info
            self.invoker[name] = invoker_wrapper(name, func)
            return func

        return decorator

    def register(
        self,
        name: str,
        description: str,
        parameters: Parameters | None = None,
        category: str | None = None,
        required: list[str] | None = None,
    ):
        """注册一个工具。

        根据是否指定工具组，将工具注册为介绍类工具或指定工具组中的工具。

        Args:
            name (str): 工具名称。
            description (str): 工具描述。
            parameters (Parameters | None): 工具参数的 JSON Schema 定义。
            category (str | None): 工具所属的工具组。如果为 None 则注册为初始化工具。。
            required (list[str] | None): 必填参数名称列表。如果为 None，则所有参数均视为必填。

        Returns:
            Callable: 用于注册工具函数的装饰器。
        """

        if name in self.invoker:
            raise ValueError(f"Tool {name} already exists.")
        info: FunctionToolInfo = {"type": "function", "function": {"name": name, "description": description}}
        if parameters:
            info["function"]["parameters"] = {
                "type": "object",
                "properties": parameters,
                "required": required if required is not None else list(parameters.keys()),
            }
        if not category:
            return self.intro_decorator(info)
        else:
            return self.category_decorator(info, category)

    def remove(self, category: str | None, name: str | None):
        """移除已注册的工具。

        Args:
            category (str | None): 要操作的工具组名称。如果为 None 则为初始化工具。。
            name (str | None): 要移除的工具名称。如果为 None 则表示移除所有工具。
        """

        if name is None:
            if category is None:
                raise ValueError("Can'not remove all intro tools")
            tools = self.select_tools(category)
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
            if self.select_tools(category):
                return
            del self.categories[category]
            if category in self.category_hooks:
                del self.category_hooks[category]
            # 不清除 category_id

    def merge(self, others: "SkillCore"):
        """从其他 SkillCore 加载。

        Args:
            others (SkillCore): 加载的 SkillCore 实例。

        Returns:
            set[str] | None: 若存在冲突，则返回冲突的工具/工具组名称。
        """

        conflict = (others.invoker.keys() & self.invoker.keys()) | (others.category_hooks.keys() & self.category_hooks.keys())
        if conflict:
            return conflict
        self.intro_tools.extend(others.intro_tools)
        self.intro_invoker.update(others.intro_invoker)
        self.manifest.update(others.manifest)
        for category, hooks in others.category_hooks.items():
            if category in self.category_hooks:
                self.category_hooks[category].extend(hooks)
            else:
                self.category_hooks[category] = hooks
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

    def detach(self, others: "SkillCore"):
        """移除从其他 SkillCore 加载的内容。

        Args:
            others (SkillCore): 被移除的 SkillCore 实例。
        """

        for info in others.intro_tools:
            self.intro_tools.remove(info)
        for name in others.intro_invoker:
            del self.intro_invoker[name]
        for category, hooks in others.category_hooks.items():
            for hook in hooks:
                self.category_hooks[category].remove(hook)
        for category in others.categories:
            tools = self.select_tools(category)
            for info in others.select_tools(category):
                name = info["function"]["name"]
                tools.remove(info)
                del self.manifest[name]
                del self.invoker[name]
            tools = self.select_tools(category)
            if tools:
                continue
            del self.categories[category]
            if category in self.category_hooks:
                del self.category_hooks[category]

    def load_skill_md(self, skill: SkillMD, category: str | None = None, func: ToolFunction | None = None):
        """从 Skill Markdown 数据中注册工具。

        Args:
            skill (SkillMD): 解析后的 Skill Markdown 数据。
            category (str | None): 工具所属的工具组。
            func (ToolFunction | None): 工具对应的实际函数。

        Returns:
            tuple[str | None, str]: 工具组名称和工具名称。
        """

        name, desc, parameters, required, content = skill
        register = self.register(name, desc, parameters, category, required)
        if skill_func := skill_wrapper(content, func):
            register(skill_func)
        return category, name

    def load_skill(self, skill_path: Path):
        """从指定路径加载 Skill。

        1. 当 `skill_path 指向 Markdown 文件或只有 `SKILL.md` 一篇 `.md` 文件的文件夹:
            该 Markdown文件会被注册成初始化工具。
        2. 当 `skill_path 指向包含 `SKILL.md` 和其他 `.md` 文件的文件夹:
            - `SKILL.md` Front Matter 中的 `name` 会作为工具组名称，`description` 会作为工具组描述，正文会作为工具组 hook 的返回内容。
            - `*.md` 会被注册为该工具组下的工具。

        当 `skill_path` 指向文件夹时，若 `./skill.py` 内有与 `./*.md` 内定义的工具同名的函数。

        那么 CloversAgent 在调用该工具时会得到 `skill.py` 中同名函数的返回值。

        同名函数必须是一个第3个参数名固定为 `content` ，值为 Markdown 正文文本的 `ToolFunction`

        详见 `AgentSkills/weather`

        Args:
            skill_path (Path): Skill 文件或 Skill 目录的路径。

        Returns:
            tuple[str | None, str | None] | None: 加载成功时返回工具组名称和工具名称，失败时返回 None。
        """

        if skill_path.is_file() and skill_path.suffix == ".md" and (skill_md := parse_skill(skill_path)):
            self.remove(None, skill_md[0])
            return self.load_skill_md(skill_md, None)
        md_file = skill_path / "SKILL.md"
        if not (md_file.exists() and (skill_md := parse_skill(md_file))):
            return
        module = load_module_from_path(skill_md[0], skill_path / "skill.py")
        other_mds = [md for file in skill_path.glob("*.md") if not file.samefile(md_file) if (md := parse_skill(file))]
        if not other_mds:
            self.remove(None, skill_md[0])
            return self.load_skill_md(skill_md, None, getattr(module, skill_md[0], None))
        category, desc, *_, content = skill_md
        self.remove(category, None)
        register = self.create_category(category, desc)
        if skill_func := skill_wrapper(content, getattr(module, category, None)):
            register(skill_func)
        for md in other_mds:
            self.load_skill_md(md, category, getattr(module, md[0], None))
        return category, None


def load_module_from_path(module_name: str, file: Path):
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        return
    try:
        spec.loader.exec_module(module)
    except:
        return
    return module


def invoker_wrapper[**P](name: str, func: ToolFunction[P]) -> WrappedToolFunction[P]:
    @wraps(func)
    async def wrapper(agent: CloversAgent, event, /, *args: P.args, **kwargs: P.kwargs) -> str:
        logger.info(f"[{agent.name}][CALL][{name}] called")
        logger.debug(kwargs)
        content = coro if isinstance(coro := func(agent, event, *args, **kwargs), str) else await coro
        logger.debug(f"[{name}][RETURNED] {content}")
        return content

    return wrapper


def skill_wrapper(content: str, func: ToolFunction | None = None) -> ToolFunction | None:
    if not content:
        return func
    if func:

        @wraps(func)
        def wrapper(agent, event, **kwargs):
            return func(agent, event, content=content, **kwargs)

        return wrapper
    return lambda agent, event: content


def parse_skill(skill_path: Path) -> SkillMD | None:
    try:
        skill = frontmatter.loads(skill_path.read_text("utf-8"))
        name = skill["name"]
        desc = skill["description"]
        if schema := skill.get("parameters"):
            parameters = schema["properties"]  # type: ignore
            required = schema.get("required")  # type: ignore
        else:
            parameters = None
            required = None
        content = skill.content.strip()
    except Exception as e:
        logger.exception(e)
        return
    if not (isinstance(name, str) and isinstance(desc, str)):
        return
    return name.replace("-", "_"), desc, parameters, required, content
