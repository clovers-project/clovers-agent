from __future__ import annotations
import contextlib
import importlib.util
import frontmatter
from pathlib import Path
from functools import wraps
from clovers.logger import logger
from collections import Counter

from typing import TYPE_CHECKING, overload, Protocol
from collections.abc import Iterable
from clovers_client import Event as BaseEvent
from .typing import FunctionToolInfo
from .typing.json_schema import JSONSchemaType

if TYPE_CHECKING:
    from .session import Turn
    from .agent import CloversAgent


class SyncInvoker[**P](Protocol):
    def __call__(self, agent: CloversAgent, turn: Turn, event: Event, /, *args: P.args, **kwargs: P.kwargs) -> str: ...
class Invoker[**P](Protocol):
    async def __call__(self, agent: CloversAgent, turn: Turn, event: Event, /, *args: P.args, **kwargs: P.kwargs) -> str: ...


type Parameters = dict[str, JSONSchemaType]
type SkillMD = tuple[str, str, Parameters | None, list[str] | None, str]
type IntroK = tuple[None, str]
type CategoryK = tuple[str, None]
type SingleK = tuple[str, str]


def cast_invoker[F: Invoker](func: F) -> F:
    return func


class Event(BaseEvent, Protocol):
    extra_context: list[str] = []


EMPTY_ITER = ()


class AgentTools:
    name: str = ""

    def __init__(self) -> None:
        # 初始化工具
        self.intro_manifest: dict[str, FunctionToolInfo] = {}
        self.intro_invoker: dict[str, Invoker] = {}
        self.intro_hooks: dict[str, Counter[Invoker]] = {}
        # 分类工具
        self.tool_manifest: dict[str, FunctionToolInfo] = {}
        self.tool_invoker: dict[str, Invoker] = {}
        self.tools_by_category: dict[str, list[str]] = {}
        self.category_hooks: dict[str, Counter[Invoker]] = {}
        self.category_descriptions: dict[str, str] = {}
        # inner
        self._merged: set[AgentTools] = set()

    def on_intro(self, intro: str):
        """添加工具组初始化钩子

        Args:
            category (str): 要绑定钩子的工具组名称。

        Returns:
            Callable: 用于注册工具组初始化钩子的装饰器。
        """

        def decorator(func: Invoker):
            if intro not in self.intro_hooks:
                self.intro_hooks[intro] = Counter()
            self.intro_hooks[intro][func] += 1
            return func

        return decorator

    def on_category(self, category: str):
        """添加工具组触发时钩子

        Args:
            category (str): 要绑定钩子的工具组名称。

        Returns:
            Callable: 用于注册分类钩子的装饰器。
        """

        def decorator(func: Invoker):
            if category not in self.category_hooks:
                self.category_hooks[category] = Counter()
            self.category_hooks[category][func] += 1
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

        if category in self.category_descriptions:
            raise ValueError(f"Category {category} already exists")
        self.category_descriptions[category] = description
        return self.on_category(category)

    def intro_decorator(self, info: FunctionToolInfo):
        """注册初始化装饰器

        Args:
            info (FunctionToolInfo): 工具的 OpenAI Function Tool 信息。

        Returns:
            Callable: 用于注册工具函数的装饰器。
        """

        def decorator(func: Invoker):
            name = info["function"]["name"]
            if name in self.intro_invoker:
                raise ValueError(f"Intro {name} already exists.")

            self.intro_manifest[name] = info
            self.intro_invoker[name] = func
            return func

        return decorator

    def category_decorator(self, info: FunctionToolInfo, category: str | Iterable[str]):
        """注册可分类工具装饰器

        Args:
            info (FunctionToolInfo): 工具的 OpenAI Function Tool 信息。
            category (str): 工具所属的工具组。

        Returns:
            Callable: 用于注册工具函数的装饰器。
        """

        def decorator(func: Invoker):
            name = info["function"]["name"]
            if name in self.tool_invoker:
                raise ValueError(f"Tool {name} already exists.")
            if isinstance(category, str):
                categories = (category,)
            else:
                categories = category
            for c in categories:
                if not isinstance(c, str):
                    continue
                if c not in self.tools_by_category:
                    self.tools_by_category[c] = []
                self.tools_by_category[c].append(name)
            self.tool_manifest[name] = info
            self.tool_invoker[name] = func
            return func

        return decorator

    def register(
        self,
        name: str,
        description: str,
        parameters: Parameters | None = None,
        category: str | Iterable[str] | None = None,
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
        info: FunctionToolInfo = {"type": "function", "function": {"name": name, "description": description}}
        if parameters:
            info["function"]["parameters"] = {
                "type": "object",
                "properties": parameters,
                "required": required if required is not None else list(parameters),
            }
        if category:
            return self.category_decorator(info, category)
        else:
            return self.intro_decorator(info)

    def select_tools(self, category: str) -> tuple[FunctionToolInfo, ...]:
        """选择指定工具组中的所有工具。
        Args:
            category (str): 选择的工具组。

        Returns:
            tuple[FunctionToolInfo, ...]: 工具列表。
        """
        if category not in self.tools_by_category:
            return EMPTY_ITER
        namelist = self.tools_by_category[category]
        return tuple(self.tool_manifest[name] for name in namelist)

    def remove_category(self, category: str):
        """移除工具组。

        Args:
            category (str): 工具组名。
        """
        if category in self.tools_by_category:
            for name in self.tools_by_category[category]:
                del self.tool_manifest[name]
                del self.tool_invoker[name]
            del self.tools_by_category[category]
        if category in self.category_descriptions:
            del self.category_descriptions[category]
        if category in self.category_hooks:
            del self.category_hooks[category]

    def validate_manifest(self):
        """验证工具清单的完整性。
        1. 可选的工具信息必须是清单中的对象。
        2. 清单中的对象必须拥有实现。
        """
        unimplemented_intros = self.intro_manifest.keys() - self.intro_invoker.keys()
        if unimplemented_intros:
            raise ValueError(f"Unimplemented intros: {unimplemented_intros}")
        for intro in tuple(self.intro_hooks):
            intro_hooks = self.intro_hooks[intro]
            hooks = +intro_hooks
            if hooks:
                intro_hooks.clear()
                intro_hooks.update(hooks)
            else:
                del self.intro_hooks[intro]
        unimplemented_tools = self.tool_manifest.keys() - self.tool_invoker.keys()
        if unimplemented_tools:
            raise ValueError(f"Unimplemented tools: {unimplemented_tools}")
        # 存在 Category Desc 一定是本实例 Create 的 Category
        # 实例可以为非自己创建的 Category 添加 Hook
        # 但是自己创建的 Category 必须要有 Hook 或者 Tools
        # 这里验证是修复而非报错
        for category in tuple(self.category_descriptions):
            if category in self.tools_by_category and not self.tools_by_category[category]:
                del self.tools_by_category[category]
            if category in self.category_hooks:
                category_hooks = self.category_hooks[category]
                hooks = +category_hooks
                if hooks:
                    category_hooks.clear()
                    category_hooks.update(hooks)
                else:
                    del self.category_hooks[category]
            if category not in self.tools_by_category and category not in self.category_hooks:
                del self.category_descriptions[category]
        manifest_keys = set(self.tool_manifest)
        if any(not manifest_keys.issuperset(namelist) for namelist in self.tools_by_category.values()):
            raise ValueError("Manifest mismatch")

    def merge(self, others: "AgentTools"):
        """从其他 SkillCore 加载。

        Args:
            others (SkillCore): 加载的 SkillCore 实例。
        """
        if others in self._merged:
            logger.warning("[%s][MERGE] %r already merged", self.name, others)
            return
        others.validate_manifest()
        conflict_intros = others.intro_manifest.keys() & self.intro_manifest.keys()
        if conflict_intros:
            raise ValueError(f"Intro Manifest conflict: {conflict_intros}")
        conflict_tools = others.tool_manifest.keys() & self.tool_manifest.keys()
        if conflict_tools:
            raise ValueError(f"Tool Manifest conflict: {conflict_tools}")
        conflict_categories = others.category_descriptions.keys() & self.category_descriptions.keys()
        if conflict_categories:
            raise ValueError(f"Category conflict: {conflict_categories}")
        # Intro 合并
        self.intro_manifest.update(others.intro_manifest)
        self.intro_invoker.update((k, others.intro_invoker[k]) for k in others.intro_manifest)
        for intro, hooks in others.intro_hooks.items():
            if intro in self.intro_hooks:
                self.intro_hooks[intro].update(hooks)
            else:
                self.intro_hooks[intro] = hooks.copy()
        # Tools 合并
        self.tool_manifest.update(others.tool_manifest)
        self.tool_invoker.update((k, others.tool_invoker[k]) for k in others.tool_manifest)
        # others 不可以创建一个已有的类，但是可以向一个已有的类添加工具。
        for category, namelist in others.tools_by_category.items():
            if category in self.tools_by_category:
                self.tools_by_category[category].extend(namelist)
            else:
                self.tools_by_category[category] = namelist.copy()
        for category, hooks in others.category_hooks.items():
            if category in self.category_hooks:
                self.category_hooks[category].update(hooks)
            else:
                self.category_hooks[category] = hooks.copy()
        self.category_descriptions.update(others.category_descriptions)
        self._merged.add(others)

    def detach(self, others: "AgentTools"):
        """移除从其他 SkillCore 加载的内容。

        Args:
            others (SkillCore): 被移除的 SkillCore 实例，`others` 必须是 `self.merge` 过的实例 。
            请勿在 `self.merge` 后修改 `others`，否则会导致 `self.detach` 失败或工具残留
        """
        if others not in self._merged:
            logger.warning("[%s][DETACH] %r not merged", self.name, others)
            return
        others.validate_manifest()
        # Intro 验证
        if missing_intros := others.intro_manifest.keys() - self.intro_manifest.keys():
            raise ValueError(f"Cannot detach: intro_manifest keys {missing_intros} are not present in the current instance.")
        for intro, others_i_hooks in others.intro_hooks.items():
            if not intro in self.intro_hooks:
                raise ValueError(f"Cannot detach: intro '{intro}' does not exist in the current instance.")
            self_i_hooks = self.intro_hooks[intro]
            if others_i_hooks - self_i_hooks:
                raise ValueError(f"Cannot detach: intro '{intro}' from others are not present in the current instance.")
        # Tool 验证
        if missing_tools := others.tool_manifest.keys() - self.tool_manifest.keys():
            raise ValueError(f"Cannot detach: manifest keys {missing_tools} are not present in the current instance.")
        for category, other_namelist in others.tools_by_category.items():
            if not category in self.tools_by_category:
                raise ValueError(f"Cannot detach: category '{category}' does not exist in the current instance.")
            self_namelist = self.tools_by_category[category]
            if set(other_namelist).difference(self_namelist):
                raise ValueError(f"Cannot detach: category '{category}' from others are not present in the current instance.")
        for category, other_c_hooks in others.category_hooks.items():
            if not category in self.category_hooks:
                raise ValueError(f"Cannot detach: hooks category '{category}' does not exist in the current instance.")
            self_c_hooks = self.category_hooks[category]
            if other_c_hooks - self_c_hooks:
                raise ValueError(f"Cannot detach: hooks category '{category}' from others are not present in the current instance.")
        for category in others.category_descriptions:
            if not category in self.category_descriptions:
                raise ValueError(f"Cannot detach: category description for '{category}' does not exist in the current instance.")
        # 移除 Intro
        for name in others.intro_manifest.keys():
            del self.intro_manifest[name]
            del self.intro_invoker[name]
        for intro, others_i_hooks in others.intro_hooks.items():
            self_i_hooks = self.intro_hooks[intro]
            hooks = self_i_hooks - others_i_hooks
            if not hooks:
                del self.intro_hooks[intro]
            else:
                self_i_hooks.clear()
                self_i_hooks.update(hooks)
        # 移除 Tool
        for name in others.tool_manifest.keys():
            del self.tool_manifest[name]
            del self.tool_invoker[name]
        for category, other_namelist in others.tools_by_category.items():
            self_namelist = self.tools_by_category[category]
            for other_name in other_namelist:
                self_namelist.remove(other_name)
            if not self_namelist:
                del self.tools_by_category[category]
        for category, other_c_hooks in others.category_hooks.items():
            self_c_hooks = self.category_hooks[category]
            hooks = self_c_hooks - other_c_hooks
            if not hooks:
                del self.category_hooks[category]
            else:
                self_c_hooks.clear()
                self_c_hooks.update(hooks)
        # others 注册的类别可能被其他工具添加新工具，所以当前还有工具的类别不丢失 Desc
        for category in tuple(self.category_descriptions):
            if category in self.tools_by_category:
                continue
            if category in self.category_hooks:
                continue
            del self.category_descriptions[category]
        self._merged.remove(others)


class AgentSkills(AgentTools):
    def __init__(self, skill_dirs: list[str]) -> None:
        super().__init__()
        self.scripts_map: dict[str, dict[str, str]] = {}
        self.references_map: dict[str, dict[str, str]] = {}
        self._skill_dirs = skill_dirs
        self._loaded_intros: set[str] = set()
        self._loaded_categories: set[str] = set()

    @overload
    def load_skill_md(self, skill: SkillMD, category: None = None, func: Invoker | None = None) -> IntroK | None: ...
    @overload
    def load_skill_md(self, skill: SkillMD, category: str = ..., func: Invoker | None = None) -> SingleK | None: ...

    def load_skill_md(self, skill: SkillMD, category: str | None = None, func: Invoker | None = None):
        """从 Skill Markdown 数据中注册工具。

        Args:
            skill (SkillMD): 解析后的 Skill Markdown 数据。
            category (str | None): 工具所属的工具组。
            func (ToolFunction | None): 工具对应的实际函数。

        Returns:
            tuple[str | None, str]: 工具组名称和工具名称。
        """

        name, desc, parameters, required, content = skill
        if f := md_wrapper(content, func, None if category else name):
            self.register(name, desc, parameters, category, required)(f)
            return category, name

    def load_skill(self, skill_path: Path):
        """从指定路径加载 Skill。

        1. 当 `skill_path` 指向 Markdown 文件或只有 `SKILL.md` 一篇 `.md` 文件的文件夹:
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
        """

        if skill_path.is_file() and skill_path.suffix == ".md" and (skill_md := parse_skill(skill_path)):
            intro_k = self.load_skill_md(skill_md, None)
            if intro_k:
                logger.info('[%s][LOAD SKILL] "Intro - %s" loaded', self.name, intro_k[1])
            return
        md_file = skill_path / "SKILL.md"
        if not (md_file.exists() and (skill_md := parse_skill(md_file))):
            return
        module = load_module_from_path(skill_md[0], skill_path / "skill.py")
        other_mds = [md for file in skill_path.glob("*.md") if not file.samefile(md_file) if (md := parse_skill(file))]
        script_path = skill_path / "scripts"
        reference_path = skill_path / "references"
        scripts = tuple(file for file in script_path.rglob("*") if file.is_file())
        references = tuple(file for file in reference_path.rglob("*") if file.is_file())
        if not other_mds and not scripts and not references:
            intro_k = self.load_skill_md(skill_md, None, getattr(module, skill_md[0], None))
            if intro_k:
                logger.info('[%s][LOAD SKILL] "Intro - %s" loaded', self.name, intro_k[1])
            return
        category, desc, *_, content = skill_md
        # 加载的 skill 必须独占一个工具组
        register = self.create_category(category, desc)
        if f := md_wrapper(content, getattr(module, category, None)):
            register(f)
        try:
            for md in other_mds:
                self.load_skill_md(md, category, getattr(module, md[0], None))
        except Exception:
            self._loaded_categories.discard(category)
            self.remove_category(category)
            raise
        for script_file in scripts:
            if not category in self.scripts_map:
                self.scripts_map[category] = {}
            path = category / script_file.relative_to(script_path)
            self.scripts_map[category][path.as_posix()] = script_file.resolve().as_posix()
        for reference_file in references:
            if not category in self.references_map:
                self.references_map[category] = {}
            path = category / reference_file.relative_to(reference_path)
            self.references_map[category][path.as_posix()] = reference_file.resolve().as_posix()
        logger.info('[%s][LOAD SKILL] "Category - %s" loaded', self.name, category)

    @property
    def skill_dirs(self):
        return (_p for _s_dir in self._skill_dirs if (_dir := Path(_s_dir).expanduser()).exists() for _p in _dir.iterdir())

    def reload_skills(self):
        for name in self._loaded_intros:
            del self.intro_manifest[name]
            del self.intro_invoker[name]
        self._loaded_intros.clear()
        for category in self._loaded_categories:
            self.remove_category(category)
        self._loaded_categories.clear()
        self.scripts_map.clear()
        self.references_map.clear()
        for path in self.skill_dirs:
            self.load_skill(path)


def md_wrapper(content: str | None = None, func: Invoker | None = None, intro: str | None = None) -> Invoker | None:
    if not content:
        return func
    if func:
        return wraps(func)(lambda agent, turn, event, **kwargs: func(agent, turn, event, content=content, **kwargs))
    if not intro:

        async def tool_invoker(agent: CloversAgent, turn: Turn, event: Event):
            return content

        invoker = tool_invoker
    else:

        async def intro_invoker(agent: CloversAgent, turn: Turn, event: Event):
            turn.create_payload(agent.api(intro), agent.style_prompt)
            turn.unit_prompts.append(content)
            return intro

        invoker = intro_invoker
    return invoker


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


def parse_skill(md: Path) -> SkillMD | None:
    with contextlib.suppress(Exception):
        skill = frontmatter.loads(md.read_text("utf-8"))
        name = skill.get("name")
        if not name or not isinstance(name, str):
            return
        desc = skill["description"]
        if not desc or not isinstance(desc, str):
            return
        if schema := skill.get("parameters"):
            parameters = schema["properties"]  # type: ignore
            required = schema.get("required")  # type: ignore
        else:
            parameters = None
            required = None
        content = skill.content.strip()
        return name.replace("-", "_"), desc, parameters, required, content
