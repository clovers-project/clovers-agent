# AgentTools

**用 Python 给 `clovers-agent` 写工具插件。**
Python plugin guide and examples for clovers-agent.

`AgentTools` 是 `clovers-agent` 的 Python 工具注册表。相比 [Skill](../AgentSkills/README.md)（Markdown + 提示词驱动），Python 插件更适合需要**访问宿主对象、复用第三方库、做结构化处理**的能力。

本目录的 [`test.py`](test.py) 是自包含示例：四个测试工具分别覆盖每日维护、上下文压缩、异常记录与 Skill 热重载。

## 加载机制

- 默认扫描 `plugin_dirs`（`./AgentTools`）下的每个 `.py` 文件
- 也可以通过 `plugins` 配置加载已安装的模块，例如 `clovers_agent_toolkit`
- 模块必须导出 `TOOLS = AgentTools()`；Agent 启动时导入模块并 `merge` 其中的工具
- 模块在 `AgentTools` 命名空间下被导入（例如 `AgentTools.test`），因此**不要直接运行文件**；示例开头的 `if not __package__` 就是这道保护
- 工具名、intro 名、分类名冲突会在合并时抛 `ValueError`

## 最小示例

```python
from clovers_agent import AgentTools, CloversAgent, Turn, Event

TOOLS = AgentTools()
TOOLS.create_category("demo", "演示工具组")

@TOOLS.register(
    "echo",
    "原样返回输入文本",
    {"text": {"type": "string", "description": "要回显的文本"}},
    category="demo",
)
async def _(agent: CloversAgent, turn: Turn, event: Event, text: str) -> str:
    return text
```

## API

### `TOOLS.register`

```python
register(
    name: str,
    description: str,
    parameters: dict[str, JSONSchema] | None = None,
    category: str | Iterable[str] | None = None,
    required: list[str] | None = None,
)
```

| 参数 | 说明 |
| --- | --- |
| `name` | 工具名，需全局唯一 |
| `description` | 给模型看的描述，决定它会不会调用 |
| `parameters` | 参数名 → JSON Schema 片段（`type` / `description` / `enum` / `items` 等） |
| `category` | 分类名；`None` 表示注册为 intro（场景入口） |
| `required` | 必填参数；缺省等于 `parameters` 的全部键，`[]` 表示无必填 |

工具函数必须是 `async def`，第一个参数固定为 `agent`、`turn`、`event`，其余参数按 JSON Schema 从模型调用中注入：

```python
async def _(agent, turn, event, city: str, days: int = 1) -> str: ...
```

### 分类

| 写法 | 效果 |
| --- | --- |
| `category=None` | intro 工具，进入路由菜单 |
| `category="demo"` + `create_category("demo", "...")` | 普通分类，进入 `load_tools` 菜单，按需加载 |
| `category=":intrinsic"` | 每轮自动注入 |
| `category=":hidden"` | 随分类整体注入，但不出现在 `load_tools` 菜单 |

`create_category(name, description)` 会登记分类描述并返回一个钩子装饰器：

```python
register = TOOLS.create_category("demo", "演示工具组")

@register
async def _(agent, turn, event):
    return "进入 demo 工具组时的提示词"
```

分类只有登记了描述，模型才能在 `load_tools` 的枚举里看到它。

### 钩子

| 钩子 | 触发时机 | 返回 |
| --- | --- | --- |
| `@TOOLS.on_intro(name)` | 该 intro 被路由选中 | 追加到系统提示的文本 |
| `@TOOLS.on_category(name)` | `load_tools` 加载该分类 | 追加到工具结果的文本 |

多个插件可以给同一个 intro / 分类挂多个钩子，执行结果会合并。

### 生命周期

```python
@TOOLS.on_startup
async def _(agent: CloversAgent):
    ...

@TOOLS.on_shutdown
async def _(agent: CloversAgent):
    ...
```

- 启动钩子在 `agent.startup()` 中并发执行，协程以外的返回值会被忽略
- 关闭钩子最多等待 30 秒
- 每日维护：在启动钩子里 `agent.daily_tasks.append(func)`，`func(agent)` 会在每天 02:00 的维护任务中执行

### 可访问对象

| 对象 | 常用成员 |
| --- | --- |
| `agent` | `api(key)`、`current_session(id)`、`sessions`、`send`、`usage_counter`、`tool_failures`、`extra` |
| `turn` | `payload`、`extend_tools`、`upsert_tool`、`extend_scripts`、`extend_references`、`pending_content`、`unit_prompts`、`session_id` |
| `event` | `send(type, data)`、`call("flat_context")`、`image_list`、`at`、`to_me`、`user_id`、`group_id` |

### 返回值与异常

- 返回 `str`；内容会作为 tool message 回灌给模型
- 参数签名不匹配会被捕获为 `` `name`: invalid args ``，不会中断调用链
- 工具内部抛出的异常会记录到 `agent.tool_failures`，模型只看到 `` `name` failed ``
- 不要返回 `None` 或复杂对象；序列化由工具自己负责

## 示例：test.py

| 工具 | 作用 | 验证点 |
| --- | --- | --- |
| `test_daily_tasks` | 把当前会话的 `last_active_time` 置 0，然后执行 `_daily_task` | 会话清理、插件 `cleanup` 协程、用量结算 |
| `test_compact_context` | 对当前会话执行一次上下文压缩 | 摘要生成、`histories/*.md` 落盘、「下一轮生效」 |
| `test_raise_exception` | 故意触发 `TypeError` | 失败提示、`tool_failures` 记录、调用链不中断 |
| `test_reload_skill` | 打印重载前后的 manifest，再 `reload_skills()` + `sync_menu()` | 分类 / 工具 / 脚本映射是否正确重建 |

四个工具都注册在 `:intrinsic`，启动后直接在对话里调用即可。`test_reload_skill` 的打印输出在宿主终端，适合开发时对照 manifest。

> `test_daily_tasks` 会把当前会话从缓存中清除，下一次消息会重新创建会话；这是预期行为。

## 调试与热重载

- 失败记录：`agent.tool_failures`（最多 5 条）保存工具名、参数与 traceback；官方工具集的 `debug_utils` 可以把它们编号、展开并查看源码
- Python 模块热重载：官方工具集的 `reload_module` 会 detach 旧工具、重新导入模块、`merge` 新工具
- Skill 热重载：`agent.reload_skills()`（示例见 `test_reload_skill`）
- 修改插件文件后，若不使用 `reload_module`，需要重启宿主

## 最佳实践

- **异步**：不要阻塞事件循环；网络与磁盘 IO 用 `await`
- **超时**：外部请求自己设置超时，工具调用链没有单独的每工具超时
- **返回值可控**：长输出截断，二进制不要直接塞进上下文
- **路径安全**：涉及文件时校验在工作区内，拒绝 `..`
- **幂等**：模型可能重复调用，副作用要可重复执行
- **密钥**：不要从工具结果里回显密钥
- **错误信息**：模型只看到 `failed`，需要可诊断的细节请写日志

## 常见问题

| 现象 | 排查 |
| --- | --- |
| 工具没出现在列表 | 模块是否被加载、是否导出 `TOOLS`、分类是否登记了描述 |
| 启动报重名 | 工具 / intro / 分类与已加载插件冲突 |
| 改了代码不生效 | 没有热重载；使用 `reload_module` 或重启 |
| 模型不调用 | `description` 太模糊，或参数 schema 缺失 |
| 调用报 invalid args | 参数名 / 类型与函数签名不一致 |

## 安全

插件代码在宿主进程内执行，权限与宿主相同：文件、网络、Docker 都可达。只安装可信插件；不要在插件里硬编码密钥；对模型生成的路径、URL、命令做校验。

## License

MIT，见工作区根目录的说明。
