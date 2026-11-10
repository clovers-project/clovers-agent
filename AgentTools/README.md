# Python 工具插件

本目录是核心默认扫描的本地插件目录。插件通过导出的 `TOOLS = AgentTools()` 注册工具、场景入口、分类及生命周期钩子。

默认 `plugin_dirs=["./AgentTools"]`，也可通过核心 `plugins` 加载已安装模块。模块由宿主导入，不要直接执行 `test.py`；它开头的包检查会阻止直接运行。

## 最小工具

```python
from clovers_agent import AgentTools, CloversAgent, Turn, Event

TOOLS = AgentTools()
TOOLS.create_category("demo", "需要原样回显文本时加载")

@TOOLS.register(
    "echo",
    "原样返回输入文本",
    {"text": {"type": "string", "description": "要回显的文本"}},
    category="demo",
)
async def echo(agent: CloversAgent, turn: Turn, event: Event, text: str) -> str:
    return text
```

模型先调用 `load_tools(category="demo")`，才会在本轮工具列表中获得 `echo`。

## 注册接口

```python
TOOLS.register(
    name,
    description,
    parameters=None,
    category=None,
    required=None,
)
```

| 参数          | 规则                                                                |
| ------------- | ------------------------------------------------------------------- |
| `name`        | 工具名，Python 注册不会自动把连字符转为下划线                       |
| `description` | 给模型看的工具说明，写清何时调用及输出                              |
| `parameters`  | 参数名到 JSON Schema 片段的映射，不是完整 object schema             |
| `category`    | 字符串或字符串集合；`None` 注册为 intro 场景入口                    |
| `required`    | 缺省为全部参数；显式 `[]` 表示无必填，Python 函数也应提供对应默认值 |

普通工具是 `async def`，接收 `agent, turn, event` 和模型生成的关键字参数，返回字符串。框架不会完整验证参数值类型；签名错误、执行异常、非字符串返回值分别转为工具错误信息。执行异常还会存入 `agent.tool_failures`，最多保留 5 条。

intro 与普通工具分别检查重名，分类描述也不能重复声明。合并冲突会失败；实际使用仍应避免跨入口和普通工具重名。多个插件可以向同一已有分类添加工具和钩子，但不要重复 `create_category()`。

## 分类与钩子

| 分类           | 用法                                                                 |
| -------------- | -------------------------------------------------------------------- |
| 普通分类       | 用 `create_category(name, description)` 声明，进入 `load_tools` 菜单 |
| `:intrinsic`   | 正常正式轮次自动注入                                                 |
| `:builtin`     | 核心按流程注入，例如图片回读与旁路补充                               |
| `:hidden`      | 不声明菜单描述，由实现显式注入；不是权限控制                         |
| intro 同名分类 | 对应场景被选中后自动注入                                             |

`create_category()` 返回分类钩子装饰器，也可以用 `@TOOLS.on_category(name)` 向已有分类挂钩子。钩子接收 `agent, turn, event`，返回需要注入的提示文本：

```python
@TOOLS.on_category("demo")
async def demo_prompt(agent, turn, event):
    return "输出保持与输入一致。"

@TOOLS.on_intro("on_chat")
async def chat_prompt(agent, turn, event):
    return "当前会话附加提示。"
```

同一入口或分类的多个钩子并发执行。intro 钩子结果进入本轮提示；分类钩子结果进入 `load_tools` 工具结果。仅创建描述而没有工具或钩子的空分类会在清单验证时被移除。

### 自定义场景

不指定 `category` 时注册为路由入口。路由不使用模型生成的入口参数，因此入口应能只用三个上下文参数执行。入口必须准备 payload，并返回实际场景名：

```python
@TOOLS.register("on_demo", "用户要求执行演示任务时使用")
async def on_demo(agent, turn, event):
    turn.create_payload(agent.api("on_demo"), agent.style_prompt)
    return "on_demo"
```

在 `apis.on_demo` 配置专用模型，未配置时使用默认模型。返回名用于选择场景分类工具和 intro 钩子，不一定必须与路由入口名相同。

## 生命周期与运行对象

```python
@TOOLS.on_startup
async def startup(agent):
    ...

@TOOLS.on_shutdown
async def shutdown(agent):
    ...
```

启动钩子返回的协程会被创建为后台任务，核心不会等它们全部完成再处理消息。关闭时取消未完成的启动任务，与关闭钩子一起最多等待 30 秒。

每日维护可向 `agent.daily_tasks` 添加接受 `agent` 的函数；在调度器时区每天 02:00 的核心清理之后运行，协程合计最多等待 30 秒。会话资源清理协程可登记在 `session.extra["cleanup"]`，参照工具集工作区实现。

| 对象    | 常用成员                                                                                                                                                  |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent` | `api(key)`、`current_session(id)`、`sessions`、`send()`、`tool_failures`、`extra`、`reload_skills()`、`sync_menu()`                                       |
| `turn`  | `payload`、`extend_tools()`、`upsert_tool()`、`extend_scripts()`、`extend_references()`、`pending_content`、`unit_prompts`、`session_id`、`create_turn()` |
| `event` | `send(type, data)`、`call("flat_context")`、`image_list`、`at`、`to_me`、`user_id`、`group_id`                                                            |

`pending_content` 用于工具返回后补入用户上下文；`unit_prompts` 用于轮次激活时的提示。后台调用链可通过 `submit_turn_for_wakeup()` 登记，详见 [核心文档](../clovers-agent/README.md)。

## 本目录的手动测试工具

[test.py](test.py) 的四个工具全部注册为 `:intrinsic`，加载本目录后会直接进入正式轮次的工具列表：

| 工具                   | 实际行为                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `test_daily_tasks`     | 将当前会话活动时间置零并执行全局 `_daily_task()`，影响过期会话、清理钩子和用量计数 |
| `test_compact_context` | 对当前会话尝试摘要压缩，返回“下一轮生效”                                           |
| `test_raise_exception` | 故意用字符串加整数触发 TypeError，验证失败记录                                     |
| `test_reload_skill`    | 打印重载前后清单，执行 `reload_skills()` 和 `sync_menu()`                          |

这些是开发工具，不是自动化测试。部署不需要时从 `plugin_dirs` 移除本目录，避免模型调用维护操作。

## 调试与开发约定

工具集 `debug_utils` 提供源码查看、失败详情与 Python 模块重载，默认关闭，见 [工具集文档](../clovers-agent-toolkit/README.md)。Skill 热重载后还需同步分类菜单；修改插件后可重启宿主。

工具在宿主进程内运行，与宿主共享权限。同一次模型响应中的工具会并发执行，共享状态应自行加锁。外部请求设置超时，阻塞任务放到线程，控制返回内容长度；框架没有独立的每工具超时，也不保证外部副作用回滚。
