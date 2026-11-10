# Skill 编写与示例

Skill 把场景提示、工具正文、Python 实现、脚本和参考资料组织到一起，由核心按需加载。加载器实现见 [core.py](../clovers-agent/clovers_agent/core.py)。需要直接使用宿主对象注册工具时，也可采用 [Python 插件](../AgentTools/README.md)。

## 发现与加载

核心默认 `skill_dirs=["./AgentSkills", "~/.agents/skills"]`。每个路径是 Skill 根目录，加载器遍历它的直接子项，子项可以是 `.md` 文件或带 `SKILL.md` 的目录；不是任意层级递归寻找 Skill。`~` 会展开，相对路径按宿主工作目录解释。

| 形式                                                                   | 注册结果                                               |
| ---------------------------------------------------------------------- | ------------------------------------------------------ |
| 根目录下单个有效 `.md`                                                 | intro 场景入口，没有配套 Python 自动匹配               |
| 子目录只有有效 `SKILL.md`，无其他有效工具 Markdown、脚本文件和引用文件 | intro，可绑定该目录 `skill.py` 中的同名函数            |
| 子目录还有有效工具 Markdown，或 `scripts/`、`references/` 内有文件     | 分类，`SKILL.md` 定义分类，其他有效顶层 `.md` 定义工具 |

是否存在附加资源按实际文件判断：空的 `scripts/` / `references/` 目录不会使 intro 变成分类。工具 Markdown 只扫描 Skill 目录顶层；脚本和引用文件递归扫描。

分类描述进入 `load_tools` 菜单，加载后注入工具、分类提示，以及脚本 / 引用参数枚举。intro 则进入场景路由菜单，不通过 `load_tools` 激活。

## Markdown 格式

文件使用 UTF-8 和 YAML front matter。`name`、`description` 必须是非空字符串，名称中的 `-` 转为 `_`。缺字段、解析失败的文件通常静默跳过；正文或 Python 实现均为空的工具不会注册。

### 分类定义 `SKILL.md`

```markdown
---
name: demo
description: 用户需要演示工具时加载
---

先确认用户需要的输出，再调用本分类工具。
```

每个 Skill 分类需要独占名称。目录作为分类时，正文成为分类钩子的返回文本；若 `skill.py` 有与分类名匹配的函数，正文作为 `content` 传入该函数。空正文且没有函数的分类仍需有效工具等内容支撑，不应只创建空分类。

### 分类工具

在同目录创建 `echo.md`：

```markdown
---
name: demo-echo
description: 将文本交给示例处理函数
parameters:
  type: object
  properties:
    text:
      type: string
      description: 输入文本
  required:
    - text
---

收到的文本：{text}
```

`parameters` 是完整 object schema，加载器读取其中的 `properties` 和 `required`。这与 Python `TOOLS.register()` 直接接收 properties 映射不同。省略 `required` 时所有参数被标为必填；`required: []` 表示无必填，函数也要有相应默认值。

正文不会自动格式化。没有 Python 实现的分类工具直接返回正文，且包装函数不接收工具参数；这类纯文本工作流工具应省略 `parameters`。需要参数或模板替换时提供同名 Python 函数。

### `skill.py`

为上面的 `demo-echo` 配置归一化后的 `demo_echo` 函数：

```python
from clovers_agent import cast_invoker, CloversAgent, Turn, Event

@cast_invoker
async def demo_echo(
    agent: CloversAgent,
    turn: Turn,
    event: Event,
    content: str,
    text: str,
) -> str:
    return content.format(text=text)
```

加载器通过关键字 `content=正文` 传入模板，因此形参不能设为 positional-only，也不要求语法上必须为 keyword-only。`cast_invoker` 是类型标注辅助，运行时直接返回原函数，不负责参数验证或格式化。

Markdown 正文为空时，函数直接作为实现，不会注入 `content`。`skill.py` 在发现目录时执行；导入异常被吞掉，可能退回正文实现，应结合工具实际返回值排查，不能只凭“加载成功”认定 Python 已生效。

### intro 的特殊要求

纯 Markdown intro 会建立使用 `agent.api(归一化名称)` 和 `agent.style_prompt` 的 payload，将正文加入本轮提示，返回自身名称。可以使用 `apis.<名称>` 配置模型。

自定义 Python intro 则负责 `turn.create_payload()`，返回实际场景名，参照 [on-erotic/skill.py](on-erotic/skill.py)。路由忽略模型产生的入口参数，入口函数应能仅凭上下文及可选注入的 `content` 执行，不能要求用户参数。

## 脚本与引用

```text
AgentSkills/
└── demo/
    ├── SKILL.md
    ├── echo.md
    ├── skill.py
    ├── scripts/
    │   └── calculate.py
    └── references/
        └── guide.md
```

脚本映射为 `demo/calculate.py`，引用映射为 `demo/guide.md`；映射名不含 `scripts/`、`references/` 前缀。分类加载后才会在本轮枚举中显示。

脚本调用示意：

```json
{ "path": "demo/calculate.py", "interpreter": "python", "args": ["1", "10"], "timeout": 30 }
```

- `execute_script` 在宿主运行，不在工具集 Docker 容器中。`python` / `python3` 使用宿主当前解释器，空字符串表示直接执行文件，其他值作为解释器命令。
- 不设置脚本工作目录，脚本需要自行按自身路径定位资源。
- 默认超时 120 秒，上限 1800 秒，超时尝试终止进程。
- 返回退出码以及 stdout / stderr，各保留末尾约 1600 字符，并处理回车覆盖的进度输出。
- `read_reference(path)` 读取完整文件并探测编码，没有内置长度截断。

Skill Python 与脚本拥有宿主权限，只应加载可信内容。参数枚举是给模型的路径提示，不是独立的执行权限边界。

## 仓库示例的实际状态

| 目录         | 注册名称与状态                                                                    |
| ------------ | --------------------------------------------------------------------------------- |
| `weather/`   | 分类 `weather`，包含真实天气请求、测试工作流、脚本和 10 个参考任务                |
| `on-erotic/` | intro `on_erotic`，由 Python 实现设置场景风格与模型，并返回普通聊天场景 `on_chat` |

`weather` 文件对应关系：

| 文件                                | 作用                                                            |
| ----------------------------------- | --------------------------------------------------------------- |
| `SKILL.md`                          | 分类定义，提示测试场景选择测试工作流                            |
| `1-weather-report.md`               | 工具 `weather_report`，参数 `city`，正文为天气模板              |
| `skill.py`                          | 请求 `wttr.in`，解析 JSON 并格式化天气模板，依赖外网            |
| `2-this-is-a-test.md`               | 工具 `this_is_a_test`，返回查天气、运行脚本、读取任务的流程正文 |
| `scripts/random_number.py`          | 随机数与进度输出演示                                            |
| `references/task1.md` … `task10.md` | 测试任务资料                                                    |

调用顺序示意：`load_tools(category="weather")` → `weather_report(city="北京")`，或调用 `this_is_a_test` 后按工作流执行脚本与引用读取。脚本路径为 `weather/random_number.py`，参考路径如 `weather/task1.md`。

示例工作流正文仍使用部分原始连字符名称和相对文件名，实际模型工具名与路径应以上述归一化名称和枚举为准。天气与动画脚本用于演示，不应把示例当作经过生产验证的业务功能。

## 重载与排查

启动时加载 Skill。开发中可调用 `agent.reload_skills()`，随后 `agent.sync_menu()` 更新分类菜单；本目录没有内置聊天重载命令，开发插件 [test.py](../AgentTools/test.py) 提供 `test_reload_skill`。

重载会移除追踪到的 Skill 入口和分类，清空脚本 / 引用映射，再重新扫描；这不是原子事务，重名或加载异常可能中断过程。应避免在活动调用链依赖旧工具时进行重载。

| 现象                       | 排查                                                    |
| -------------------------- | ------------------------------------------------------- |
| Skill 不出现               | 根目录配置、直接子项布局、UTF-8、front matter 必填字段  |
| 名称看起来正确但调不到     | `-` 已转成 `_`，或分类尚未加载                          |
| 只返回模板、占位符原样保留 | `skill.py` 导入失败、函数名不匹配，或没有实现模板格式化 |
| 参数报 invalid args        | 纯正文工具配置了参数，或函数签名与 schema 不一致        |
| 脚本 / 引用找不到          | 使用完整枚举路径，且先加载对应分类                      |
| 重载冲突                   | 分类独占要求、归一化后名称重复、加载目录重叠            |
