# AgentSkills

**用 Markdown 给 `clovers-agent` 写工具。**
Authoring guide and examples for clovers-agent skills.

Skill 是「渐进披露」的工具包：分类只暴露一段描述，模型真正需要时才加载整个工具组、提示词、脚本与参考资料。相比 Python 工具插件，Skill 更适合**以提示词和脚本为主**、需要频繁调整文案的任务。

本目录下的 [`weather/`](weather/) 是可运行的模板，演示了分类、Markdown 工具、`skill.py`、脚本与引用文件的完整组合。

## 目录结构

```text
AgentSkills/
└── weather/
    ├── SKILL.md                 # 必需：分类定义与分类提示词
    ├── 1-weather-report.md      # 工具定义：front matter + 正文
    ├── 2-this-is-a-test.md      # 工具定义：正文是一段工作流
    ├── skill.py                 # 可选：同名函数实现工具
    ├── scripts/
    │   └── random_number.py     # 由 execute_script 执行
    └── references/
        └── task*.md             # 由 read_reference 读取
```

一个 Skill 是 `skill_dirs`（默认 `./AgentSkills`、`~/.agents/skills`）下的一个子目录；也可以直接是一个 `.md` 文件。

## SKILL.md 规范

```markdown
---
name: weather
description: 包含天气相关的工具。如查询天气等
---

此工具包为 Clovers Agent SKILL 模板，非生产环境使用。
若确认当前为测试场景，你应该调用 `this_is_a_test` 而非 `weather_report`。
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | 是 | 分类名 / intro 名；`-` 会转成 `_` |
| `description` | 是 | 分类描述，会出现在 `load_tools` 的菜单里 |
| 正文 | 否 | 分类钩子的提示词；在 `load_tools` 加载该分类时注入 |

加载规则：

1. 目录里**只有** `SKILL.md`（没有其他 `.md`、`scripts/`、`references/`）：注册为 intro 工具；没有同名 `skill.py` 函数时，正文在该 intro 被选中时作为系统提醒（`<system-reminder>`）注入
2. 目录里还有 `.md` / `scripts/` / `references/`：`SKILL.md` 的 `name` 成为分类名，正文成为分类钩子的返回值（有同名函数时作为 `content` 传入），其他 `.md` 注册为分类工具
3. 单个 `.md` 文件：注册为 intro 工具；没有同名 `skill.py` 函数时，正文在该 intro 被选中时作为系统提醒注入
4. 一个 Skill 独占一个分类，不能与其他 Skill 共用

## 工具 Markdown 规范

```markdown
---
name: weather-report
description: 查询天气预报信息
parameters:
  type: object
  properties:
    city:
      type: string
      description: 需要查询的城市
  required:
    - city
---

## {city}城市天气信息

时间: {now}
天气: {weather}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | 是 | 工具名；`-` 会转成 `_` |
| `description` | 是 | 给模型看的工具描述 |
| `parameters` | 否 | JSON Schema；读取 `properties` 与 `required` |
| 正文 | 否 | 传给 `skill.py` 同名函数的 `content`；没有同名函数时直接作为工具结果返回 |

要点：

- 正文不是「说明」，而是**工具的输出或模板**。分类内没有 `skill.py` 时，模型拿到的就是正文原文（占位符不会被替换）
- 工具名会归一化：`weather-report` 注册为 `weather_report`，正文里引用工具时建议写归一化后的名字
- `parameters.required` 缺省时，`register` 会把所有参数都标为必填；不需要参数的工具请省略 `parameters`
- 同一个目录内的工具名不能重复

## skill.py

`skill.py` 中与 Markdown 工具同名的函数会被接到该工具上。名称按归一化后的形式匹配：

| Markdown `name` | `skill.py` 函数 |
| --- | --- |
| `weather-report` | `weather_report` |
| `this-is-a-test` | `this_is_a_test` |

函数签名固定（`content` 必须是关键字参数）：

```python
from clovers_agent import cast_invoker, CloversAgent, Turn, Event

@cast_invoker
async def weather_report(
    agent: CloversAgent,
    turn: Turn,
    event: Event,
    content: str,
    city: str,
) -> str:
    ...
    return content.format(city=city, now=..., weather=...)
```

- 必须是 `async`，返回值会被包装成工具消息
- `content` 是该 Markdown 的正文（已 `strip()`），通常用它做模板
- 其他参数来自 `parameters`，类型由模型按 schema 生成
- `skill.py` 在加载 Skill 时被 `importlib` 执行；导入失败不会中断加载，工具会退回「直接返回正文」
- 没有同名函数时无需 `skill.py`

## scripts

`scripts/` 下的所有文件会被映射为 `<skill_name>/<相对路径>`，并在分类加载时作为 `execute_script` 的参数枚举注入：

```text
scripts/random_number.py  →  weather/random_number.py
```

模型调用：

```json
{"path": "weather/random_number.py", "interpreter": "python", "args": ["1", "50"], "timeout": 30}
```

- `interpreter` 为 `python` / `python3` 时使用宿主解释器；留空表示直接执行文件；其他值按命令名查找
- `timeout` 默认 120 秒，最大 1800 秒；超时会终止进程
- 只保留输出末尾约 1600 字符；回车覆盖的进度条会被清理

脚本适合做确定性、可复现的脏活（抓数据、转格式、跑计算），不要把需要模型判断的逻辑写进脚本。

## references

`references/` 下的文件同样映射为 `<skill_name>/<相对路径>`，由 `read_reference` 读取：

```text
references/task1.md  →  weather/task1.md
```

- 读取完整文件，编码由 chardet 探测
- 适合放长文档、检查清单、任务模板；不要让模型一次读入大量无关资料
- 引用文件只在对应分类被加载后可用

## 加载与重载

- `startup()` 时扫描 `skill_dirs`，逐个调用 `load_skill`
- `reload_skills()` 会清空已加载的 intro / 分类 / 脚本 / 引用映射，然后重新扫描
- 没有内置的聊天命令触发热重载；示例插件 [`AgentTools/test.py`](../AgentTools/test.py) 提供了 `test_reload_skill` 便于验证
- 加载失败通常**静默跳过**（例如 front matter 缺字段），排查时看启动日志里的 `[LOAD SKILL]`
- 重载时分类名 / 工具名冲突会抛 `ValueError`

## weather 示例

| 文件 | 作用 |
| --- | --- |
| `SKILL.md` | 定义分类 `weather`；正文说明这是模板，测试场景应走 `this_is_a_test` |
| `1-weather-report.md` | 工具 `weather_report`，参数 `city`；正文是带 `{city}` 等占位符的天气模板 |
| `2-this-is-a-test.md` | 工具 `this_is_a_test`；正文是一段工作流：查天气 → 跑脚本 → 读引用 → 汇总 |
| `skill.py` | `weather_report` 请求 `wttr.in` 的 JSON 接口，把实时数据渲染进模板 |
| `scripts/random_number.py` | 模拟带进度条 / 转圈动画的随机数脚本，演示回车覆盖输出的处理 |
| `references/task1.md` … | 10 个纯文本任务，供测试流程随机抽取 |

一次典型的测试流程：

1. 用户询问天气，模型调用 `load_tools("weather")`
2. 分类钩子注入 `SKILL.md` 正文，`weather_report` / `this_is_a_test` 变得可用
3. 模型调用 `weather_report(city="北京")`，`skill.py` 拉取数据并渲染模板
4. 若走测试分支：`this_is_a_test` 指示模型执行 `execute_script` 跑 `random_number.py`，再用 `read_reference` 读取随机任务并完成

注意事项：

- 示例依赖外网 `wttr.in`，离线环境会失败
- 脚本故意包含耗时动画（约 20 秒），用于验证输出清洗与超时设置，不是生产代码
- 正文里的工具名建议与注册名一致（`weather_report` / `this_is_a_test`），否则模型可能调用不存在的名字

## 编写建议

- **一个分类一件事**：分类描述写清楚「什么时候该加载」，比堆工具更重要
- **控制加载体积**：`SKILL.md` 正文会随 `load_tools` 进入工具消息，工具正文会进入工具结果，长文本请放 `references/`
- **工具名用下划线**：归一化规则容易让正文与注册名不一致
- **确定性逻辑放脚本**：返回结构化结果，让模型负责解释
- **路径不要写死**：脚本与引用都用映射路径，不要依赖宿主的当前目录
- **可重入**：脚本可能被重复执行，避免不可逆的副作用

## 常见问题

| 现象 | 原因 |
| --- | --- |
| Skill 没加载 | 缺 `SKILL.md`、front matter 缺 `name` / `description`、目录不在 `skill_dirs` |
| 分类不在菜单 | `SKILL.md` 缺 `description`，或加载时被静默跳过 |
| 工具调不到 | 工具名被归一化（`-` → `_`），或分类还没 `load_tools` |
| 脚本找不到 | `execute_script` 的 `path` 要写完整映射路径（`weather/random_number.py`） |
| 重载后重复 | 同名分类 / 工具已存在；检查是否手工复制了 Skill 目录 |

## 安全

`skill.py` 会在加载时执行，`scripts/` 会被模型按需执行。Skill 目录等价于可执行代码目录，只放可信内容；不要把密钥写进 Skill 或脚本，模型与日志都可能看到。

## License

MIT，见工作区根目录的说明。
