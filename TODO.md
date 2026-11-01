# clovers_agent 项目 TODO

> 快照：2026-11-01 21:45（Asia/Shanghai）
> 版本：`pyproject.toml` 0.3.3 / `clovers_agent/__init__.py` 0.3.0（不一致）
> 规模：约 2.9k 行 Python（10 个顶层模块 + `typing/` 的 4 个类型定义）
> 排序原则：先能跑 → 再可信 → 再好维护 → 最后加功能。本清单基于静态审阅（当前环境依赖未安装，未执行运行时验证）。

## 一、现状盘点

**定位**：clovers 的群聊 Agent 插件。事件进入后按会话加锁，经语义路由选择 intro（场景），注入分类工具后进入最多 40 步的工具调用循环，边生成边分片发送，并在轮末落盘记录；另有主动插话、旁路修正、上下文压缩和唤醒（wakeup）链路。

**已具备，不用重复造**：

- 插件入口与生命周期：`main.py` 的 `PLUGIN`/`AGENT`，白名单 / 黑名单 / 控制台三种准入
- OpenAI 兼容调用：重试与退避、`retry-after`、vision 混合调用（`api.py`）
- 会话状态机：`Turn` / `Session` / `WakeupContext`，JSONL 持久化、日切清理、执行锁与旁路锁
- 对话策略：router 路由、chime-in 主动插话、aside 旁路修正、compact 摘要压缩
- Skill 体系：`SKILL.md` frontmatter、`scripts/`、`references/`、`execute_script` / `read_reference`
- 工具注册：分类、合并 / 卸载、manifest 校验（`core.py`）
- 配置模型（pydantic）与每日维护任务

**关键缺口**：

| 领域 | 现状 |
| --- | --- |
| 文档 | `README.md` 为空（0 字节），无架构、配置、Skill 格式说明 |
| 测试 | 无 `tests/`、无 `conftest.py`、无 CI |
| 版本 | 0.3.3 与 `__version__` 0.3.0 不一致；无 CHANGELOG |
| 环境 | 工作区无 venv、无锁文件；默认 `python`（msys 3.14）未安装任何项目依赖 |
| 仓库 | 未 `git init`；无 `.gitignore`；`__pycache__` 混在源码树中 |
| 静态质量 | 无 ruff / mypy 配置，无 pre-commit |

## 二、P0 — 发布 0.3.4 前必须完成

### P0-1 修复 `execute_script` 超时参数（S）

- 位置：`clovers_agent/agent.py:841`
- 现象：判断条件是 `isinstance(timeout, float)`，而模型传的 JSON 数字通常是 int（如 `600`），会被直接丢弃并回落到 120s；默认值本身也是 int。
- 动作：统一转 `float` 后 `min(max(timeout, 1), EXECUTE_SCRIPT_MAX_TIMEOUT)`；在返回文本里回显实际生效的超时值。
- 验收：手工冒烟覆盖 int / float / 超上限 / 负数 / 缺省，随后在 P1-1 固化为单测。

### P0-2 修复脚本与引用的路径查找（S）

- 位置：`clovers_agent/agent.py:822`、`clovers_agent/agent.py:857`
- 现象：`scripts_map[...][path]`、`references_map[...][path]` 在 `try` 之外，非法 category/path 直接抛 `KeyError`，模型只能看到 `` `execute_script` failed ``，没有可纠正的信息。
- 动作：改为 `.get(category, {}).get(path)`，缺失时返回明确的错误文本；同时校验 `path` 必须是映射中的键。
- 验收：非法路径返回可读错误，不产生 traceback，模型可据此重试（手工冒烟即可，单测随 P1-1）。

### P0-3 工具返回值规范化（S）

- 位置：`clovers_agent/agent.py:299`
- 现象：`content` 原样写入 tool message，插件返回 `None` / dict 时违反 `ToolMessage` 类型；OpenAI 兼容网关通常要求 content 为字符串，可能直接返回 400。
- 动作：非 str 统一 `json.dumps` 或 `str()`，`None` 映射为空串；超过阈值截断并注明。
- 验收：注册一个返回 `None` 的测试工具，调用链不中断。

### P0-4 修复 `reload_skills` 的状态登记（M）

- 位置：`clovers_agent/core.py:520`（`_loaded_intros` 与 `_loaded_categories` 在 427-428 初始化，仅 501 行有过 `discard`，从未写入）
- 现象：`reload_skills()` 清不掉已加载的 intro / category；第二次调用会撞上 `Intro already exists` 或 `Category already exists`。热重载（以及未来的 `/reload skills`）不可用。
- 动作：在 `load_skill_md` / `load_skill` 成功处登记；重载时先 `detach` hooks 再 `remove_category`；单条 skill 加载失败要回滚并记日志。
- 验收：连续调用两次 `reload_skills()` 不抛错，工具总数不变；新增一个 skill 后重载可见。

### P0-5 版本号与发布元数据（S）

- 位置：`pyproject.toml:3`、`clovers_agent/__init__.py:6`
- 现象：版本号有两处来源且已漂移；`description` 为空，缺 `license` / `urls` / `classifiers` / `keywords`。
- 动作：单一版本源（hatch `dynamic = ["version"]` 指向 `__init__.__version__`），补齐元数据，发布前校验 tag 与版本一致。

### P0-6 README 与最小上手路径（M）

- 位置：`README.md`（0 字节）
- 内容建议：定位与特性 → 安装（uv）→ 最小 `config.toml`（api / vision / path）→ 以 clovers 插件方式启动 → 目录结构 → 一个最小 Skill 示例 → 已知限制 → License。
- 验收：新机器照文档 15 分钟内能跑通控制台模式。

### P0-7 开发环境可复现（M）

- 现象：无 venv、无锁文件；`python` 是 msys 3.14，`clovers` / `pydantic` / `torch` 均未安装；`py -0p` 里没有 3.12（3.15 为默认）。已确认 `uv 0.10.12` 可用。
- 动作：用 uv 固定 3.12 并 `uv sync`；新增 `[project.optional-dependencies] dev = ["pytest", "pytest-asyncio", "respx", "ruff", "mypy", "pip-audit"]`；提交 `uv.lock`。
- 验收：干净目录 `uv sync && python -c "import clovers_agent"` 通过（模型下载可延迟或 mock）。

### P0-8 仓库卫生（S）

- 现象：不是 git 仓库；无 `.gitignore`；`clovers_agent/__pycache__/skill.cpython-312.pyc` 是已删除模块的残留（`skill.py` 已并入 `core.py`）。
- 动作：`git init`；`.gitignore` 覆盖 `__pycache__/`、`*.py[cod]`、`.venv/`、`data/`、`*.log`、`.env`；清理 pycache 后首次提交。
- 验收：`git status` 干净，仓库内无 pyc 与运行数据。

## 三、P1 — 工程质量基建

- [ ] **P1-1 测试骨架与首批用例（M）**：`tests/` + `pytest.ini`（`asyncio_mode=auto`）。优先覆盖纯函数：`parse_b64uri`（分支很多）、`format_message`、`deep_add`、`to_b64url`、`build_payload`、`Turn.extend_tools` 去重、`AgentTools.merge/detach`、`snapshot/resume`、`should_summarize`。再用 `respx` mock `/chat/completions` 跑通「工具调用 → 二次请求 → 最终回复」以及异常落盘分支。目标：`core` / `session` / `utils` 覆盖率 ≥ 60%。
- [ ] **P1-2 CI（S）**：GitHub Actions，矩阵 3.12 / 3.13，依次 `ruff check`、`ruff format --check`、`mypy`、`pytest -q`，可选 `pip-audit`。
- [ ] **P1-3 lint 与类型配置（S）**：ruff 选 `E,F,I,UP,B,SIM,RUF`；mypy 从 `core` / `session` / `utils` 渐进收紧，先排除 `embedding`（torch 类型噪音）。
- [ ] **P1-4 日志与故障可观测（M）**：统一 `[agent][session][turn]` 字段；工具调用记录耗时与结果长度；`tool_failures` 目前只在内存里且无出口，补「最近 N 条失败」查询或落盘；把 `load_module_from_path`（`core.py:565` 裸 `except`）和 `parse_skill`（`core.py:571` 静默 `suppress`）改为带原因的 warning。
- [ ] **P1-5 后台任务集中管理（S）**：`agent.py:176` 与 `main.py:23` 的 `asyncio.create_task` 未持有引用，异常被吞且任务可能被 GC；收敛到 `self._background_tasks`，用 `done_callback` 记录异常并清理。
- [ ] **P1-6 请求层健壮性（S）**：`api.py:33` 取 `choices[0]` 前校验非空；记录 `finish_reason == "length"`；`usage` 缺失不报错；`build_payload`（`api.py:27`）避免 `extra_body` 覆盖 `model` / `messages`；修 `pretty_payload` 的 `prefix + content or ""` 优先级 bug（`api.py:116`）。
- [ ] **P1-7 数据一致性（M）**：`save_jsonl` 无 fsync；`load_jsonl` 遇一行坏数据就丢弃整个文件，改为逐行容错并备份；失败快照会无限增长（`agent.py:337`），补保留上限与清理；`Session.cleanup` 删除 `recorder.jsonl`（`session.py:374`）需确认是否符合「记忆超时」预期。

## 四、P2 — 架构与可维护性

- [ ] **P2-1 拆分 `agent.py`（860 行）**：建议拆为 `runtime`（startup/shutdown/scheduler/usage）、`pipeline`（input → route → intro → execute）、`tools`（image / prompt / compact / script / reference）、`prompts`。`CloversAgent` 只做装配。
- [ ] **P2-2 拆分 `core.py`（586 行）**：`AgentTools`、`AgentSkills`、Markdown 解析、模块加载各自独立，便于插件作者只依赖工具基类。
- [ ] **P2-3 `clovers_agent/typing` 改名**：与标准库 `typing` 同名，在包目录内直接执行脚本或把包目录加入 `sys.path` 时会解析错；改为 `_typing` / `schemas`。
- [ ] **P2-4 去掉 import 期副作用**：`CONFIG = Config.sync_config(...)`（`config.py:262`）与 `AGENT` / httpx client 的模块级构造（`main.py:38-40`）改为工厂或懒加载，否则测试无法多实例、导入即写盘。
- [ ] **P2-5 收敛 `Event` 契约**：`core.Event` 用类属性 `extra_context: list[str] = []`（`core.py:40-41`）声明默认值，而代码同时依赖 `event.properties`；需确认 `clovers_client` 的属性访问方式，并保证值不在协议类上共享。
- [ ] **P2-6 清理死代码**：`AgentTools.detach`（`core.py:343`，无调用）、`pretty_payload`（`api.py:102`，无调用）、`HIDDEN_CATEGORY`（`config.py:139`，无引用）、`main.py:43` 的 `check` 占位、`_route` 中 `session.api` / `session.payload` 两条空语句（`agent.py:453-454`）。
- [ ] **P2-7 明确工具分类暴露规则**：`sync_menu`（`agent.py:158-165`）只枚举 `category_descriptions`，因此 `:builtin` / `:intrinsic` 以及未建 description 的插件分类都不会出现在 `load_tools` 的 enum 里。要么支持 hidden 标记，要么自动补描述并加校验。
- [ ] **P2-8 会话路径安全**：`session_id` 直接拼目录（`agent.py:244`、`agent.py:260`），而它来自平台群号/用户 id。加字符白名单，`resolve()` 后校验仍在 `session_dir` 下。
- [ ] **P2-9 可变默认值**：`Event.extra_context = []`、`OpenAIConfig.extra_body = {}`（`config.py:18`）改用 `default_factory` / 只读类型。

## 五、P3 — 功能与体验 backlog

- [ ] 命令面：`/reload prompts|skills`、`/usage [today|month]`、`/compact`、`/clear`、`/skill list`（运营闭环，配合 P0-4）
- [ ] 流式输出：SSE 分片发送，降低首字延迟（牵动 `send` 与发送锁）
- [ ] 成本与预算：按 model 汇总 token / 费用，日/月上限与告警
- [ ] 观测指标：调用延迟、失败率、工具 Top N、活跃会话数
- [ ] 视觉降级：vision 失败时显式提示或重试；图片体积/格式前置校验
- [ ] 路由策略：可关闭、可强制 `on_chat`、低置信度回退，路由结果统计
- [ ] 部署：Dockerfile（CPU torch 镜像较大，建议拆分 base / extras）、systemd unit、示例配置
- [ ] 文档：架构与时序图、Skill 编写指南、AgentTools 插件指南、配置项表、FAQ
- [ ] 依赖瘦身：`torch` / `sentence-transformers` 目前是硬依赖（`pyproject.toml:18-19`），可改为可选 extra，或提供远程 embedding 兜底
- [ ] 国际化：prompts 中英混排，抽语言包或明确只支持中文

## 六、安全专项

- [ ] **`execute_script` 的信任边界**（`agent.py:804`）：解释器任意、继承完整 env、cwd 不受限、无 CPU/内存/网络限制，等价于把命令执行交给模型。至少加：解释器白名单、cwd 限定在 skill 目录、env 白名单、超时与输出上限、调用审计；文档明确风险与开关。
- [ ] **`file://` 图片读取**（`api.py:53`、`utils.py:143`）：限制到允许目录。
- [ ] **密钥管理**：`api_key` 明文写在配置里，支持环境变量覆盖；`.gitignore` 排除 `data/`。
- [ ] **失败快照脱敏**：`payload_*.json` 含完整用户内容，补保留天数 / 容量上限 / 脱敏。
- [ ] **依赖审计**：`pip-audit` 纳入 CI；torch 固定小版本范围。

## 七、里程碑建议

| 版本 | 范围 | 预估 |
| --- | --- | --- |
| 0.3.4 | P0-1 ~ P0-8（脚本修复 + 文档 + 环境 + 仓库） | 1-2 天 |
| 0.4.0 | P1 全部 + P2-1 / P2-3 / P2-4 / P2-6 | 1-2 周 |
| 0.5.0 | 命令面、部署、文档、安全加固 | 2-3 周 |

## 八、本周执行顺序

1. P0-8 建仓库 + P0-7 建环境（先让工程能跑）
2. P0-1 / P0-2 / P0-3（脚本工具三连，手工冒烟）
3. P1-1 测试骨架（锁住上面的修复）
4. P0-4 / P0-5 / P0-6（热重载、版本、README）
5. 打 0.3.4 tag

## 九、需要拍板的问题

1. 最低 Python 版本是否维持 3.12？本机没有 3.12，CI 要覆盖 3.12 / 3.13 还是只从 3.13 起？
2. `:builtin` 工具组是否要暴露给 `load_tools`，还是明确定位为 hidden？
3. `torch` / `sentence-transformers` 是否可改为可选依赖（影响安装体积与首启时间）？
4. wakeup（`submit_turn_for_wakeup`）的并发语义：同一会话多次提交应合并还是排队？需要用测试固化。
5. 回复是否需要流式，部署形态是长期群聊插件还是也要 Web 入口？
6. `execute_script` 的定位：只跑可信 skill，还是允许模型自由执行？这决定安全投入的上限。
