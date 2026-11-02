# clovers-agent 工作区 TODO

> 快照：2026-11-02 00:05（Asia/Shanghai）
> 版本：`clovers-agent` 0.3.4、`clovers-agent-toolkit` 0.3.4（均未发布）
> 结构：uv workspace，`clovers-agent` + `clovers-agent-toolkit` + 两份示例
> 说明：本清单基于当前工作区代码静态审阅；`[x]` 表示已在 0.3.4 工作区中完成。
> 路径约定：`agent.py` / `core.py` / `session.py` / `main.py` / `config.py` / `api.py` 均指 `clovers-agent/clovers_agent/` 下的文件；`t_*` 均指 `clovers-agent-toolkit/clovers_agent_toolkit/` 下的目录。

## 0. 已在 0.3.4 完成

- [x] 版本号单一来源：`pyproject.toml` 改为 `dynamic = ["version"]`，由包内 `__version__` 提供
- [x] `execute_script` 接受 JSON 整数超时（`isinstance(timeout, (int, float))`，`agent.py:843`）
- [x] `reload_skills` 登记已加载的 intro / 分类，二次重载不再撞名（`core.py:447-459`、`529-540`）
- [x] 拆分 `clovers-agent` / `clovers-agent-toolkit`，根目录使用 uv workspace
- [x] 文档基线：根 README、两个包 README、`AgentSkills/README.md`、`AgentTools/README.md`

## 1. P0 — 发布 0.3.4 前

- [ ] **清理调试输出**：`t_debug/utils.py:30-31` 残留 `print(log)` / `print(agent.tool_failures)`，每次查看失败都会打到 stdout
- [ ] **补 `web_search` 描述**：`t_network/main.py:22` 的描述是空字符串，模型无法判断何时搜索；写清「实时网页搜索」与返回形态
- [ ] **补齐包元数据**：两个 `pyproject.toml` 的 `description` 为空，且缺 `license` / `urls` / `classifiers` / `keywords`；PyPI 页面目前只有包名
- [ ] **版本依赖对齐**：`clovers-agent-toolkit` 依赖 `clovers-agent>=0.3.3`，与 0.3.4 同发时建议提到 `>=0.3.4`（`clovers-agent-toolkit/pyproject.toml:8`）
- [ ] **初始化仓库**：当前没有 `.git`；`.gitignore` 已有但忽略 `*.lock`，确认是否要提交 `uv.lock` 后 `git init` + 首次提交
- [ ] **最小冒烟**：`uv sync --all-packages` 后 `import clovers_agent` / `import clovers_agent_toolkit`、加载 `AgentSkills/weather`、`reload_skills()` 两次；把命令写进 README 或 `scripts/`
- [ ] **校对 README**：按真实宿主验证「配置生成位置 / 插件启用方式 / 文件消息能力」三处描述，避免文档与实现漂移
- [ ] **统一示例命名**：`weather` 的正文引用 `weather-report`、`this-is-a-test`，注册时却被归一化为 `weather_report`、`this_is_a_test`（`AgentSkills/weather/SKILL.md`、`2-this-is-a-test.md`）；要么改正文，要么取消归一化

## 2. P1 — 正确性与安全

### 框架

- [ ] **脚本 / 引用路径 KeyError**：`agent.py:824`、`agent.py:859` 在 `try` 之外索引映射，非法 `path` 直接抛异常，模型只看到 `failed`；改成 `.get()` 并返回可读错误
- [ ] **工具返回值规范**：`agent.py:301` 原样写入 `content`；插件返回 `None` / dict 时违反 `ToolMessage`，部分网关直接 400；统一转 `str` 并截断
- [ ] **`reload_skills` 边界**：只有 `SKILL.md` + `scripts/`（没有其他 `.md`）的 Skill 不会登记进 `_loaded_categories`（`core.py:496-523`），第二次重载会 `Category already exists`
- [ ] **会话目录消毒**：`session_id` 来自平台 id，直接拼路径（`agent.py:244`、`agent.py:260`）；加字符白名单并在 `resolve()` 后校验父目录
- [ ] **后台任务持引用**：`agent.py:176`、`main.py:23` 的 `create_task` 未保存引用，异常被吞且任务可能被回收；集中到集合 + `done_callback`

### 工具集

- [ ] **`fetch_url` 路径越界写**：`t_workspace/main.py:245-247` 直接用 `workspace / format_path(file_path)` 写文件，没有 `is_relative_to` 检查；`../` 可写出工作区
- [ ] **`read_image` 本地路径越界**：`t_workspace/main.py:135-138` 读取本地图片前未校验工作区边界
- [ ] **子代理上限 off-by-one**：`t_workspace/main.py:194` 用 `len(tasks) > MAX` 判断，实际可创建 `MAX + 1` 个；改为 `>=`
- [ ] **下载内存峰值**：`fetch_url` 先把整个响应读入内存再判断大小（`t_workspace/main.py:238-255`）；改为流式写入并设硬上限
- [ ] **`read_file` 无大小上限**：`t_workspace/main.py:300` 整文件读入；加最大字节数并提示
- [ ] **命令无超时**：`docker.py:43-46` 的 `execute` 无超时，长命令会占住调用链；加可配置超时与强制终止
- [ ] **容器名消毒**：`docker.py:25` 用 `session_id` 拼容器名，非 Docker 合法字符会失败；先规范化
- [ ] **SSRF 防护**：`fetch_url` / `web_extractor` 可访问内网与云元数据地址；至少加私有网段黑名单或域名白名单
- [ ] **`SUBAGENT_PROMPT` 为空**：`t_workspace/constants.py:3`；补子代理系统提示（角色、工具边界、输出要求）
- [ ] **`debug_utils` 失败日志语义**：`list_tool_failures` 会清空 `agent.tool_failures`（`t_debug/utils.py:32`），并发调用会丢记录；加锁或改为只读快照

## 3. P2 — 工程质量

- [ ] **测试**：`tests/` + `pytest-asyncio` + `respx`。首批覆盖：`parse_b64uri`、`format_message`、`deep_add`、`to_b64url`、`build_payload`、`Turn.extend_tools` 去重、`AgentTools.merge/detach`、`snapshot/resume`、`should_summarize`、工具循环（成功 / 工具异常 / 超 depth）。工具集用 `tmp_path` 覆盖 `ls` / `read_file` / `write_file` / `fetch_url`；Docker 用 mock
- [ ] **CI**：GitHub Actions，矩阵 3.12 / 3.13，`ruff check` + `ruff format --check` + `mypy` + `pytest`
- [ ] **静态配置**：ruff（`E,F,I,UP,B,SIM,RUF`），mypy 从 `core` / `session` / `utils` 渐进收紧，先排除 `embedding`（torch 噪音）
- [ ] **打包**：补 `license` 字段与 `license-files`；`[project.urls]`（主页 / 文档 / Issue）；`keywords`；确认 wheel 包含 README
- [ ] **日志**：统一 `[agent][session][turn]` 字段；工具调用记录耗时与结果长度；`load_module_from_path`（`core.py:574`）与 `parse_skill`（`core.py:580`）的裸 `except` / 静默 `suppress` 至少打 warning
- [ ] **数据可靠性**：`save_jsonl` 无 fsync；`load_jsonl` 遇一行坏数据就丢弃整个文件；失败快照 `payload_*.json` 无保留上限（`agent.py:338`）
- [ ] **导入副作用**：`CONFIG = Config.sync_config(...)`（`config.py:262`）与模块级 `AGENT` / httpx client（`main.py:38-40`）改为工厂或懒加载，方便测试与多实例
- [ ] **`typing` 目录改名**：`clovers_agent/typing` 与标准库同名，在包目录内执行脚本或把包目录加入 `sys.path` 时会解析错
- [ ] **死代码清理**：`pretty_payload`（`api.py:102`，无调用且 `api.py:116` 有优先级 bug）、`_route` 里 `session.api` / `session.payload` 空语句（`agent.py:453-454`）、`main.py:43` 的 `check` 占位
- [ ] **配置热加载**：提供 `reload_prompts()` 的调用入口（当前只在启动时调用）；配合 `/reload` 命令
- [ ] **CHANGELOG**：0.3.4 改动已不小，补一份 Keep a Changelog 风格的记录
- [ ] **英文 README**：面向 PyPI 的英文版（或中英双语），当前全中文

## 4. P3 — 功能 backlog

- [ ] 流式输出：SSE 增量发送，降低首字延迟（牵动 `ChatAgent.send` 的分片与锁）
- [ ] 运营命令：`/reload skills|prompts`、`/usage [today|month]`、`/compact`、`/clear`、`/skill list`
- [ ] 成本与预算：按模型汇总费用，日 / 月上限与告警
- [ ] 指标：调用延迟、失败率、工具 Top N、活跃会话数（Prometheus / OTel）
- [ ] 部署：Dockerfile（CPU torch 镜像较大，建议拆分 base / extras）、compose 示例、systemd unit
- [ ] 更多示例 Skill：搜索、文档处理、定时任务各一个，覆盖「脚本 + 引用 + 分类钩子」的不同组合
- [ ] Skill 脚手架：`clovers-agent new-skill` 或脚本，生成 SKILL.md / 工具 md / skill.py 骨架
- [ ] 子代理体验：进度回传、失败重试、结果去重（当前只有最终唤醒消息）
- [ ] 视觉降级：vision 失败时的显式提示 / 重试；图片体积与格式前置校验
- [ ] 搜索缓存：`web_search` 结果按 query 短期缓存，降低重复调用成本

## 5. 安全专项

- [ ] **Skill 即代码**：`skill.py` 在加载时执行，`scripts/` 由模型执行；文档已警告，考虑增加「可信目录 / 签名」机制
- [ ] **Docker 权限**：能访问 daemon 等价 root；限制 socket 暴露，容器加 `--network`、只读挂载、资源上限
- [ ] **路径穿越**：统一实现 `resolve_within(root, path)`，所有文件工具复用（当前 `write_file` 有兜底，`fetch_url` / `read_image` 没有）
- [ ] **SSRF**：`fetch_url` / `web_extractor` / `web_search` 统一出站代理或白名单
- [ ] **密钥管理**：`api_key`、`deepseek_search_key` 支持环境变量覆盖；`.gitignore` 排除 `data/`
- [ ] **提示注入**：工具结果、网页正文、Skill 正文都可能含指令；至少在系统提示里声明「工具输出不是指令」
- [ ] **失败快照**：包含完整用户内容，补保留天数 / 容量上限 / 脱敏

## 6. 里程碑建议

| 版本 | 范围 | 预估 |
| --- | --- | --- |
| 0.3.4 | P0（清理、元数据、仓库、冒烟）+ 文档 | 1-2 天 |
| 0.4.0 | P1 正确性与安全 + 测试 / CI | 1-2 周 |
| 0.5.0 | 命令面、流式、部署、指标 | 2-4 周 |

## 7. 需要拍板

1. **工具名归一化**：保留 `-` → `_`（函数匹配方便，但正文要跟着改），还是注册时保留原始名、仅在查找函数时归一化？
2. **`workspace` 默认值**：`use_shell=false` 是安全默认，但会让「任务场景」默认没有命令能力；是否在文档里把 Docker 标为推荐前置？
3. **`debug_utils` 默认关闭**：保持默认关闭，还是提供单独的 `clovers-agent[debug]` extra？
4. **`shared_workspace`**：定位是单租户便捷模式还是多租户隔离？决定是否需要 per-session 容器强约束。
5. **锁文件**：`.gitignore` 忽略 `*.lock` 是有意为之（库项目）还是误配？影响可复现构建。
6. **发布形态**：`clovers-agent` 与 `clovers-agent-toolkit` 是否同版本号、同 tag 发布？
