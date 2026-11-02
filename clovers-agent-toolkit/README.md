# clovers-agent-toolkit

**`clovers-agent` 的官方工具集。**
Official tool bundle for [clovers-agent](../clovers-agent/README.md): a Docker workspace, web search / extraction, and debugging utilities.

`clovers-agent` 本身只提供「聊天 + 工具循环 + Skill」的骨架，具体能力由工具包补齐。本包提供三组可按需启用的工具：

| 分类          | 作用                                                                 | 默认 |
| ------------- | -------------------------------------------------------------------- | ---- |
| `workspace`   | 每个会话独立的 Docker 沙箱、文件读写、命令执行、文件上传、异步子代理 | 启用 |
| `network`     | DeepSeek 网页搜索、网页正文提取（trafilatura）                       | 启用 |
| `debug_utils` | 工具失败排查、源码查看、模块热重载                                   | 关闭 |

## 安装

```bash
pip install clovers-agent-toolkit
```

需要 `clovers-agent >= 0.3.3`。从本工作区安装时，`uv sync --all-packages` 会以 editable 方式同时装上两者。

`workspace` 分类中的 `execute_command` / `subagent` 依赖 **Docker**：宿主进程需要能访问 Docker daemon（本包通过 `docker.from_env()` 连接）。仅使用文件工具时不需要 Docker。

## 启用与配置

配置由 `Config.sync_config("clovers_agent_toolkit")` 托管，字段名即配置键。

| 键               | 默认值                     | 说明                                   |
| ---------------- | -------------------------- | -------------------------------------- |
| `use_categories` | `["workspace", "network"]` | 启用的分类；不在列表里的模块不会被导入 |
| `network`        | 见下                       | 网络工具配置                           |
| `workspace`      | 见下                       | 沙箱与文件工具配置                     |

`use_categories` 的合法值：`workspace`、`network`、`debug_utils`。`debug_utils` 会在导入时打印警告，仅建议开发环境启用。

### network

| 键                    | 默认值                                     | 说明                                  |
| --------------------- | ------------------------------------------ | ------------------------------------- |
| `deepseek_search_key` | `sk-xxx`                                   | DeepSeek Search API Key，**必须替换** |
| `deepseek_search_url` | `https://api.deepseek.com/v1/tools/search` | 搜索接口地址                          |

### workspace

| 键                         | 默认值                                       | 说明                                                    |
| -------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| `use_shell`                | `false`                                      | 是否启用 Docker shell（`execute_command` / `subagent`） |
| `max_subagent_count`       | `8`                                          | 单条调用链最多创建的子代理数                            |
| `max_download_return_size` | `4096`                                       | 超过该字节数的下载会落盘，只把路径返回给模型            |
| `docker_image`             | `nikolaik/python-nodejs:python3.12-nodejs20` | 沙箱镜像，需自带 `bash` / `python` / `node`             |
| `shared_workspace`         | `false`                                      | 所有会话共用一个工作区与容器                            |

> `use_shell=false` 时仍可使用 `ls` / `read_file` / `write_file` / `fetch_url` / `upload_file` / `read_image`，只是不能执行命令或派生子代理。

## 工具清单

### workspace

| 工具              | 参数                                               | 说明                                                                                                    |
| ----------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `on_task`         | —                                                  | 任务场景入口（intro）。初始化工作区，注入 workspace / network 工具，加载工作区 `README.md` 作为环境说明 |
| `read_image`      | `path`                                             | 读取 URL 或工作区内的图片，加入下一轮请求（配合 `api.vision` 或原生多模态模型）                         |
| `upload_file`     | `file_path`                                        | 把工作区文件发送给用户；依赖宿主支持文件消息                                                            |
| `execute_command` | `command`                                          | 在沙箱内执行 shell 命令（`bash -c`，工作目录 `/workspace`）                                             |
| `subagent`        | `prompt`                                           | 异步子代理，结果在下一条调用链后通过唤醒消息返回                                                        |
| `fetch_url`       | `method`、`url`、`headers?`、`data?`、`file_path?` | GET / POST；可把响应直接写入工作区；大响应自动落盘                                                      |
| `ls`              | `path?`、`depth?`                                  | 列出工作区目录，默认深度 1，最多返回 200 项                                                             |
| `read_file`       | `file_path`                                        | 读取工作区文本文件（自动检测编码，二进制文件会被拒绝）                                                  |
| `write_file`      | `file_path`、`content`                             | 写入文本文件；路径越界时会改用随机文件名写入工作区                                                      |

`execute_command`、`subagent`、`fetch_url`、`ls`、`read_file`、`write_file` 注册在 `:hidden` 分类：不出现在 `load_tools` 的菜单中，由 `on_task` / `enter_workspace` 根据 shell 是否可用显式注入。

### network

| 工具            | 参数                                                                                                                                  | 说明                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `web_search`    | `query`（必填）、`question?`、`sites?`、`exclude_sites?`、`exclude_terms?`、`exact_phrases?`、`exclude_phrases?`、`after?`、`before?` | 调用 DeepSeek 搜索接口，把站点 / 时间 / 短语过滤拼进查询，返回前几条结果摘要 |
| `web_extractor` | `url`                                                                                                                                 | 用 trafilatura 把网页正文提取为 Markdown                                     |

两者注册在 `:intrinsic` 分类，每一轮都可用。

### debug_utils

| 工具                 | 参数           | 说明                                                 |
| -------------------- | -------------- | ---------------------------------------------------- |
| `list_tool_failures` | —              | 把尚未编号的工具失败记录编号并列出（最多保留 10 条） |
| `get_tool_failure`   | `failure_id`   | 查看某次失败的参数、源码与完整 traceback             |
| `get_tool_source`    | `tool_name`    | 读取已注册工具的源码                                 |
| `reload_module`      | `module_name?` | 卸载并重新导入指定工具模块，见下                     |

分类钩子还会把「当前调试目标」与调试守则注入上下文，提醒模型：工具报错时先如实告知用户，不要擅自重载。

## workspace 详解

### 目录与容器

```text
{clovers_agent.path}/workspace/
├── <session_id>/          # 每个会话一个目录，挂载到容器 /workspace
└── shared/                # shared_workspace=true 时共用
```

- 容器名：`CloversAgentSandbox-<session_id>`；容器启动命令为 `sleep infinity`，空闲时不占 CPU
- 镜像内的 `/workspace` 绑定到上面的会话目录（读写）
- 命令通过 Docker exec 执行，`workdir=/workspace`，并用 `< /dev/null` 避免命令等待 stdin
- 会话被每日清理时，注册的 `cleanup` 协程会删除容器；共享模式下清理协程会挂在多个会话上，需避免重复删除（见 TODO）

### 进入任务场景

`on_task` 会：

1. 创建会话工作区，并优先读取其中的 `README.md`（否则用工作区根目录的 `README.md`）作为环境说明
2. 注入 `workspace` 与 `network` 两个分类的工具
3. 调用 `enter_workspace`：shell 可用时注入 `execute_command` / `subagent`，否则回退到文件工具，并提示当前目录 `/workspace`

`load_tools("workspace")` 也会触发同一个钩子，因此模型也可以从普通聊天里按需进入工作区。

### 路径规则

- 绝对路径 `/workspace/...` 会映射为工作区内的相对路径
- `../` 越界访问：`read_file` / `upload_file` / `ls` 会直接拒绝；`write_file` 会改用 `<原名>.<随机后缀>` 写入工作区
- 读取大文件没有截断，`fetch_url` 默认把超过 `max_download_return_size` 的响应落盘，避免把大段二进制塞进上下文

### 子代理与唤醒

`subagent` 会创建独立 `Turn`，继承 `:intrinsic`、`workspace`、`network` 工具，并移除 `subagent` 自身防止递归。任务通过 `agent.submit_turn_for_wakeup` 异步执行；当前调用链结束后，结果以唤醒消息回到会话，因此**不要在派生子代理后原地等待**。

同一调用链内的子代理数量受 `max_subagent_count` 限制；名额只在当前调用链结束后释放。

### 文件上传

`upload_file` 调用 `event.send("file", file)`。部分平台适配器不支持文件消息，此时会返回「File upload is not available in this environment.」。

## network 详解

### web_search

- 把 `sites` / `exclude_sites` / `exclude_terms` / `exact_phrases` / `exclude_phrases` / `after` / `before` 拼成搜索操作符
- 固定请求 10 条，按 `rerank_score` 排序；前 3 条必返，第 4 条起低于均值则截断
- 返回 Markdown 摘要（标题、来源、日期、摘要），不返回完整网页
- 需要有效的 `deepseek_search_key`，否则接口会返回鉴权错误

### web_extractor

- 带浏览器 UA 请求页面，由 trafilatura 提取正文
- 输出 Markdown（含链接、图片、评论），失败时返回 `none`
- 不处理需要登录或 JS 渲染的页面

## debug_utils 详解

工具失败时，`clovers-agent` 会把 `{name, args, traceback}` 放进 `agent.tool_failures`（长度 5 的 deque）。本分类把它搬进独立的失败日志（最多 10 条，带自增编号）：

1. `list_tool_failures` 收集并编号，返回按时间从新到旧的列表
2. `get_tool_failure(id)` 返回参数、源码与完整 traceback
3. `get_tool_source(name)` 直接读源码，便于对照修改
4. `reload_module(name)` 卸载旧模块、清理 `sys.modules`、重新导入并 `merge` 新 `TOOLS`，最后 `sync_menu()`

`reload_module` 的约束：

- 第一次调用必须提供模块名（例如 `clovers_agent_toolkit.t_workspace.main`），后续可省略
- 失败时会恢复旧的 `sys.modules` 条目并重新合并旧工具，但源码改动不会生效，需要修复后再次重载
- 提示词要求它单独成轮：同一轮里不要和其他工具并行调用

> `reload_module` 只重载 Python 模块，不会重载 Markdown Skill。Skill 请用 `agent.reload_skills()`（示例见 `AgentTools/test.py` 的 `test_reload_skill`）。

## 安全

- **Docker 权限**：能访问 daemon 的人等价于 root。请把本包运行在受控宿主上，不要暴露 Docker socket
- **命令执行**：`execute_command` 让模型在容器内执行任意命令；容器内仍挂载了会话工作区，不要在其中放宿主密钥
- **SSRF**：`fetch_url` / `web_extractor` / `web_search` 可访问任意 URL 或内网地址，请按部署环境加网络隔离
- **数据外带**：`upload_file` 能把工作区文件发回聊天，注意工作区内容与权限
- **共享工作区**：`shared_workspace=true` 会让不同会话看到同一批文件，多租户场景不要开启
- **密钥**：`deepseek_search_key` 明文存在配置中，注意文件权限

## 已知限制

- 文件工具面向文本；二进制请用 `fetch_url` 落盘或 `upload_file` 传出
- 子代理结果只在主调用链结束后送达，不适合需要即时答案的场景

## License

MIT，见 [LICENSE](LICENSE)。
