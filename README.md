# Clovers Agent

**Clovers 的异步群聊 Agent 插件与官方工具集。**
An async group-chat agent plugin for [clovers](https://clovers-project.github.io/), powered by OpenAI-compatible APIs.

本仓库是一个 [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)，包含核心 Agent 插件、官方工具集，以及 Skill / Tool 两份可直接照抄的示例。它本身不是机器人，需要由 clovers 宿主加载。

## 组成

| 目录                                                        | 包                      | 说明                                                               |
| ----------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------ |
| [`clovers-agent/`](clovers-agent/README.md)                 | `clovers-agent`         | 核心插件：会话、语义路由、工具循环、Skill 加载、记忆压缩           |
| [`clovers-agent-toolkit/`](clovers-agent-toolkit/README.md) | `clovers-agent-toolkit` | 官方工具集：`workspace` / `network` / `debug_utils`                |
| [`AgentSkills/`](AgentSkills/README.md)                     | —                       | 示例 Skill：`weather`（天气查询 + 脚本 / 引用演示）                |
| [`AgentTools/`](AgentTools/README.md)                       | —                       | 示例 Python 工具插件：`test.py`（日常任务 / 压缩 / 异常 / 热重载） |
| [`TODO.md`](TODO.md)                                        | —                       | 当前工作区的待办与优先级                                           |

## 能做什么

- **群聊理解**：区分「跟我说话」和「群友闲聊」，前者进入正式回复，后者可主动插话（chime-in）
- **语义路由**：注册了多个场景（intro）时，先用轻量模型选择场景，再加载该场景的工具
- **工具循环**：单轮最多 `max_call_depth` 步工具调用，支持并行执行、中途注入用户补充说明
- **Skill 系统**：`SKILL.md` + Markdown 工具 + 可选 `skill.py` + `scripts/` / `references/`
- **记忆管理**：滑动窗口、话题解耦、超长时自动摘要压缩为 Markdown 历史
- **旁路修正**：任务执行中收到新消息，可即时补充或纠正，而不打断主任务
- **多模型接入**：默认 / 路由 / 插话 / 摘要 / 任务可分别指向不同模型
- **视觉输入**：纯文本模型可搭配 `vision` 模型，把图片转写为文字描述
- **子代理与唤醒**：官方工具集提供异步 `subagent`，结果在下一条调用链后以唤醒消息送达
- **运维**：每日清理过期会话、按日统计 token 用量、失败调用的现场快照

## 快速开始

### 1. 环境

- Python **3.12+**
- 一个可用的 OpenAI 兼容接入点（`url` / `model` / `api_key`）
- [uv](https://docs.astral.sh/uv/)（推荐）
- Docker（仅当需要官方工具集的 `workspace` 命令 / 子代理能力时需要）

### 2. 安装

开发工作区：

```bash
uv sync --all-packages
```

部署到已有的 clovers 宿主环境：

```bash
pip install ./clovers-agent
pip install ./clovers-agent-toolkit   # 可选：workspace / network / debug 工具
```

### 3. 配置

`clovers_agent` 的配置由 `clovers_client.Config.sync_config` 托管，首次运行会生成默认配置。最小必填项：

| 键            | 示例                        | 说明         |
| ------------- | --------------------------- | ------------ |
| `api.url`     | `https://api.openai.com/v1` | 接入点根地址 |
| `api.model`   | `gpt-4o-mini`               | 模型名       |
| `api.api_key` | `sk-...`                    | API Key      |

本地联调时把 `check.console_mode` 设为 `true`，即可跳过群聊白 / 黑名单限制。完整字段与默认值见 [`clovers-agent/clovers_agent/config.py`](clovers-agent/clovers_agent/config.py)。

### 4. 启动

由 clovers 宿主加载 `clovers_agent` 插件（包内导出 `PLUGIN` 与 `AGENT`），启动宿主即可。日志出现 `[CLOVERS_AGENT] 启动控制台模式` 表示插件已接管事件。

## 目录结构

```text
.
├── clovers-agent/                 # 核心 Agent 插件
│   ├── clovers_agent/
│   │   ├── agent.py               # 运行时与工具循环
│   │   ├── core.py                # 工具注册 / Skill 加载
│   │   ├── session.py             # 会话、轮次、压缩、唤醒
│   │   ├── api.py                 # OpenAI 兼容客户端
│   │   ├── config.py              # 配置模型与默认提示词
│   │   └── ...
│   └── pyproject.toml
├── clovers-agent-toolkit/         # 官方工具集
│   └── clovers_agent_toolkit/
│       ├── t_workspace/           # Docker 沙箱、文件、子代理
│       ├── t_network/             # 搜索与网页正文提取
│       └── t_debug/               # 失败排查与模块热重载
├── AgentSkills/weather/           # 示例 Skill
├── AgentTools/test.py             # 示例工具插件
├── TODO.md
└── pyproject.toml                 # uv workspace 定义
```

## 文档

| 文档                                                     | 内容                                                    |
| -------------------------------------------------------- | ------------------------------------------------------- |
| [clovers-agent](clovers-agent/README.md)                 | 配置项、会话模型、工具与 Skill 机制、数据目录、二次开发 |
| [clovers-agent-toolkit](clovers-agent-toolkit/README.md) | 官方工具清单、workspace 沙箱、网络工具、调试工具        |
| [AgentSkills](AgentSkills/README.md)                     | Skill 目录规范与 `weather` 示例拆解                     |
| [AgentTools](AgentTools/README.md)                       | Python 工具插件规范与 `test.py` 示例                    |
| [TODO](TODO.md)                                          | 未完成事项、优先级与已知问题                            |

## 版本与兼容性

| 项                    | 要求                          |
| --------------------- | ----------------------------- |
| Python                | `>=3.12,<4.0`                 |
| `clovers`             | `>=0.5.1`                     |
| `clovers-client`      | `>=0.3.2`                     |
| `clovers-apscheduler` | `>=0.3.0`                     |
| Docker                | 可选，仅 `workspace` 工具需要 |

## 安全提示

`clovers-agent` 会执行 Skill 携带的 Python（`skill.py`）与脚本（`scripts/`），官方工具集的 `workspace` 还能在 Docker 容器内执行 shell 命令。请只加载可信来源的 Skill / Tool，并阅读各包 README 的「安全」章节。

## License

MIT。详见 [`clovers-agent/LICENSE`](clovers-agent/LICENSE) 与 [`clovers-agent-toolkit/LICENSE`](clovers-agent-toolkit/LICENSE)。
