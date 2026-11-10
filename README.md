# Clovers Agent

Clovers 的异步群聊 Agent 插件，使用 OpenAI 兼容的 Chat Completions 接口。核心插件处理会话、场景路由、工具调用、图片与记忆；可选工具集提供文件工作区、联网搜索和调试能力。

本仓库是 uv workspace，需要由 Clovers 宿主加载，不提供独立机器人启动程序。当前源码版本：核心 `0.3.9`、工具集 `0.3.6`。

## 项目组成

| 目录                     | 内容                                        | 文档                                                |
| ------------------------ | ------------------------------------------- | --------------------------------------------------- |
| `clovers-agent/`         | 核心插件，导入名 `clovers_agent`            | [配置与运行机制](clovers-agent/README.md)           |
| `clovers-agent-toolkit/` | 可选工具集，导入名 `clovers_agent_toolkit`  | [工具清单与工作区](clovers-agent-toolkit/README.md) |
| `AgentTools/`            | 本地 Python 插件及开发测试工具              | [Python 插件开发](AgentTools/README.md)             |
| `AgentSkills/`           | Markdown Skill、脚本与参考资料示例          | [Skill 编写与示例状态](AgentSkills/README.md)       |
| `prompt-generator/`      | 独立提示词生成实验脚本，不属于 workspace 包 | [使用说明](prompt-generator/README.md)              |

## 从源码接入

### 1. 安装

要求 Python `>=3.12,<4.0`。在仓库根目录执行：

```bash
uv sync --all-packages
```

也可在已有 Clovers 宿主使用的 Python 环境中安装本地包：

```bash
python -m pip install ./clovers-agent
python -m pip install ./clovers-agent-toolkit
```

工具集可按需安装。核心依赖 Clovers `>=0.5.1`、clovers-client `>=0.3.2`、clovers-apscheduler `>=0.3.0`，并用 PyTorch / sentence-transformers 加载句向量模型。完整依赖以各包 `pyproject.toml` 为准。首次启动需要能获取所配置的句向量模型，或预先准备本地模型与缓存。

### 2. 配置

核心通过 `Config.sync_config("clovers_agent")` 接入宿主配置系统，配置存放位置由宿主决定。以下是配置内容的 TOML 示意，不是独立启动文件：

```toml
[clovers_agent]
plugins = ["clovers_agent_toolkit"]
plugin_dirs = []
skill_dirs = []

[clovers_agent.api]
url = "https://your-provider.example/v1"
model = "your-model"
api_key = "your-api-key"

[clovers_agent.check]
whitelist = ["your-group-id"]
```

`api` 必填，模型需要支持 function tool calling。`url` 填 API 根路径，程序会追加 `/chat/completions`。

默认 `check` 不接收任何消息。群聊可设置字符串群号白名单，本地联调可改用 `check.console_mode=true`。规则优先级为白名单 → 黑名单 → 控制台模式，白 / 黑名单模式均拒绝私聊。

上例清空本地示例加载目录。源码默认扫描 `./AgentTools`、`./AgentSkills` 和 `~/.agents/skills`，会加载维护测试工具与风格示例，部署时应明确选择需要的内容。相对路径以宿主启动时的工作目录为基准。

### 3. 启动宿主

让 Clovers 宿主加载 `clovers_agent` 模块。模块导出 `PLUGIN`、`AGENT`，由宿主调用启动和关闭钩子。工具集通过核心配置 `plugins` 加载，仅安装不会自动启用。

工具集默认启用 `workspace`、`network`。文件工具默认操作宿主工作区；需要 Docker 命令和子代理时设置工具集 `workspace.use_shell=true`。搜索另需 `network.deepseek_search_key`。

## 主要行为

- 群聊按 `group_id` 隔离会话，私聊使用 `private-<user_id>`。
- 定向消息进入正式轮次，多个场景由路由模型选择，路由失败回退普通聊天。
- 模型按需加载分类工具；同一次模型响应内的多个工具调用并发执行。
- 主任务运行时，新的定向消息可走旁路回复，或向主任务注入补充说明。
- 普通群聊消息先进入短期窗口；无其他 @、会话空闲且冷却结束时可主动插话。
- 完成轮次保存为 JSONL；达到长度、时间或话题切换条件时生成 Markdown 摘要。
- 工具集子代理异步执行，主调用链空闲且已登记任务完成后，通过唤醒轮次继续处理结果。

## 数据与扩展

默认数据目录为 `./data/CloversAgent/`，包括会话、历史摘要、用量、提示词覆盖与工具集工作区。句向量缓存另由 `sentence_model_cache` 指定。

`recorder.jsonl` 是可清理的上下文记录，不是永久聊天存档：每日维护会删除过期会话的该文件，压缩也会删除它并留下摘要。图片映射仅保存在内存，重启后旧图片标签不能恢复。

人设与风格可通过配置或数据目录的 `prompts/*.md` 调整。访问宿主对象、复用第三方库用 Python 插件；提示词工作流、脚本和资料用 Skill。详见分包文档。

## 开发与打包

```bash
uv build --all
```

根目录 `build.bat` 执行同一命令。`AgentTools/test.py` 是需要宿主和模型参与的手动测试工具，不是可直接运行的自动化测试套件。

## License

核心与工具集均采用 MIT，见 [核心 LICENSE](clovers-agent/LICENSE) 和 [工具集 LICENSE](clovers-agent-toolkit/LICENSE)。
