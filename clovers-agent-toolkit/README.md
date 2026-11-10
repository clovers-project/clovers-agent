# clovers-agent-toolkit

Clovers Agent 的可选工具集，当前源码版本 `0.3.6`。提供工作区、联网搜索与网页正文提取、开发调试。

## 启用

先安装到核心所在的 Python 环境，再在核心 `clovers_agent` 配置中加入：

```json
{ "plugins": ["clovers_agent_toolkit"] }
```

工具集独立使用 `Config.sync_config("clovers_agent_toolkit")` 同步配置，字段定义见 [config.py](clovers_agent_toolkit/config.py)。安装包不等于加载插件。

默认启用 `workspace`、`network`；`debug_utils` 默认关闭。`use_categories` 是模块加载开关，不代表所有同名分类都能用 `load_tools` 加载：network 工具实际注册在 `:intrinsic`。

## 配置

| 字段                                 | 默认值                                       | 说明                                           |
| ------------------------------------ | -------------------------------------------- | ---------------------------------------------- |
| `use_categories`                     | `["workspace", "network"]`                   | 需要导入的模块，可选加 `debug_utils`           |
| `network.deepseek_search_key`        | `sk-xxx`                                     | 搜索鉴权，使用前填写有效密钥                   |
| `network.deepseek_search_url`        | `https://api.deepseek.com/v1/tools/search`   | 搜索接口                                       |
| `workspace.use_shell`                | `false`                                      | 尝试启用 Docker shell，失败时回退文件工具      |
| `workspace.max_subagent_count`       | `8`                                          | 当前唤醒上下文可登记的后台任务数量上限         |
| `workspace.max_download_return_size` | `4096`                                       | `fetch_url` 直接返回内容的字节上限，超出则落盘 |
| `workspace.docker_image`             | `nikolaik/python-nodejs:python3.12-nodejs20` | 容器镜像                                       |
| `workspace.shared_workspace`         | `false`                                      | 是否让会话共用 `workspace/shared` 和 shell     |
| `workspace.exec_default_timeout`     | `120`                                        | 命令默认超时秒数                               |
| `workspace.exec_max_timeout`         | `1800`                                       | 命令超时上限；实际请求值至少 5 秒              |

Docker 仅在启用 shell 时需要。需让宿主进程能连接 Docker daemon，并能获得所配置镜像；代码从环境构造 Docker 客户端。默认文件模式不需要 Docker。

## 工具如何进入轮次

- `on_workspace` 是路由入口。被选中时使用核心 `apis.on_task`（缺失则回退默认模型），准备工作区并注入工具。
- 普通聊天也可调用 `load_tools(category="workspace")` 激活工作区分类。
- `read_image`、`web_search`、`web_extractor` 注册在 `:intrinsic`，在正常正式轮次中自动可用，前提是启用了对应模块。
- `:hidden` 只是不出现在分类菜单，不是权限隔离；工作区初始化按模式显式注入其中的工具。

| 模式                                    | 工作区额外注入的工具                         |
| --------------------------------------- | -------------------------------------------- |
| `use_shell=false`，或 Docker 初始化失败 | `fetch_url`、`ls`、`read_file`、`write_file` |
| `use_shell=true` 且初始化成功           | `execute_command`、`subagent`                |

两种模式均有工作区分类的 `upload_file`。shell 模式下可用命令完成文件操作，但不会自动同时注入上述四个回退工具。

## 工作区与文件

宿主目录是核心配置 `{path}/workspace/<session_id>/`；共享模式使用 `{path}/workspace/shared/`。Docker 将该目录以读写方式挂载为 `/workspace`，命令也在此目录执行。

`on_workspace` 优先读取会话工作区的 `README.md` 作为提示，缺失时读取 `{path}/workspace/README.md`。这一步属于场景入口，单纯 `load_tools("workspace")` 不执行 README 读取。

文件工具接受相对路径或 `/workspace/...` 形式。当前路径处理只转换前缀并用 `is_relative_to()` 做词法检查，没有完整解析 `..` 或符号链接，不能把它当作严格的宿主文件沙箱。

| 工具          | 参数                                                 | 行为                                                                                                           |
| ------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `ls`          | 可选 `path="./"`、`depth=1`                          | 列直接子项或递归列表，目录优先，显示类型 / 大小 / 时间；输出最多 200 项，不递归跟随符号链接和 Windows junction |
| `read_file`   | `file_path`                                          | 读取文本并探测编码；包含 NUL 或无法解码时返回不支持；没有文本长度截断                                          |
| `write_file`  | `file_path`、`content`                               | UTF-8 写入完整文本，覆盖同名文件，创建父目录                                                                   |
| `fetch_url`   | `method`、`url`；可选 `headers`、`data`、`save_path` | GET 的 `data` 为查询参数，POST 为 JSON；注意参数名是 `save_path`                                               |
| `upload_file` | `file_path`                                          | 通过 `event.send("file", Path)` 把文件发送给用户，依赖平台适配器支持                                           |
| `read_image`  | `path`                                               | 从 HTTP(S) 或工作区读取图片，追加到 `pending_content`，下次模型请求时查看                                      |

`fetch_url` 有 `save_path` 时保存原始字节；未指定且大于字节阈值时自动保存为内容 MD5 命名的 `.bin`，否则解码后直接返回。它不会把所有非 2xx 响应都判为工具失败，错误响应体也可能返回或落盘，应结合返回的 HTTP 状态判断。

写文件或下载时，若词法检查判定目标不在工作区，程序会改为工作区内带 UUID 的文件名；读取、列目录、上传和本地读图则返回错误。该处理不弥补前述路径校验限制。

## Docker 命令与子代理

### execute_command

参数为必填 `command`、可选 `timeout`。使用容器内 `bash -c` 执行，外层套 `timeout -k 5`，标准输入重定向自 `/dev/null`。

命令之间保留工作区文件，但每次是新的 shell，不应依赖上一次 `cd` 或环境变量。返回命令输出文本，当前没有输出长度截断，也没有单独返回结构化退出码。容器名为 `CloversAgentSandbox-<session_id>`，已有同名容器会被复用或启动。

### subagent

参数 `prompt` 应包含完整任务、文件路径、约束及预期结果。子代理使用独立上下文，继承调用者 API，与父代理共享会话工作区；初始化时提供自动工具、工作区工具及对应 shell / 文件能力，再移除子代理工具，避免直接递归委派。

调用立即返回 `OK`，结果不在本次工具返回值中。核心在主调用链空闲且已登记任务完成后启动唤醒轮次。数量限制按当前唤醒上下文的任务列表计算，不是全局并发池，也不是每完成一个子任务就立即归还名额。

过期会话清理会调用已登记的 shell 清理协程删除容器，工作区文件保留。共享模式使用同一目录和 shell，各会话可以看到相同文件；某个相关会话触发清理也可能移除共享容器。

## 联网工具

### web_search

必填 `query`，可选 `question`、`sites`、`exclude_sites`、`exclude_terms`、`exact_phrases`、`exclude_phrases`、`after`、`before`。过滤项拼成搜索操作符，`question` 缺省使用当前事件文本。

向配置的搜索接口请求 10 条，按 `rerank_score` 降序，最多展示 6 条；结果足够时前 3 条保留，第 4 条起低于全部结果均分则截断。返回标题、来源、日期及摘要，不是完整网页正文。

### web_extractor

参数 `url`。HTTP 下载后由 trafilatura 提取 Markdown，保留链接、图片和评论，提取不到正文返回 `none`。没有浏览器 JS 渲染或登录流程。两个网络工具均设置 30 秒请求超时；工具集请求重试最多 5 次，针对网络异常、429 和 5xx。

## 调试工具

在工具集 `use_categories` 中加入 `debug_utils`，再由模型加载该分类。源码会在启用时输出开发调试警告。

| 工具                 | 参数               | 用途                                                        |
| -------------------- | ------------------ | ----------------------------------------------------------- |
| `list_tool_failures` | 无                 | 收集核心最近的工具异常、分配编号、按新到旧列出              |
| `get_tool_failure`   | `failure_id`       | 展示调用参数、当前工具源码及 traceback                      |
| `get_tool_source`    | `tool_name`        | 读取当前已注册普通工具的源码                                |
| `reload_module`      | 可选 `module_name` | detach 旧注册表、移除模块缓存、重新导入并合并，最后同步菜单 |

核心异常队列最多 5 条，收集后清空；调试记录最多 10 条，仅存内存。读取失败详情前先列出编号，展示的源码是当前版本，不是失败时快照。

首次重载需提供目标模块名，后续可沿用；选择实际加载并拥有工具注册表的插件入口，如 `AgentTools.test` 或 `clovers_agent_toolkit`。工具集子模块共享同一 `TOOLS`，不适合把某个内部 `main` 模块当成完整插件独立重载。导入失败会恢复旧模块缓存并合并旧工具，但不保证回滚任意模块顶层副作用。重载应单独成轮、避开并行调用。

Markdown Skill 重载使用核心 `agent.reload_skills()`，随后 `agent.sync_menu()`，不由 `reload_module` 完成。

## License

MIT，见 [LICENSE](LICENSE)。
