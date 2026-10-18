# clovers-agent 优化 TODO

> 审计范围：`clovers-agent/` 全部 14 个 Python 文件（约 1,950 行）＋ `pyproject.toml` / `LICENSE` / `README.md`。
> 已核实：全部源码可在 Python 3.14 下编译通过；抽查的 9 个依赖版本（numpy/pydantic/torch/sentence-transformers/clovers/clovers-client/clovers-apscheduler/chardet/puremagic）在 PyPI 均存在；项目当前**不是 git 仓库**、**没有任何测试 / CI / lint 配置**、`README.md` 为 0 字节。
> 未运行：`clovers` 系列依赖未安装，动态行为（路由、插件加载）未实测，相关结论来自静态阅读，标注了"需确认"。

图例：`S` ≈ 半小时内，`M` ≈ 半天，`L` ≈ 1 天以上。

---

## P0 — 正确性与安全（先修，直接影响可用性）

- [ ] **5. `read_reference` 无大小上限** `clovers_agent/core.py:505-509` `S`
      直接返回整份文件，模型可能读到一个几 MB 的日志/数据文件。
      修法：限制返回字节数（如 64KB）并附"已截断"提示；明确只允许 `references/` 下的文件（当前靠 map key 约束，尚可）。

- [ ] **6. 持久化非原子 + 启动加载无容错** `clovers_agent/utils.py:41-44`、`clovers_agent/core.py:74-79,206-211,364-365` `S`
      `save_json` 直接覆盖写，进程被 kill 时会留下半截 JSON，下次启动 `json.load` 直接让整个插件导入失败；`recorder.json` 同理。
      修法：`save_json` 写临时文件 + `os.replace`；加载侧捕获 `JSONDecodeError` 并备份坏文件；`usage_counter` 初始化失败时降级为空字典。

- [ ] **7. 事件处理器没有兜底异常** `clovers_agent/main.py:37-46` `S`
      `AGENT.chat(event)` 任何未捕获异常都会冒泡到宿主框架，用户侧表现为"机器人没反应"且没有日志关联。
      修法：包一层 `try/except Exception`，`logger.exception` + 返回一条简短错误提示；同时保证 `update_usage` 在 `finally` 中执行（见 #18）。

- [ ] **8. `router` 的短路条件恒为假，每条消息都多一次 LLM 调用** `clovers_agent/core.py:254-255` `S`（需确认设计意图）
      `len(self.intro_tools) < 2`，而 `ON_CHAT` 与 `SKILL_MENU` 在 `core.py:89-90` 必然注册为 intro，因此该条件永远不成立；即使没有任何插件工具，每条消息仍会先跑一次 router 模型。
      修法：改为统计"非内置 intro"数量，或去掉短路条件并在文档说明 router 始终执行；若确实需要为图片回看保留 `builtin` 分类，请显式注释原因。

- [ ] **9. 视觉模型重复识别同一批图片，且失败时污染 payload** `clovers_agent/api.py:88-109` `M`
      `HybridOpenAIAPI.call_api` 每轮（每次工具调用后）都会对"最后一条 user 消息"重新调用视觉模型——多轮工具调用会重复计费；另外 `message["content"]` 只在成功返回后恢复，`super().call_api` 抛错时 payload 会残留纯文本版本。
      修法：把图片描述缓存到 `Session`（按图片 URL/`image_id` 缓存），同一张图只描述一次；用 `try/finally` 恢复原始 content；缓存放在 message 的临时字段而非原地改写。

- [ ] **10. 空回复会原样重试最多 40 次** `clovers_agent/core.py:241-252` `S`
      `call_turn` 在 `content` 为空时 `continue`，但既没有把 assistant 消息写回 payload，也没有改变请求 → 重复发送完全相同的请求，最多 `call_depth=40` 次，纯烧 token。
      修法：记录一次空回复后 `break`/返回兜底文案；或把空 assistant 消息 append 进 payload 并最多重试 1-2 次。

---

## P1 — 工程化与稳定性（决定项目能否长期维护）

- [ ] **11. 补齐包元数据与项目门面** `pyproject.toml`、`README.md` `S`
  - `README.md` 是 0 字节，`description = ""`：PyPI/GitHub 首页全空。至少写：简介、安装、最小配置示例、技能目录规范、运行方式、安全提示。
  - `LICENSE` 是 MIT，但 `pyproject.toml` 没有 `license` / `license-files` 字段。
  - 版本号在 `pyproject.toml:3` 与 `clovers_agent/__init__.py:5` 双写，改用 `dynamic = ["version"]` + `[tool.hatch.version] path = "clovers_agent/__init__.py"`。
  - 补 `[project.urls]`、`keywords`、`classifiers`；`copyright` 署名与 `authors` 不一致，顺手统一。

- [ ] **12. 纳入版本控制** `S`
      当前目录不是 git 仓库，也没有 `.gitignore`。建议 `git init` 并忽略：`__pycache__/`、`*.pyc`、`./data/`（含 `usages`、`sessions`、`sentence_model_cache`）、`.venv/`、`.env`、`payload_*.json`。
      没有 VCS 意味着上述"回滚/对比"建议都无法落地，这是最高性价比的一步。

- [ ] **13. 建立质量基线：ruff + 类型检查 + pytest + CI** `M`
  - 加 `ruff`（`E,F,I,UP,B,SIM,RUF`）与 `ruff format`；当前有 16 行超过 120 字符（如 `skill.py:246,249,255`、`core.py:101,238`、`api.py:136`），格式也不统一。
  - 类型检查建议 `basedpyright` 或 `mypy`：P0-1/P0-2 这类问题本可被静态发现。现有必改项：`load_skill_md` 隐式返回 `None` 但 overload 声明 `IntroK | SingleK`（`skill.py:291-316`）；`execute_script` 的 `process` 可能未绑定；`Session.api/payload` 用 `None` 再 `type: ignore`（`session.py:144-148`）。
  - 首批测试（纯逻辑、无需网络）：
    - `parse_skill` / `skill_wrapper` 的前置元数据与参数注入；
    - `deep_add` 遇到 `None` / 类型冲突；
    - `Session.over` 的图片占位、`step` 的裁剪与解耦；
    - `TopicDecoupler` 的初始状态与 topic change 分支；
    - `main.format_message` 的分行与锁行为；
    - `OpenAIAPI.call_api` 用 `httpx.MockTransport` 覆盖 429/5xx/非法 JSON/`content:null`；
    - 回归测试 P0-1：非法工具参数不得抛 `UnboundLocalError`。
  - CI 矩阵至少 3.12 / 3.13（`torch` 对 3.14 的支持要单独验证；本机默认是 3.14，装 `torch>=2.5` 可能没有轮子）。

- [ ] **14. 给会话记忆加上硬上限** `clovers_agent/session.py:109-111`、`clovers_agent/core.py:336` `M`
      `refresh()` 里裁剪 `recorder` 的代码被注释掉了，只按 `memory_size` 触发总结；一旦 `summary_context` 失败（网络/超时），`recorder` 会无限增长，之后每轮都带超大 prompt，最终 413 或费用爆炸。
      修法：总结失败时仍裁剪到 `memory_size` 的一半（或丢弃最旧记录）；`recorder.json` 加载时也做上限截断；`image_recorder` 加 `maxlen`。

- [ ] **15. 去掉 import 期的重副作用** `clovers_agent/main.py:11-14` `M`
      模块一被导入就会：创建 `httpx.AsyncClient`、加载 `SentenceTransformer`（下载/占用内存）、读配置、构建整个 Agent。导致无法单测、文档构建失败、任何缺失 API key 的环境直接 import error。
      修法：提供 `create_agent()` 工厂 + 惰性单例；把模型加载放到首次使用或 `startup` 钩子；`main.py` 只保留装配代码。

- [ ] **16. `format_message` 持锁 sleep，阻塞后续回复** `clovers_agent/main.py:49-57` `S`
      最多 4 条消息、每条 `sleep(min(0.12*len, 8))`，最坏 ~32 秒，期间 `sending_lock` 一直占用，同一会话的新回复全部排队。
      修法：sleep 移到锁外；或改成每会话一个发送队列/worker，锁只保护"取任务"；同时避免按 `\n` 切分代码块/多行 Markdown（>4 行才整段发送的阈值可配置）。

- [ ] **17. 技能加载的 Markdown 扫描范围过大** `clovers_agent/skill.py:348-350` `S`
      `skill_path.rglob("*.md")` 会把 `references/**/*.md`、`scripts/**` 下的文档也算作技能定义；若参考资料自带 `name/description` front matter，会被误注册成工具，同时还会进 `references_map`。
      修法：排除 `references/` 与 `scripts/` 子目录（或只扫描 skill 根目录的 `*.md`）；排序后加载以保证工具顺序稳定（也有利于 prompt 缓存）。

- [ ] **18. 技能加载失败静默 + 回滚不完整** `clovers_agent/skill.py:397-408,318-378` `M`
      `load_module_from_path` 用裸 `except: return` 吞掉所有导入错误，用户只会看到"技能没加载"；`load_skill` 只对 `other_mds` 循环做了回滚，`scripts`/`references` 阶段失败会留下半注册的分类。
      修法：记录完整异常堆栈；把注册过程包在 `try` 中，失败时统一 `remove_category` + `_loaded_categories.discard`；加载结果汇总成一份报告写日志。

- [ ] **19. 日志可能泄露隐私且体积不可控** `clovers_agent/api.py:59`、`clovers_agent/skill.py:414-418`、`clovers_agent/core.py:382` `S`
      4xx 时打印完整 payload（整段聊天记录）；工具调用 DEBUG 日志打印完整返回值（可能是整份参考文档/脚本输出）；用量日志逐条打印。
      修法：日志级别默认 INFO，payload 只打印角色/长度/工具名；工具输出截断到 N 字符；提供 `LOG_REDACT` 配置。

- [ ] **20. 图片下载与内联无边界** `clovers_agent/api.py:68-77`、`clovers_agent/utils.py:16-19` `M`
      `download_url` 把整张图读进内存转 base64，无大小/类型限制；`puremagic.from_string` 识别失败会生成 `data:None;base64,...`；失败静默返回 `None`。
      修法：`httpx` 流式读取 + 上限（如 10MB）；校验 `content-type`；`mime or "application/octet-stream"`；失败时记录一次并给模型一个"图片不可用"的文本占位。

- [ ] **21. 用量统计可能丢数据** `clovers_agent/core.py:174-183,385-389` `S`
      `chat()` 中 `update_usage` 不在 `finally` 里，`handle_chat` 抛异常时本轮用量丢失；`daily_tasks` 清空 `usage_counter` 前没有落盘；2:00 前跨天仍写前一天的文件。
      修法：`try/finally` 保证落盘；`daily_tasks` 先 `save_json` 再清空；以 `datetime.now().date()` 判断跨天，而不是只在 2 点更新。

- [ ] **22. 异常兜底时 dump 的调试文件会无限增长** `clovers_agent/core.py:364-365` `S`
      文件名用 `id(session.payload)`（内存地址，无意义）且永不清理；payload 里含 base64 图片，单文件可达数十 MB。
      修法：改用 `session_id + 时间戳`，只保留最近 N 份；写入前剥离 data URL 或替换成 `[image]`。

- [ ] **23. 依赖策略：体积、下限与环境** `pyproject.toml:8-20` `M`
  - `torch` + `sentence-transformers` + `numpy` 是硬依赖（安装量级 GB），但只用于话题解耦器；建议改成 optional extra `[embedding]`，缺失时降级为"不做话题解耦"而非 import 失败。
  - `numpy>=2.4.4` 下限过新，会排除大量已有环境（代码只用到 `quantile/mean/std`）；建议 `numpy>=1.26`，或说明为何必须 2.4。
  - `torch` 无上界（`>=2.5.0`），大版本升级可能破坏 API；建议 `>=2.5,<3`。
  - 增加 `[dependency-groups] dev` 或 `uv.lock`，固定可复现开发环境；文档注明推荐 3.12/3.13。

- [ ] **24. 配置层的健壮性与密钥管理** `clovers_agent/config.py:18,207-224,231-233` `S`
  - `extra_body: dict[str, Any] = {}`、`whitelist/blacklist: list[str] = []` 是可变的类级默认值（pydantic 会拷贝，但 lint 会报，也易误用），改用 `Field(default_factory=...)`。
  - `path`、`sentence_model_cache` 默认相对 CWD（`./data/...`），从不同目录启动会读到不同数据；建议基于用户数据目录或显式要求绝对路径。
  - `api_key` 明文存配置；至少支持 `${ENV_VAR}` 展开，或在 README 中提示权限设置。
  - `url` 未做校验/归一化，若用户填了完整的 `/chat/completions` 会拼成两遍。

---

## P2 — 性能、架构与体验

- [ ] **25. 拆分 `core.py`（509 行）里的多重职责** `L`
      `CloversAgent` 同时负责：提示词加载、模型工厂、会话存取、路由、工具注册/执行、脚本沙箱、持久化、定时任务。建议拆成 `PromptStore`、`SessionStore`、`AgentRuntime`（路由 + 工具循环）、`ToolSandbox`、`UsageRecorder`，`handle_chat` 中的 chime-in / wait / execute 三个分支各抽一个方法。收益：可测试性、可替换模型、降低 #1/#3 类 bug 的复发率。

- [ ] **26. 持久化改为增量写入** `M`
      当前每条消息都重写整个 `recorder.json`，历史越长写入越慢（O(n²) I/O）。改成按行追加 `recorder.jsonl` + 定期 compact；或仅在本轮结束时 append 两条记录。

- [ ] **27. 会话生命周期与内存** `M`
      `self.sessions` 只有每日 2 点清理；`Session` 内含 `TopicDecoupler`（torch 张量）与图片 deque。建议 TTL + LRU 上限（如 1000 会话）、`image_recorder` 按字节/数量双限、`nicknames` 字典设上限，并在清理时跳过持有 `execute_lock` 的会话。

- [ ] **28. 工具并发上限** `S`
      `core.py:249` 用 `asyncio.gather` 无限制并发执行工具；多个 `execute_script` 可同时拉起大量进程。加 `asyncio.Semaphore(4)` 或按工具类型限流，并处理 `gather` 中一个工具抛 `SessionComplete` 时其余协程的取消。

- [ ] **29. 清理死代码与重复实现** `S`
  - `core.py:104-106` 与 `skill.py:380-382` 重复的 `skill_dirs` property；
  - `utils.py:12-13` `is_base64` 未使用；
  - `api.py:23-33` `build_message` 未使用；
  - `embedding.py:12-34` `similarity` / `batch_similarity` 未使用；
  - `session.py:109-111` 被注释的裁剪逻辑、`session.py:167-170` `unimportant_context` 未使用；
  - `constants.py:10` `HIDDEN_CATEGORY`、`config.py:145` 未使用（若为预留请在注释中说明）；
  - `embedding.py:1-9` 注释掉的 `HF_HUB_OFFLINE` 与仅为降噪的 `transformers` 导入——改为配置环境变量 + 标准 logging。

- [ ] **30. 性能微优化** `S`
  - `session.step` 每次遍历全部记录做 `char_count`（`session.py:127`），维护增量计数即可；
  - `session.image_url` 线性扫描（`session.py:104-105`），改为 `dict[image_id, url]`；
  - `core.py:352` 每轮都拼 `Now:`，可在跨分钟/跨天时才更新；
  - `retry` 增加 jitter 并尊重 `Retry-After`（`api.py:12,43-66`）。

- [ ] **31. 文档与示例** `M`
      仓库里没有 `AgentSkills/`、`AgentTools/` 示例，而默认配置指向它们；`CONFIG.plugins`/`plugin_dirs`/`skill_dirs` 的关系、`SKILL.md` front matter 格式、`skill.py` 中 `content` 参数约定、`Session.complete()`/`session.unimportant` 等插件扩展点都没有文档。建议补 `docs/` + `examples/weather` 最小技能，并在 README 给出"5 分钟跑起来"路径。

- [ ] **32. 可观测性** `S`
      `self.tool_failures`（`core.py:85,238`）只进不出，建议提供只读访问器/调试命令并把最近失败写进日志摘要；补 token/延迟/工具成功率指标（结构化日志即可），便于定位"机器人为什么不回复"。

- [ ] **33. 技能热重载** `S`
      `reload_skills()` 已实现但没有任何触发入口。可加一个管理工具/命令（如 `reload_skills`，仅管理员可用）或文件监听，避免改完技能必须重启进程。

- [ ] **34. 统一代码风格细节** `S`
      行宽不统一（见 #13）；`config.py` 中的大段提示词可考虑移到 `prompts/*.md` 作为默认模板（当前是"代码里的默认值 + 首次运行落盘"，两份容易漂移）；`CloversAgent` 的 docstring 只有 `"""OpenAI API"""`，与实际职责不符。
      顺手修正拼写：`creat_api`（`core.py:115`）、`update_context(messags)`（`session.py:172`）、`chat_prompt` 里的"情绪连贯**：：**"双冒号（`config.py:95`）。

---

## 建议执行顺序

1. **第 1 天（止血 + 门面）**：#1、#2、#3、#6、#7、#11、#12。完成后可正常发布、崩溃面显著收敛。
2. **第 2-3 天（可控性）**：#4、#5、#9、#10、#14、#16、#17、#19。控制成本与安全边界。
3. **第 1 周内（质量基线）**：#13、#15、#18、#20-#24。让后续改动有测试与静态检查兜底。
4. **第 2 周（架构）**：#25-#28、#31-#33。按实际迭代压力排期，不建议一次性大重构。

---

## 验收标准

- [ ] `ruff check .`、`ruff format --check .` 通过；类型检查无 error。
- [ ] `pytest` 覆盖 P0 回归用例，CI 在 3.12/3.13 绿灯。
- [ ] 在干净虚拟环境执行 `pip install -e ".[dev]"` + README 的最简配置可启动，进程 import 不再要求网络与 API key。
- [ ] 注入非法工具参数、`content:null`、损坏的 `usage.json`、超大脚本输出四种故障，均有明确日志且服务不中断。
- [ ] 24 小时群聊压测后，`recorder` 长度、内存、单轮 prompt token 均有上限。
