# clovers-agent 优化 TODO

> 审计范围：`clovers-agent/` 全部 14 个 Python 文件（约 1,950 行）＋ `pyproject.toml` / `LICENSE` / `README.md`。
> 已核实：全部源码可在 Python 3.14 下编译通过；抽查的 9 个依赖版本（numpy/pydantic/torch/sentence-transformers/clovers/clovers-client/clovers-apscheduler/chardet/puremagic）在 PyPI 均存在；项目当前**不是 git 仓库**、**没有任何测试 / CI / lint 配置**、`README.md` 为 0 字节。
> 未运行：`clovers` 系列依赖未安装，动态行为（路由、插件加载）未实测，相关结论来自静态阅读，标注了"需确认"。

图例：`S` ≈ 半小时内，`M` ≈ 半天，`L` ≈ 1 天以上。

---

## P2 — 性能、架构与体验

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
