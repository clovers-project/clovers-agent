from pathlib import Path
from clovers_agent import CloversAgent, Event
from clovers_agent.main import CONFIG as AGENT_CONFIG
from clovers_agent.session import extract_plain_text
from clovers_agent.embedding import similarity
from clovers_agent.constants import ON_CHAT
from clovers.logger import logger
from .toolkit import TOOLS, CONFIG
from .workspace import get_session_id, WORKSPACE


@TOOLS.on_category(ON_CHAT)
async def _(agent: CloversAgent, event: Event):
    session_id = get_session_id(agent, event)
    note_file = WORKSPACE / session_id / "NOTE.md"
    if not note_file.exists():
        return ""
    try:
        note = note_file.read_text(encoding="utf-8")
    except Exception as e:
        note_file.unlink()
        logger.error(f"笔记读取失败: {e}")
        return ""
    return f"笔记内容\n\n{note}\n"


SCRIBE_PROMPT = """
你是一个专业的记忆碎片整合专家。你的任务是对一组记录进行维护和清理。

请严格遵守以下准则：
1. 每一行输出只能包含一个独立的信息点。严禁在单条信息中使用换行符或分段。
2. 如果记录之间存在相互矛盾的信息，请仅输出在原始文本中顺序靠后的内容。严禁输出被覆盖的旧矛盾记录。
3. 将相同、相近的信息，以及对同一事物在不同侧面的记录整合到一起。整合后的条目应涵盖所有相关细节，但需保持简洁。
4. 对于不涉及合并或冲突的独立条目，请完全按照原文内容输出。对于修改或合并的内容，必须严格保留原文的语言风格。

请开始你的任务。
""".strip()
similarity_threshold = CONFIG.note_similarity_threshold


@TOOLS.register(
    "write_note",
    "当**助手**认为上下文中出现了重要或需要长期记录信息时调用",
    {"content": {"type": "string", "description": "笔记内容。内容应为简洁清晰的陈述句。"}},
    ON_CHAT,
)
async def _(agent: CloversAgent, event: Event, content: str):
    session_id = get_session_id(agent, event)
    note_file = WORKSPACE / session_id / "NOTE.md"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not note_file.exists() or not (note := note_file.read_text(encoding="utf-8").strip()):
            note_file.write_text(content, encoding="utf-8")
        elif all(similarity(line, content, agent.sentence_model) < similarity_threshold for line in note.split("\n")):
            note_file.write_text(note + "\n" + content, encoding="utf-8")
        else:
            api = agent.current_session(event).api
            payload = api.build_payload(({"role": "user", "content": note + "\n" + content},), SCRIBE_PROMPT)
            new_note = await api.call_api(payload, agent.current_session(event).usage_counter)
            note_file.write_text(new_note["content"], encoding="utf-8")
        return "Done"
    except Exception as e:
        logger.error(e)
        return f"Error: {e}"


USER_PROFILE = Path(AGENT_CONFIG.path) / "UserProfile"

ARCHIVIST_PROMPT = """
你是一位专业、细致且观察力敏锐的“档案维护员”。你的任务是根据"触发点"信息，更新助手对用户的私密档案。

### 更新要求：

- 将新观察到的信息与旧档案合并。
- 如果新旧记录存在冲突，请务必以"触发点"为准。
- 对于因冲突被覆盖的信息，或因本次互动而要求删除的信息，请在更新后的档案中将其彻底移除，不留任何痕迹。
- 合并重复项，使档案保持简洁、客观、准确。
- 禁止输出除文档外的任何内容。

### 处理流程：

- 确定哪些维度需要更新，哪些需要保持不变，哪些需要删除。
- 构思是否需要添加新的档案维度。
- 筛选值得记录的“记忆碎片”，对不重要的信息进行删除。

完成思考后，请按照以下格式输出更新后的档案：

# 用户档案：[用户昵称]

- **称呼**：(记录你应当如何称呼对方)
- **标签**：(为用户贴上一个或几个核心关键词标签)
- **约定**：(记录你与用户达成的任何承诺、协议或长期的互动约定)
- **偏好**：(记录用户的话题偏好、语言风格偏好、特定观点、禁忌等)
- **印象**：(基于长期互动和本次感受，你对该用户形成的整体印象)
- **关系**：(描述目前你与用户的关系状态)
- **[额外维度]**：(如有必要，请根据互动自行添加)

# 记忆碎片（可选）
(请在此处以简练的语言记录你认为值得永久保存的、具有代表性的互动瞬间或细节，必须是重要互动。如果没有，可略过此部分。)

请开始更新档案。
""".strip()


@TOOLS.on_category(ON_CHAT)
async def _(agent: CloversAgent, event: Event):
    user_profile = USER_PROFILE / f"{event.user_id}.md"
    if not user_profile.exists():
        return f"当前关注：\n\n# 用户档案：{event.nickname}\n\n目前尚无该用户档案，请在**上下文足够充分**时进行第一次更新。"
    return f"当前关注：\n\n{user_profile.read_text(encoding='utf-8')}"


@TOOLS.register(
    "update_user_profile",
    "当用户与助手约定某事、提及偏好、展现出性格、人际关系、或上下文中出现可以一定程度上修正你对用户印象的语境时应主动调用此工具。",
    {
        "observation": {"type": "string", "description": "从上下文中你观察到的重点信息（性格、癖好、言行风格等）"},
        "impression": {"type": "string", "description": "你对用户当前的主观情感评价。"},
    },
    ON_CHAT,
)
async def _(agent: CloversAgent, event: Event, observation: str, impression: str):
    USER_PROFILE.mkdir(parents=True, exist_ok=True)
    user_profile_path = USER_PROFILE / f"{event.user_id}.md"
    old_profile = user_profile_path.read_text(encoding="utf-8") if user_profile_path.exists() else "无"
    session = agent.current_session(event)
    context = "\n".join(extract_plain_text(msg["content"]) for msg in session)
    user_prompt = (
        f"上下文：\n\n{context}\n"
        f"档案待更新：\n\n{old_profile}\n"
        f"触发点：\n\n"
        f"- 观察到：{observation}\n"
        f"- 你的对用户的感受：{impression}"
    )
    api = session.api
    payload = api.build_payload(({"role": "user", "content": user_prompt},), ARCHIVIST_PROMPT)
    resp = await api.call_api(payload, session.usage_counter)
    user_profile_path.write_text(resp["content"], encoding="utf-8")
    return ""
