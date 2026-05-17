from pathlib import Path
from clovers_agent import CloversAgent, Event
from clovers_agent.config import CONFIG as AGENT_CONFIG
from clovers_agent.session import extract_plain_text
from clovers_agent.embedding import similarity
from clovers_agent.constants import ON_CHAT
from clovers.logger import logger
from .constants import WRITE_NOTE_PROMPT, UPDATE_USER_PROFILE_PROMPT
from ..toolkit import TOOLS, CONFIG
from ..workspace import WORKSPACE

SIM_THRESHOD = CONFIG.note_similarity_threshold
REMINDER_THRESHOLD = CONFIG.reminder_threshold
STRONG_REMINDER_THRESHOLD = CONFIG.strong_reminder_threshold
USER_PROFILE = Path(AGENT_CONFIG.path) / "UserProfile"

UPDATE_USER_PROFILE = "update_user_profile"


@TOOLS.on_category(ON_CHAT)
async def _(agent: CloversAgent, event: Event):
    extra = agent.current_session(event).extra
    if UPDATE_USER_PROFILE not in extra:
        extra[UPDATE_USER_PROFILE] = {}
    counter = extra[UPDATE_USER_PROFILE]
    user_id = event.user_id
    count = counter[user_id] = counter.get(user_id, 0) + 1
    note_file = WORKSPACE / agent.session_id(event) / "NOTE.md"
    notes = []
    if note_file.exists():
        try:
            note = note_file.read_text(encoding="utf-8").strip()
            if note:
                notes.append(f"笔记内容\n\n{note}\n")
        except Exception as e:
            note_file.unlink()
            logger.error(f"笔记读取失败: {e}")
    user_profile = USER_PROFILE / f"{user_id}.md"
    if not user_profile.exists():
        notes.append(f"""\
# 用户档案：{event.nickname}

目前尚无该用户档案，请在**上下文足够充分**时进行使用 '{UPDATE_USER_PROFILE}' 工具进行第一次更新。""")
    else:
        notes.append(user_profile.read_text(encoding="utf-8"))
        if count > STRONG_REMINDER_THRESHOLD:
            notes.append(f"档案在 {count} 次对话前更新，请及时使用 '{UPDATE_USER_PROFILE}' 工具对档案进行更新。")
        elif count > REMINDER_THRESHOLD:
            notes.append(f"档案在 {count} 次对话前更新，请确认用户档案是否过时。")
    return "\n\n".join(notes)


@TOOLS.register(
    "archive_memory",
    "用于将当前上下文中的关键事实、约定、备忘事项等具有长期保存价值的信息进行持久化归档。当识别到此类信息时，应主动调用此工具",
    {"content": {"type": "string", "description": "需要归档的记忆内容。请使用客观、简洁的陈述句，避免冗余修饰"}},
    ON_CHAT,
)
async def _(agent: CloversAgent, event: Event, content: str):
    note_file = WORKSPACE / agent.session_id(event) / "NOTE.md"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not note_file.exists() or not (note := note_file.read_text(encoding="utf-8").strip()):
            note_file.write_text(content, encoding="utf-8")
        elif all(similarity(line, content, agent.sentence_model) < SIM_THRESHOD for line in note.split("\n")):
            note_file.write_text(note + "\n" + content, encoding="utf-8")
        else:
            api = agent.current_session(event).api
            payload = api.build_payload(({"role": "user", "content": note + "\n" + content},), WRITE_NOTE_PROMPT)
            new_note = await api.call_api(payload, agent.current_session(event).usage_counter)
            note_file.write_text(new_note["content"], encoding="utf-8")
        return "OK"
    except Exception as e:
        logger.error(e)
        return f"Error: {e}"


@TOOLS.register(
    UPDATE_USER_PROFILE,
    "用于更新助手对用户的印象档案。当用户约定称呼、展现偏好、或你对该用户的印象需要修正时，应主动调用此工具",
    {
        "observation": {"type": "string", "description": "从对话中提取的关键观察（如用户性格特征、偏好变化、言行风格等）。"},
        "impression": {"type": "string", "description": "你对用户当前的主观情感评价。"},
    },
    ON_CHAT,
)
async def _(agent: CloversAgent, event: Event, observation: str, impression: str):
    USER_PROFILE.mkdir(parents=True, exist_ok=True)
    user_id = event.user_id
    user_profile_path = USER_PROFILE / f"{user_id}.md"
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
    payload = api.build_payload(({"role": "user", "content": user_prompt},), UPDATE_USER_PROFILE_PROMPT)
    resp = await api.call_api(payload, session.usage_counter)
    user_profile_path.write_text(resp["content"], encoding="utf-8")
    try:
        del session.extra[UPDATE_USER_PROFILE][user_id]
    except KeyError:
        pass
    return "OK"
