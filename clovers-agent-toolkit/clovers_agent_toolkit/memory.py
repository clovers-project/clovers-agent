from pathlib import Path
from clovers_agent import CloversAgent, Event
from clovers_agent.main import CONFIG as AGENT_CONFIG
from clovers_agent.session import extract_plain_text
from clovers_agent.embedding import similarity
from clovers_agent.constants import ON_CHAT
from clovers.logger import logger
from .toolkit import TOOLS, CONFIG
from .workspace import get_session_id, WORKSPACE

SIM_THRESHOD = CONFIG.note_similarity_threshold
SCRIBE_PROMPT = CONFIG.scribe_prompt
ARCHIVIST_PROMPT = CONFIG.archivist_prompt
REMINDER_THRESHOLD = CONFIG.reminder_threshold
STRONG_REMINDER_THRESHOLD = CONFIG.strong_reminder_threshold
USER_PROFILE = Path(AGENT_CONFIG.path) / "UserProfile"


@TOOLS.on_category(ON_CHAT)
async def _(agent: CloversAgent, event: Event):
    session_id = get_session_id(agent, event)
    extra = agent.current_session(event).extra
    if "update_user_profile" not in extra:
        extra["update_user_profile"] = {}
    counter = extra["update_user_profile"]
    user_id = event.user_id
    count = counter[user_id] = counter.get(user_id, 0) + 1
    note_file = WORKSPACE / session_id / "NOTE.md"
    notes = []
    if note_file.exists():
        try:
            note = note_file.read_text(encoding="utf-8")
            notes.append(f"笔记内容\n\n{note}\n")
        except Exception as e:
            note_file.unlink()
            logger.error(f"笔记读取失败: {e}")
    user_profile = USER_PROFILE / f"{user_id}.md"
    if user_profile.exists():
        notes.append(f"当前关注：\n\n# 用户档案：{event.nickname}\n\n目前尚无该用户档案，请在**上下文足够充分**时进行第一次更新。")
    else:

        if count > STRONG_REMINDER_THRESHOLD:
            notes.append(f"当前关注：（{count} 次对话前更新，请确认是否过时。）")
        elif count > REMINDER_THRESHOLD:
            notes.append(f"当前关注：（{count} 次对话前更新）")
        else:
            notes.append(f"当前关注：")
        notes.append(user_profile.read_text(encoding="utf-8"))

    return "\n\n".join(notes)


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
        elif all(similarity(line, content, agent.sentence_model) < SIM_THRESHOD for line in note.split("\n")):
            note_file.write_text(note + "\n" + content, encoding="utf-8")
        else:
            api = agent.current_session(event).api
            payload = api.build_payload(({"role": "user", "content": note + "\n" + content},), SCRIBE_PROMPT)
            new_note = await api.call_api(payload, agent.current_session(event).usage_counter)
            note_file.write_text(new_note["content"], encoding="utf-8")
        return "OK"
    except Exception as e:
        logger.error(e)
        return f"Error: {e}"


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
    payload = api.build_payload(({"role": "user", "content": user_prompt},), ARCHIVIST_PROMPT)
    resp = await api.call_api(payload, session.usage_counter)
    user_profile_path.write_text(resp["content"], encoding="utf-8")
    if "update_user_profile" in session.extra:
        if user_id in session.extra["update_user_profile"]:
            del session.extra["update_user_profile"][user_id]
    return "OK"
