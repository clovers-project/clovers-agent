from pathlib import Path
from clovers_agent import CloversAgent, Event
from clovers_agent.main import CONFIG as AGENT_CONFIG
from clovers_agent.session import extract_plain_text
from clovers.logger import logger
from .toolkit import TOOLS
from .workspace import get_session_id, WORKSPACE


@TOOLS.hook
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


@TOOLS.register(
    "write_note",
    "记录信息。当用户与助手约定、提出长期要求、让助手记住某事、或有其他需要记录的信息时，必须调用此工具记录。",
    {"content": {"type": "string", "description": "笔记内容。内容应为简洁清晰的陈述句。"}},
)
async def _(agent: CloversAgent, event: Event, content: str):
    session_id = get_session_id(agent, event)
    note_file = WORKSPACE / session_id / "NOTE.md"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not note_file.exists() or not (note := note_file.read_text(encoding="utf-8")):
            note_file.write_text(content, encoding="utf-8")
            return ""
        aux = agent.auxiliary
        payload = aux.build_payload(
            ({"role": "user", "content": "\n".join(note) + "\n" + content},),
            "你是一个记忆碎片整合专家。你的任务是维护一组记录信息。请遵循以下严格准则：\n\n"
            "1. 每一行只能包含一个独立的信息。严禁换行。\n"
            "2. 若记录内容相互矛盾，仅输出顺序靠后的内容，严禁输出其他矛盾记录。\n"
            "3. 合并相同、相近信息与对同事物的不同记录。\n"
            "4. 不符合上述情况的笔记条目完全原文输出，对改动的内容完全保留原文风格。",
        )
        new_note = await aux.call_api(payload)
        note_file.write_text(new_note["content"], encoding="utf-8")
        return ""
    except Exception as e:
        logger.error(e)
        return ""


USER_PROFILE = Path(AGENT_CONFIG.path) / "UserProfile"


@TOOLS.hook
async def _(agent: CloversAgent, event: Event):
    user_profile = USER_PROFILE / f"{event.user_id}.md"
    if not user_profile.exists():
        return ""
    return f"当前关注：\n\n{user_profile.read_text(encoding='utf-8')}"


@TOOLS.register(
    "update_user_profile",
    "当用户展现出性格特征、提及偏好或与你发生深刻互动时，调用此工具以更新**你**对该用户的私密印象。",
    {
        "observation": {"type": "string", "description": "从上下文中你观察到的重点信息（性格、癖好、言行风格等）"},
        "impression": {"type": "string", "description": "简述你现在对他的主观感觉"},
    },
)
async def _(agent: CloversAgent, event: Event, observation: str, impression: str):
    USER_PROFILE.mkdir(parents=True, exist_ok=True)
    user_profile_path = USER_PROFILE / f"{event.user_id}.md"
    old_profile = user_profile_path.read_text(encoding="utf-8") if user_profile_path.exists() else "空"
    session = agent.current_session(event)
    context = "\n".join(extract_plain_text(msg["content"]) for msg in session)
    system_prompt = f"""任务：你的任务是根据新的互动，更新关于用户 {event.nickname} 的私密档案。
档案待更新：
{old_profile}

要求：
结合档案与触发点，输出一份 Markdown 格式的档案。不要写废话。
请按以下维度更新：
# 用户档案：[用户昵称]

- **称呼**：(我怎么称呼对方)
- **标签**：(一个或几个标签)
- **约定**：(与用户有没有约定)
- **印象**：(对该用户的印象)
- **关系**：(目前与用户的关系怎么样)
... (额外维度，请自行添加)

# 记忆碎片（可选）
你认为值得记录的互动。
"""
    user_prompt = f"""触发点：

```text
{context}
```

- 观察到：{observation}
- 你的对用户的感受：{impression}
"""
    aux = agent.auxiliary
    payload = aux.build_payload(({"role": "user", "content": user_prompt},), system_prompt)
    resp = await aux.call_api(payload)
    user_profile_path.write_text(resp["content"], encoding="utf-8")
    return ""
