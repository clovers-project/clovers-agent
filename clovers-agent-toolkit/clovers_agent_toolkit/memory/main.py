from pathlib import Path
from clovers_agent import CloversAgent, Event
from clovers_agent.config import CONFIG as AGENT_CONFIG
from clovers_agent.constants import ON_CHAT, HIDDEN_CATEGORY
from .constants import UPDATE_USER_PROFILE, UPDATE_USER_PROFILE_PROMPT, EDIT_USER_PROFILE, EDIT_USER_PROFILE_DESC
from .typing import UserProfile, MemoryItem, MemoryItemTuple
from ..toolkit import TOOLS, CONFIG

REMINDER_THRESHOLD = CONFIG.reminder_threshold
STRONG_REMINDER_THRESHOLD = CONFIG.strong_reminder_threshold
USER_PROFILE = Path(AGENT_CONFIG.path) / "UserProfile"


@TOOLS.on_category(ON_CHAT)
async def _(agent: CloversAgent, event: Event):
    extra = agent.current_session(event).extra
    if UPDATE_USER_PROFILE not in extra:
        extra[UPDATE_USER_PROFILE] = {}
    counter = extra[UPDATE_USER_PROFILE]
    user_id = event.user_id
    count = counter[user_id] = counter.get(user_id, 0) + 1
    notes = []
    user_profile = USER_PROFILE / f"{user_id}.json"
    if not user_profile.exists():
        notes.append(f"""\
# 用户档案：{event.nickname}

目前尚无该用户档案，请在**上下文足够充分**时进行使用 '{UPDATE_USER_PROFILE}' 工具进行第一次更新。""")
    else:
        notes.append(UserProfile.model_validate_json(user_profile.read_text(encoding="utf-8")).to_markdown())
        if count > STRONG_REMINDER_THRESHOLD:
            notes.append(f"档案在 {count} 次对话前更新，请及时使用 '{UPDATE_USER_PROFILE}' 工具对档案进行更新。")
        elif count > REMINDER_THRESHOLD:
            notes.append(f"档案在 {count} 次对话前更新，请确认用户档案是否过时。")
    return "\n\n".join(notes)


@TOOLS.register(
    UPDATE_USER_PROFILE,
    "用于更新助手对用户的印象档案。当用户约定称呼、展现偏好、或你对该用户的印象需要修正时，应主动调用此工具",
    category=ON_CHAT,
)
async def _(agent: CloversAgent, event: Event, **kwargs):
    session = agent.current_session(event)
    assert "tools" in session.payload
    if not any(tool["function"]["name"] == EDIT_USER_PROFILE for tool in session.payload["tools"]):
        session.payload["tools"].append(agent.manifest[EDIT_USER_PROFILE])
    return UPDATE_USER_PROFILE_PROMPT


@TOOLS.register(
    EDIT_USER_PROFILE,
    EDIT_USER_PROFILE_DESC,
    {
        "address_as": {"type": "string", "description": "记录你应当如何称呼对方"},
        "tags": {"type": "string", "description": "为用户贴几个核心关键词"},
        "preferences": {"type": "string", "description": "记录用户的话题偏好、语言风格偏好、特定观点、禁忌等"},
        "impression": {"type": "string", "description": "你对该用户的整体印象"},
        "new_memories": {"type": "array", "items": {"type": "string", "description": "新记忆"}},
        "promote_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "number", "description": "升星的记忆序号"},
                    "content": {"type": "number", "description": "更新后的记忆，该记忆应融合"},
                },
            },
        },
        "demote_memory_ids": {"type": "array", "items": {"type": "number", "description": "降星的记忆序号"}},
    },
    category=HIDDEN_CATEGORY,
    required=[],
)
async def _(
    agent: CloversAgent,
    event: Event,
    address_as: str = "",
    tags: str = "",
    preferences: str = "",
    impression: str = "",
    new_memories: list[str] | None = None,
    promote_memories: list[MemoryItem] | None = None,
    demote_memory_ids: list[int] | None = None,
):
    user_id = event.user_id
    user_profile = USER_PROFILE / f"{user_id}.json"
    USER_PROFILE.mkdir(parents=True, exist_ok=True)
    if not user_profile.exists():
        profile = UserProfile(
            nickname=event.nickname,
            address_as=address_as,
            tags=tags,
            preferences=preferences,
            impression=impression,
            memories=[{"id": i, "content": c, "star": 1} for i, c in enumerate(new_memories, 1)] if new_memories else [],
        )
    else:
        profile = UserProfile.model_validate_json(user_profile.read_text(encoding="utf-8"))
        profile.nickname = event.nickname
        if address_as:
            profile.address_as = address_as
        if tags:
            profile.tags = tags
        if preferences:
            profile.preferences = preferences
        if impression:
            profile.impression = impression
        old_memories: dict[int, MemoryItemTuple] = {int(x["id"]): (x["content"], x["star"]) for x in profile.memories}
        memories: list[MemoryItemTuple] = []
        if promote_memories:
            for x in promote_memories:
                if not "id" in x:
                    if "content" in x:
                        memories.append((x["content"], 1))
                    continue
                key = int(x["id"])
                c = x.get("content")
                if key not in old_memories:
                    if c:
                        memories.append((c, 1))
                else:
                    old_c, star = old_memories[key]
                    memories.append((c or old_c, min(star + 1, 5)))
                    del old_memories[key]
        if demote_memory_ids:
            for key in demote_memory_ids:
                if key not in old_memories:
                    continue
                old_c, star = old_memories[key]
                memories.append((old_c, star - 1))
                del old_memories[key]
        if new_memories:
            memories.extend((x, 1) for x in new_memories)
        memories.extend(old_memories.values())
        dstar = 1 if len(memories) >= 20 else 0
        memories = sorted((x for x in memories if x[1] >= dstar), key=lambda x: x[1], reverse=True)
        profile.memories.clear()
        profile.memories.extend({"id": i, "content": c, "star": star} for i, (c, star) in enumerate(memories, 1))
    user_profile.write_text(profile.model_dump_json(indent=4), encoding="utf-8")
    return profile.to_markdown()
