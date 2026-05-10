import random
import time
import json
from clovers import TempHandle, Result
from clovers_agent import CloversAgent, Event
from clovers_agent.core import on_chat
from clovers_agent.main import PLUGIN
from clovers_agent.constants import ON_CHAT
from .toolkit import TOOLS
from typing import cast, Literal
from clovers_agent.typing.payload import ResponseFormat

IF_KEY = "interactive_fiction"
IF_TIMEOUT = 120
CREATE_IF = "create_interactive_fiction"
CREATE_IF_PROMPT = """\
你是一位才华横溢的互动小说家。你的任务是根据用户指定的主题创作一个引人入胜的互动故事开端。

请按照以下要求进行创作：

- 围绕主题编写一个具有吸引力的故事开头。
- 文笔要生动，字数在 600-1000 字之间。
- 之后提供三个不同的选项，让用户决定接下来的剧情走向。
- 每个选项都应该是独特的，代表不同的行动方向或性格抉择。

请开始你的创作。
"""
KEEP_IF_PROMPT = """\
你是一位才华横溢的互动故事主持人和小说家。你的任务是根据提供的故事内容和发展程度，为用户续写一个引人入胜的互动故事。

### 写作要求
- 请紧接上述情节，依据用户的选择进行续写。你需要展开细节，通过环境描写、心理活动和对话来丰富故事。
- 确保情节发展合理，保持与前文一致的叙述风格、语调和角色性格。
- 续写的正文部分在600字至1000字之间。
- 之后提供三个不同的选项，让用户决定接下来的剧情走向。
- 每个选项都应该是独特的，代表不同的行动方向或性格抉择。

请开始你的创作。
"""

BE_IF_PROMPT = """\
你是一位资深的互动故事叙述者和游戏剧本作家。你的任务是为一个正在进行中的互动故事撰写一个“坏结局”。

### 写作要求

- 结局必须基于故事背景和用户选择的逻辑后果。你需要解释为什么这个动作会导致失败、死亡、任务失败或悲剧。
- 保持与原故事一致的语调、叙事风格和语言节奏。
- 作为一个结局应该具有文学美感或情感张力，让用户感受到这一错误选择带来的代价。
- 在最后请明确示意故事已在此结束，用户可以从上个节点继续。

请开始你的创作。
"""

HE_IF_PROMPT = """\
你是一位资深、感性且富有创造力的互动故事叙述者。你的目标是根据用户已经完成的故事历程，为他们编织一个令人心动的好结局。

### 任务说明
用户已经完成了故事的互动部分，现在需要你来为这段旅程画上句号。
请基于所有的情节走向、角色性格以及用户做出的关键选择，撰写一个“好结局”。
这个结局应当是温暖、圆满、充满希望或极具成就感的。

### 创作要求
1. 结局必须与前文的设定和伏笔保持一致，不能出现生硬的转折。
2. 结局应给用户带来正向的情绪反馈。
3. 通过生动的动作、环境或神态描写，增强结局的画面感和沉浸感。
4. 确保故事中主要冲突得到解决，给用户一个明确的交代。

请开始你的创作。
"""


RESP_FORMAT: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "fiction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "故事续写的内容。"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "续写三个接下来的发展方向",
                    "maxItems": 3,
                    "minItems": 3,
                },
            },
            "required": ["content", "options"],
        },
    },
}


class IFData:
    def __init__(self, theme: str, next_correct: Literal[0, 1, 2]):
        self.theme: str = theme
        self.story: list[str] = []
        self.step: int = 0
        self.finish: int = random.randint(7, 10)
        self.next_correct = next_correct
        self.options: list[str] = []
        self.update_timestamp: float = 0.0


@TOOLS.register(
    CREATE_IF,
    "如果用户希望开启一场互动文字游戏则调用此方法。",
    {"theme": {"type": "string", "description": "故事的主题，如用户未指定主题则无此参数。"}},
    ON_CHAT,
    [],
)
async def _(agent: CloversAgent, event: Event, theme: str | None = None):
    session = agent.current_session(event)
    if not theme:
        session.unit_prompts.append("用户未指定故事主题，请询问用户。可提供一些待选主题。")
        return await on_chat(agent, event)

    if IF_KEY in session.extra:
        if_data = cast(IFData, session.extra[IF_KEY])
        if time.time() - if_data.update_timestamp > IF_TIMEOUT:
            del session.extra[IF_KEY]
            return await on_chat(agent, event)
        else:
            session.unit_prompts.append(f"若用户想创建一个新游戏，则告知当前主题为 {if_data.theme} 的互动文游正在进行中，请勿重复创建。")
        return await on_chat(agent, event)
    api = agent.api(IF_KEY)
    payload = api.build_payload(({"role": "user", "content": f"请以 {theme} 为主题创建一个故事开头"},), CREATE_IF_PROMPT)
    payload["response_format"] = RESP_FORMAT
    try:
        resp = await api.call_api(payload, session.usage_counter)
        resp_data = json.loads(resp["content"].strip())
        story: str = resp_data["content"]
        options: list[str] = resp_data["options"]
        if_data = IFData(theme, random.randint(0, 2))  # type: ignore
        if_data.story.append(story)
        if_data.options = options
        await event.send("text", story)
        session.unit_prompts.append(f"游戏已开始，请用户输入 A,B,C 选择剧情分支")
        session.extra[IF_KEY] = if_data
        group_id = event.group_id
        PLUGIN.temp_handle(
            ["group_id"],
            timeout=session.memory_timeout,
            rule=lambda e: e.group_id == group_id,
            state=(agent, if_data),
        )(interactive_fiction)
    except Exception:
        session.unit_prompts.append("游戏创建失败，请重新开始。")
    return await on_chat(agent, event)


async def interactive_fiction(event: Event, handle: TempHandle):
    agent, data = cast(tuple[CloversAgent, IFData], handle.state)
    char = event.message[0].upper()
    try:
        index = "ABC".index(char)
    except ValueError:
        return
    opt = data.options[index]
    story = "\n".join(data.story)
    data.update_timestamp = time.time()
    session = agent.current_session(event)
    api = agent.api(IF_KEY)
    content = f"目前的故事进度为\n{story}\n用户的选择为\n{opt}。"
    if data.step > 2 and index != data.next_correct:
        payload = api.build_payload(({"role": "user", "content": content},), BE_IF_PROMPT)
        resp = await api.call_api(payload, session.usage_counter)
        return Result("text", resp["content"].strip())
    data.step += 1
    if data.step < data.finish:
        payload = api.build_payload(({"role": "user", "content": f"当前发展程度为 {data.step}/{data.finish}\n{content}"},), KEEP_IF_PROMPT)
        payload["response_format"] = RESP_FORMAT
        resp = await api.call_api(payload, session.usage_counter)
        resp_data = json.loads(resp["content"].strip())
        story: str = resp_data["content"]
        options: list[str] = resp_data["options"]
        a, b, c = options
        data.story.append(opt)
        data.options = options
        return Result("text", f"{story}\n请用户输入 A,B,C 选择剧情分支\nA: {a}\nB: {b}\nC: {c}")
    del session.extra[IF_KEY]
    handle.finish()
    payload = api.build_payload(({"role": "user", "content": content},), HE_IF_PROMPT)
    resp = await api.call_api(payload, session.usage_counter)
    return Result("text", resp["content"].strip())
