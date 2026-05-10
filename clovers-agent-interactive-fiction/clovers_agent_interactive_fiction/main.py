import random
import time
import json
from clovers import TempHandle, Result
from clovers.logger import logger
from clovers_agent import CloversAgent, Event
from clovers_agent.main import PLUGIN
from clovers_agent.utils import deep_add
from clovers_agent.constants import ON_CHAT
from .toolkit import TOOLS
from typing import cast, Literal
from clovers_agent.typing.payload import ResponseFormat

IF_KEY = "interactive_fiction"
IF_TIMEOUT = 120

RESP_PROMPT = """\
- 正文部分在300字至500字之间。
- 之后提供三个不同的选项，让用户决定接下来的剧情走向。
- 每个选项都应该是独特的，代表不同的行动方向或性格抉择。
- 输出格式为 JSON，包含以下字段：
  - content: 剧情内容
  - options: 三个剧情发展选项，本字段为一个长度为3的列表，每个元素是一个字符串
"""


CREATE_IF_PROMPT = f"""\
你是一位才华横溢的互动小说家。你的任务是根据用户指定的主题创作一个引人入胜的互动故事开端。

请按照以下要求进行创作：

- 围绕主题编写一个具有吸引力的故事开头。
{RESP_PROMPT}

请开始你的创作。
"""
KEEP_IF_PROMPT = f"""\
你是一位才华横溢的互动故事主持人和小说家。你的任务是根据提供的故事内容和发展程度，为用户续写一个引人入胜的互动故事。

### 写作要求
- 请紧接上述情节，依据用户的选择进行续写。你需要展开细节，通过环境描写、心理活动和对话来丰富故事。
- 确保情节发展合理，保持与前文一致的叙述风格、语调和角色性格。
{RESP_PROMPT}

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


class IFData:
    def __init__(self, theme: str, next_correct: Literal[0, 1, 2]):
        self.theme: str = theme
        self.story: list[str] = []
        self.step: int = 0
        # self.finish: int = random.randint(6, 10)
        self.finish = 6
        self.next_correct = next_correct
        self.options: list[str] = []
        self.update_timestamp: float = 0.0


@TOOLS.register(
    "create_interactive_fiction",
    "创建一个互动文游，当用户想要开启一段文游时必须调用此方法",
    {"theme": {"type": "string", "description": "描述文游主题。如未指定则需要向用户询问"}},
    ON_CHAT,
)
async def _(agent: CloversAgent, event: Event, theme: str):
    session = agent.current_session(event)
    if IF_KEY in session.extra and time.time() - (if_data := cast(IFData, session.extra[IF_KEY])).update_timestamp < IF_TIMEOUT:
        return f"当前主题为 {if_data.theme} 的互动文游正在进行中，请勿重复创建。"
    api = agent.api(IF_KEY)
    payload = api.build_payload(({"role": "user", "content": f"请以 {theme} 为主题创建一个故事开头"},), CREATE_IF_PROMPT)
    payload["response_format"] = {"type": "json_object"}
    try:
        resp = await api.call_api(payload, session.usage_counter)
        resp_data = json.loads(resp["content"])
        story: str = resp_data["content"]
        options: list[str] = resp_data["options"]
        if_data = IFData(theme, random.randint(0, 2))  # type: ignore
        if_data.story.append(story)
        if_data.options = options
        a, b, c = options
        session.extra[IF_KEY] = if_data
        group_id = event.group_id
        PLUGIN.temp_handle(
            ["group_id"],
            timeout=session.memory_timeout,
            rule=lambda e: e.group_id == group_id,
            state=(agent, if_data),
        )(interactive_fiction)
        session.complete(f"{story}\n请输入 A,B,C 选择剧情分支\nA: {a}\nB: {b}\nC: {c}")
        return ""
    except Exception as e:
        logger.exception(e)
        return "游戏创建失败，请重新开始。"


async def interactive_fiction(event: Event, handle: TempHandle):
    try:
        agent, data = cast(tuple[CloversAgent, IFData], handle.state)
        char = event.message[0].upper()
        session = agent.current_session(event)
        try:
            index = "ABC".index(char)
        except ValueError:
            if data.update_timestamp < IF_TIMEOUT:
                session.unit_prompts.append(f"用户进行了以 {data.theme} 为主题的互动文游。当前剧情\n{data.story[-1]}")
            return
        opt = data.options[index]
        content = f"目前的故事进度为\n{"\n".join(data.story)}\n用户的选择为\n{opt}。"
        data.update_timestamp = time.time()
        api = agent.api(IF_KEY)
        if data.step > 1 and index != data.next_correct:
            payload = api.build_payload(({"role": "user", "content": content},), BE_IF_PROMPT)
            resp = await api.call_api(payload, session.usage_counter)
            return Result("text", resp["content"].strip())
        data.step += 1
        if data.step < data.finish:
            payload = api.build_payload(
                ({"role": "user", "content": f"{content}\n\n当前发展程度为 {data.step}/{data.finish}"},),
                KEEP_IF_PROMPT,
            )
            payload["response_format"] = {"type": "json_object"}
            resp = await api.call_api(payload, session.usage_counter)
            resp_data = json.loads(resp["content"].strip())
            story: str = resp_data["content"]
            options: list[str] = resp_data["options"]
            a, b, c = options
            data.story.append(opt)
            data.story.append(story)
            data.options = options
            data.next_correct = random.randint(0, 2)
            return Result("text", f"{story}\n\n请输入 A,B,C 选择剧情分支\nA: {a}\nB: {b}\nC: {c}")
        del session.extra[IF_KEY]
        handle.finish()
        payload = api.build_payload(({"role": "user", "content": content},), HE_IF_PROMPT)
        resp = await api.call_api(payload, session.usage_counter)
        return Result("text", resp["content"].strip())
    except Exception as e:
        logger.exception(e)
        return Result("text", "互动文游发生错误，请重试。")
    finally:
        if session.usage_counter:
            deep_add(agent.usage_counter, session.usage_counter)
            session.usage_counter.clear()
            usage = {k: v.get("total_tokens") for k, v in session.usage_counter.items()}
            logger.info(f"[{agent.name}][IF_USAGE] {usage}")
            agent.save_usage()
