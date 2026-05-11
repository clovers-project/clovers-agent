if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

import random
import time
import json
from clovers import Plugin, Result, TempHandle
from clovers.logger import logger
from clovers_client import Event as EventProtocol
from clovers_agent import AGENT, Event
from typing import cast
from .constant import (
    IF_KEY,
    IF_TIMEOUT,
    CREATE_IF_PROMPT,
    KEEP_IF_PROMPT,
    BE_IF_PROMPT,
    HE_IF_PROMPT,
)

PLUGIN = Plugin[Event](priority=90, build_result=lambda x: Result("text", x))
PLUGIN.protocol = EventProtocol


class IFData:

    def __init__(self, theme: str):
        self.theme: str = theme
        self.story: list[str] = []
        self.step: int = 0
        self.finish: int = random.randint(6, 10)
        self.next_correct: int = 0
        self.options: list[str] = []
        self.update_timestamp: float = 0.0

    def correct(self, index: int) -> bool:
        return self.step > 1 and self.next_correct == index


@PLUGIN.handle(["文游"], ["user_id", "group_id", "nickname", "to_me", "image_list", "at"], rule=lambda e: e.to_me)
async def _(event: Event):
    session = AGENT.current_session(event)
    if IF_KEY in session.extra and time.time() - (data := cast(IFData, session.extra[IF_KEY])).update_timestamp < IF_TIMEOUT:
        return f"当前主题为 {data.theme} 的互动文游正在进行中，请勿重复创建。"
    api = AGENT.api(IF_KEY)
    theme = event.message[2:].strip()
    if not theme:
        return "请指定文游主题。"
    payload = api.build_payload(({"role": "user", "content": f"请以 {theme} 为主题创建一个故事开头"},), CREATE_IF_PROMPT)
    payload["response_format"] = {"type": "json_object"}
    resp = await api.call_api(payload, session.usage_counter)
    resp_data = json.loads(resp["content"])
    story: str = resp_data["content"]
    options: list[str] = resp_data["options"]
    data = IFData(theme)
    data.story.append(story)
    data.options = options
    a, b, c = options
    session.extra[IF_KEY] = data
    group_id = event.group_id
    PLUGIN.temp_handle(
        ["group_id"],
        timeout=session.memory_timeout,
        rule=lambda e: e.group_id == group_id,
        state=data,
    )(interactive_fiction)
    return f"{story}\n请输入 A,B,C 选择剧情分支\nA: {a}\nB: {b}\nC: {c}\nQ: 退出游戏"


async def interactive_fiction(event: Event, handle: TempHandle):
    data = cast(IFData, handle.state)
    char = event.message[0].upper()
    session = AGENT.current_session(event)
    if char == "Q":
        handle.finish()
        if IF_KEY in session.extra:
            del session.extra[IF_KEY]
        return "游戏已结束。"
    timestamp = time.time()
    try:
        index = "ABC".index(char)
    except ValueError:
        if timestamp - data.update_timestamp < IF_TIMEOUT:
            session.unit_prompts.append(f"当前正在进行以 {data.theme} 为主题的互动文游。当前剧情\n{data.story[-1]}")
        return
    opt = data.options[index]
    data.update_timestamp = timestamp
    api = AGENT.api(IF_KEY)
    if data.correct(index):
        payload = api.build_payload(system_prompt=BE_IF_PROMPT)
        payload["messages"].append({"role": "user", "content": f"当前剧情\n{"\n".join(data.story)}\n用户的选择\n{opt}。"})
        resp = await api.call_api(payload, session.usage_counter)
        return resp["content"].strip()
    data.step += 1
    if data.step < data.finish:
        handle.delay(session.memory_timeout)
        next_correct = random.randint(0, 2)
        payload = api.build_payload(system_prompt=KEEP_IF_PROMPT)
        content = F"""\
当前剧情
{"\n".join(data.story)}
当前发展程度
{data.step}/{data.finish}
用户的选择
{opt}
请将你输出的第 {next_correct + 1} 个选项作为正确的选项，但不要太明显。"""
        payload["messages"].append({"role": "user", "content": content})
        payload["response_format"] = {"type": "json_object"}
        for _ in range(3):
            try:
                resp = await api.call_api(payload, session.usage_counter)
                resp_data = json.loads(resp["content"].strip())
                story: str = resp_data["content"]
                options: list[str] = resp_data["options"]
                a, b, c = options
                break
            except Exception as e:
                logger.exception(e)
            return "获取游戏内容超时，请稍后重试。"
        data.story.append(opt)
        data.story.append(story)
        data.options = options
        data.next_correct = next_correct
        return f"{story}\n\n请输入 A,B,C 选择剧情分支\nA: {a}\nB: {b}\nC: {c}\nQ: 退出游戏"
    handle.finish()
    if IF_KEY in session.extra:
        del session.extra[IF_KEY]
    payload = api.build_payload(system_prompt=HE_IF_PROMPT)
    payload["messages"].append({"role": "user", "content": f"当前剧情\n{"\n".join(data.story)}\n用户的选择\n{opt}。"})
    resp = await api.call_api(payload, session.usage_counter)
    return resp["content"].strip()
