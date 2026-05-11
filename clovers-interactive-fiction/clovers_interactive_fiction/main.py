if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

import random
import time
import json
from clovers import Plugin, Result, TempHandle
from clovers.logger import logger
from clovers_agent import AGENT, Event
from clovers_client import Event as EventProtocol
from clovers_agent.utils import deep_add
from typing import cast
from .constant import (
    IF_KEY,
    IF_TIMEOUT,
    CREATE_IF_PROMPT,
    KEEP_IF_PROMPT,
    BE_IF_PROMPT,
    HE_IF_PROMPT,
)

PLUGIN = Plugin[Event](priority=90)
PLUGIN.protocol = EventProtocol


class IFData:
    next_correct: int

    def __init__(self, theme: str):
        self.theme: str = theme
        self.story: list[str] = []
        self.step: int = 0
        self.finish: int = random.randint(6, 10)
        self.options: list[str] = []
        self.update_timestamp: float = 0.0


@PLUGIN.handle(["文游", "互动文游"], ["user_id", "group_id", "nickname", "to_me", "image_list", "at"], rule=lambda e: e.to_me)
async def _(event: Event):
    session = AGENT.current_session(event)
    if IF_KEY in session.extra and time.time() - (if_data := cast(IFData, session.extra[IF_KEY])).update_timestamp < IF_TIMEOUT:
        return f"当前主题为 {if_data.theme} 的互动文游正在进行中，请勿重复创建。"
    api = AGENT.api(IF_KEY)
    theme = event.message[0]
    if not theme:
        return "请指定文游主题。"
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
            state=if_data,
        )(interactive_fiction)
        session.complete(f"{story}\n请输入 A,B,C 选择剧情分支\nA: {a}\nB: {b}\nC: {c}\nQ: 退出游戏")
        return ""
    except Exception as e:
        logger.exception(e)
        return "游戏创建失败，请重新开始。"


async def interactive_fiction(event: Event, handle: TempHandle):
    try:
        data = cast(IFData, handle.state)
        char = event.message[0].upper()
        session = AGENT.current_session(event)
        if char == "Q":
            if IF_KEY in session.extra:
                del session.extra[IF_KEY]
            handle.finish()
            return Result("text", "游戏已结束。")
        timestamp = time.time()
        try:
            index = "ABC".index(char)
        except ValueError:
            if timestamp - data.update_timestamp < IF_TIMEOUT:
                session.unit_prompts.append(f"用户进行了以 {data.theme} 为主题的互动文游。当前剧情\n{data.story[-1]}")
            return
        opt = data.options[index]
        data.update_timestamp = timestamp
        api = AGENT.api(IF_KEY)
        if data.step > 1 and index != data.next_correct:
            payload = api.build_payload(system_prompt=BE_IF_PROMPT)
            payload["messages"].append({"role": "user", "content": f"当前剧情\n{"\n".join(data.story)}\n用户的选择\n{opt}。"})
            resp = await api.call_api(payload, session.usage_counter)
            return Result("text", resp["content"].strip())
        data.step += 1
        if data.step < data.finish:
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
            else:
                raise TimeoutError("获取游戏内容超时")
            data.story.append(opt)
            data.story.append(story)
            data.options = options
            data.next_correct = next_correct
            return Result("text", f"{story}\n\n请输入 A,B,C 选择剧情分支\nA: {a}\nB: {b}\nC: {c}\nQ: 退出游戏")
        if IF_KEY in session.extra:
            del session.extra[IF_KEY]
        handle.finish()
        payload = api.build_payload(system_prompt=HE_IF_PROMPT)
        payload["messages"].append({"role": "user", "content": f"当前剧情\n{"\n".join(data.story)}\n用户的选择\n{opt}。"})
        resp = await api.call_api(payload, session.usage_counter)
        return Result("text", resp["content"].strip())
    except Exception as e:
        logger.exception(e)
        return Result("text", "互动文游发生错误，请重试。")
    finally:
        if session.usage_counter:
            deep_add(AGENT.usage_counter, session.usage_counter)
            session.usage_counter.clear()
            usage = {k: v.get("total_tokens") for k, v in session.usage_counter.items()}
            logger.info(f"[{AGENT.name}][IF_USAGE] {usage}")
            AGENT.save_usage()
