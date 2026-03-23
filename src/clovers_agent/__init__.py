import httpx
from clovers import Plugin, Result
from clovers.logger import logger
from .core import ToolManager, CloversAgent
from .typing import Event
from .config import Config

__plugin__ = Plugin(priority=100)
__plugin__.set_protocol("properties", Event)
__config__ = Config.sync_config()

agent: CloversAgent


type Rule = Plugin.Rule.Checker[Event]
whitelist = set(__config__.whitelist)
blacklist = set(__config__.blacklist)
console_mode = __config__.console_mode

switch_check: Rule
if console_mode:
    switch_check = lambda e: True
else:
    if whitelist:
        logger.info(f"[CloversAgent] 检查规则设置为白名单模式：{whitelist}")
        switch_check = lambda e: (e.group_id is not None) and (e.group_id in whitelist)
    elif blacklist:
        logger.info(f"[CloversAgent] 检查规则设置为黑名单模式：{blacklist}")
        switch_check = lambda e: (e.group_id is not None) and (e.group_id not in blacklist)
    else:
        logger.info(f"[CloversAgent] 检查规则设置为 False 模式")
        switch_check = lambda e: False

permission_check: Rule = lambda e: e.permission > 0
to_me_check: Rule = lambda e: e.to_me or "extra_context" in e.properties
args_check: Rule = lambda e: bool(e.args)


@__plugin__.startup
async def _():
    global agent
    agent = CloversAgent("CloversAgent", httpx.AsyncClient(timeout=300), __config__)


@__plugin__.handle(
    None,
    ["user_id", "group_id", "nickname", "to_me", "image_list"],
    rule=switch_check,
    priority=2,
    block=False,
)
async def _(event: Event):
    result = await agent.chat(event)
    return Result("text", result) if result else None


@__plugin__.handle(
    ["记忆清除"],
    ["user_id", "group_id", "to_me", "permission"],
    rule=[switch_check, permission_check, to_me_check],
    block=True,
)
async def _(event: Event):
    session = agent.current_session(event)
    async with session.lock:
        session.clear()
    return Result("text", "记忆已清除")


if console_mode:
    console_protocol = b"\x05\x03\x01".decode()

    @__plugin__.handle(
        [f"{console_protocol}"],
        ["user_id", "group_id"],
        rule=[switch_check, args_check],
        block=True,
    )
    async def _(event: Event):
        match event.args[0]:
            case "cleanup":
                session = agent.current_session(event)
                async with session.lock:
                    session.clear()
                return Result("console", ["log", "记忆已清除"])
            case "title":
                prompt = f"给这句话生成一个标题，长度不超过20个字。禁止输出标题以外的内容。\n{event.message[9:]}"
                payload = agent.auxiliary.build_payload([{"role": "user", "content": prompt}])
                return Result("console", ["title", (await agent.auxiliary.call_api(payload))["content"].strip()])
            case _:
                return


__version__ = "0.1.0"
__all__ = ["ToolManager", "CloversAgent", "Event", "__plugin__"]
