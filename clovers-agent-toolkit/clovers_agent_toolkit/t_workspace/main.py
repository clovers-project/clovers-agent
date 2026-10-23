from pathlib import Path
from datetime import datetime
from clovers_agent import CloversAgent, Turn, Event
from clovers_agent.session import Session
from clovers_agent.utils import decode_bytes
from clovers_agent.config import CONFIG as AGENT_CONFIG
from clovers_agent.constants import HIDDEN_CATEGORY, SKILL_MENU
from clovers.logger import logger
from ..base import TOOLS, CONFIG, CATEGORIES, retry_request
from .turn_batch import TurnBatch
from .utils import format_path, format_size, md5
from .constants import WORKSPACE, KEY_SUBAGENTS, KEY_SUBAGENTS, SUBAGENT_PROMPT

from typing import cast

WORKSPACE_PATH = Path(AGENT_CONFIG.path) / "workspace"
DOCKER_IMAGE = CONFIG.docker_image
MAX_SUBAGENT_COUNT = CONFIG.max_subagent_count
MAX_DOWNLOAD_RETURN_SIZE = CONFIG.max_download_return_size
SUCCESS_TOOLIST = ("execute_command", "run_subagent")
FALLBACK_TOOLIST = ("ls", "read_files", "write_file")


async def enter_workspace(agent: CloversAgent, session: Turn):
    prompts = ["# workspace 已初始化\n"]
    if CONFIG.use_shell:
        try:
            if "shell" in session.extra:
                shell = session.extra["shell"]
            else:
                from .docker import Shell

                shell = Shell(session.session_id, WORKSPACE_PATH, DOCKER_IMAGE)
                session.extra["shell"] = shell
            await shell.activate()
            session.extra["shell"].workdir = "/workspace"
            session.extend_tools(agent.manifest[x] for x in SUCCESS_TOOLIST)
            prompts.append(f"当前系统：Debian\n当前工作目录: /workspace")
            return "\n".join(prompts)
        except Exception as e:
            logger.error(e)
    session.extend_tools(agent.manifest[x] for x in FALLBACK_TOOLIST)
    prompts.append(f"当前工作目录: /workspace")
    return "\n".join(prompts)


@TOOLS.register("on_task", "当前用户指令为一个需要调用工具的任务时，调用此工具")
async def on_task(agent: CloversAgent, session: Turn, event: Event):
    session.create_payload(agent.api("on_task"), agent.style_prompt)
    workspace = WORKSPACE_PATH / session.session_id
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    readme = workspace / "README.md"
    if readme.exists():
        session.unit_prompts.append(decode_bytes(readme.read_bytes()))
    session.payload["tools"] = [
        agent.manifest[SKILL_MENU],
        *agent.select_tools(CATEGORIES.WORKSPACE),
        *agent.select_tools(CATEGORIES.NETWORK),
    ]
    await enter_workspace(agent, session)
    return ""


@TOOLS.create_category(WORKSPACE, "包含文件读写与发送、命令执行等工具，用于进行相关工作。")
async def workspace_category(agent: CloversAgent, session: Turn, event: Event):
    return await enter_workspace(agent, session)


@TOOLS.register(
    "upload_file",
    "把文件上传给用户。工作区可能对用户不可见，如你想让用户收到文件则必须使用此工具。",
    {"file_path": {"type": "string", "description": "需要上传的的文件路径"}},
    WORKSPACE,
)
async def _(agent: CloversAgent, session: Turn, event: Event, file_path: str):
    workspace = WORKSPACE_PATH / session.session_id
    file = workspace / format_path(file_path)
    if not file.is_relative_to(workspace):
        return f"{file_path} 非工作区文件"
    if not file.exists():
        return f"{file_path} 文件不存在"
    if (coro := event.send("file", file)) is None:
        return "未实现上传文件接口"
    try:
        await coro
    except Exception as e:
        return f"上传失败：{e}"
    return "上传成功！"


@TOOLS.register(
    "download_file",
    "通过 HTTP GET/POST 请求下载 URL 资源。",
    {
        "method": {
            "type": "string",
            "description": "HTTP 请求方法：get 用于获取资源，post 用于提交数据",
            "enum": ["get", "post"],
        },
        "url": {
            "type": "string",
            "description": "要请求的完整 URL，需包含协议（http:// 或 https://）",
        },
        "headers": {
            "type": "object",
            "description": '自定义请求头，如 {"Authorization": "Bearer xxx", "User-Agent": "..."}；一般可不填',
        },
        "data": {
            "type": "object",
            "description": "请求参数。method=get 时作为 URL 查询参数；method=post 时作为 JSON 请求体发送",
        },
        "file_path": {
            "type": "string",
            "description": '保存文件到工作区的文件名，如 "logo.png"。不填则把响应内容直接作为工具输出返回。',
        },
    },
    WORKSPACE,
    required=["method", "url"],
)
async def _(
    agent: CloversAgent,
    session: Turn,
    event: Event,
    method: str,
    url: str,
    headers: dict | None = None,
    data: dict | None = None,
    file_path: str | None = None,
):
    if method.lower() == "post":
        resp = await retry_request(agent.async_client.post, url, json=data, headers=headers)
    else:
        resp = await retry_request(agent.async_client.get, url, params=data, headers=headers)
    content = resp.content
    if not content:
        return f"请求成功（HTTP {resp.status_code}），但响应内容为空。"
    r_size = len(content)
    s_size = format_size(r_size)
    # 指定了保存路径：写入工作区
    if file_path:
        file = WORKSPACE_PATH / session.session_id / format_path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
        return f'文件 "{file_path}" 保存成功（大小：{s_size}，HTTP {resp.status_code}）。'
    # 未指定保存路径，但内容过大：拦截并落盘
    if r_size > MAX_DOWNLOAD_RETURN_SIZE:
        saved_name = f"{md5(content)}.bin"
        file = WORKSPACE_PATH / session.session_id / saved_name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)  # 原代码遗漏，修复
        return f"响应内容过大（{s_size}），未直接返回。内容已保存至工作区文件 {saved_name}。"
    return decode_bytes(content)


@TOOLS.register(
    "execute_command",
    "在工作区环境下执行命令",
    {"command": {"type": "string", "description": "需要执行的命令，如需要执行多条命令，请使用 `&&` 或 `;`隔开"}},
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, session: Turn, event: Event, command: str):
    from .docker import Shell

    extra = session.extra
    if "shell" not in extra or not isinstance(shell := extra["shell"], Shell):
        return f"Error: shell 初始化失败，请返回故障原因。在故障排除前不要重复调用此方法。"
    output = await shell.execute(command)
    return f"{output}\n当前工作目录: {shell.workdir}"


@TOOLS.register(
    "run_subagent",
    "创建一个子代理执行你的指令\n子代理会在本次调用链结束后返回结果并唤醒你",
    {"command": {"type": "string", "description": "指令"}},
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, session: Turn, event: Event, command: str):
    extra = session.extra
    if not KEY_SUBAGENTS in extra:
        extra[KEY_SUBAGENTS] = {}
    batchs = cast(dict[str, TurnBatch], extra[KEY_SUBAGENTS])
    session_id = session.session_id
    if session_id in batchs:
        batch = batchs[session_id]
        if len(batch.tasklist) > MAX_SUBAGENT_COUNT:
            return f"子代理数量已达到上限，请等待其他子代理完成。"
    else:
        if not isinstance(session, Session):
            return f"只有主会话才能初始化子代理"
        batch = batchs[session_id] = TurnBatch(agent, session, event)
        batch.start().add_done_callback(lambda _: batchs.pop(session_id, None))

    s = session.create_turn()
    s.create_payload(session.api, SUBAGENT_PROMPT)
    s.payload["tools"] = [
        agent.manifest[SKILL_MENU],
        *agent.select_tools(CATEGORIES.WORKSPACE),
        *agent.select_tools(CATEGORIES.NETWORK),
    ]
    s.unit_prompts.append(await enter_workspace(agent, s))
    s.input_text(command)
    batch.submit(s, event)
    return "OK"


@TOOLS.register(
    "ls",
    "查看工作区文件",
    {"path": {"type": "string", "description": "需要查看的目录路径"}},
    HIDDEN_CATEGORY,
    [],
)
async def _(agent: CloversAgent, session: Turn, event: Event, path: str = "./"):
    folder = WORKSPACE_PATH / session.session_id / format_path(path)
    if not folder.exists():
        return f"路径 '{path}' 不存在。"
    if not folder.is_dir():
        return f"'{path}' 不是目录。"
    lines = [
        f"`{path}` 下的文件列表：\n",
        "kind|size|mtime|name| ",
        "---|---|---|---",
    ]
    entries = sorted(folder.glob("*"), key=lambda f: (f.is_file(), f.name))
    for file in entries:
        name = file.name
        try:
            mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            if file.is_file():
                ks = f"f|{format_size(file.stat().st_size)}"
            else:
                ks = "d|-"
            lines.append(f"{ks}|{mtime}|{name}")
        except OSError as e:
            lines.append(f"?|?|?|{name}")
    return "\n".join(lines)


@TOOLS.register(
    "read_file",
    "读取并查看指定文件的内容",
    {"file_path": {"type": "string", "description": "文件路径"}},
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, session: Turn, event: Event, file_path: str):
    workspace = WORKSPACE_PATH / session.session_id
    file = workspace / format_path(file_path)
    if not file.is_relative_to(workspace):
        return f"{file_path} 非工作区文件"
    if not file.exists():
        return f"{file_path} 文件不存在"
    return decode_bytes(file.read_bytes())


@TOOLS.register(
    "write_file",
    "写入文件",
    {
        "file_path": {"type": "string", "description": "需要写入的文件路径"},
        "file_content": {"type": "string", "description": "需要写入到文件的内容"},
    },
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, session: Turn, event: Event, file_path: str, file_content: str):
    file = WORKSPACE_PATH / agent.session_id(event) / format_path(file_path)
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        file.write_text(file_content, encoding="utf-8")
        return f"文件写入成功。"
    except Exception as e:
        logger.error(e)
        return f"文件写入失败：{e}"
