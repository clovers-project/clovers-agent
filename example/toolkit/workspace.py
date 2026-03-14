import asyncio
from httpx import AsyncClient
from pathlib import Path
from clovers_agent import Event, CloversAgent
from .toolkit import toolkit

WORKSPACE = Path("workspace")
GITIGNORE = WORKSPACE / ".gitignore"
README = WORKSPACE / "README.md"
GIT = WORKSPACE / ".git"
workspace_dict: dict[str, Path] = {}


def current_dir(session_id: str):
    if session_id not in workspace_dict:
        workspace = WORKSPACE / session_id
        workspace_dict[session_id] = workspace
    else:
        workspace = workspace_dict[session_id]
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


async def list_dir(dir: Path):
    process = await asyncio.create_subprocess_shell(
        f"git ls-files --others --exclude-standard -c",
        cwd=dir.as_posix(),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    result = stdout.decode("gbk").strip()
    return f"所有文件:\n{result}" if result else "目录为空"


@toolkit.on_skill("工作区工具")
async def _(agent: CloversAgent, event: Event):
    if not WORKSPACE.exists():
        WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not GITIGNORE.exists():
        resp = await agent.async_client.get("https://raw.githubusercontent.com/github/gitignore/refs/heads/main/Python.gitignore")
        GITIGNORE.write_text(resp.text)
    if not README.exists():
        README.write_text("CloversAgent 工作空间")
    if not GIT.exists():
        process = await asyncio.create_subprocess_shell(
            f"git init",
            cwd=str(WORKSPACE),
            stdout=asyncio.subprocess.DEVNULL,
        )
        await process.communicate()
    session_id = agent.session_id(event)
    workspace = WORKSPACE / session_id
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    return f"工作空间已初始化。\n初始工作目录: /"


@toolkit.tool(
    "get_pwd",
    "获取当前所在的工作目录路径",
    None,
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event):
    session_id = agent.session_id(event)
    workspace = WORKSPACE / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    if session_id not in workspace_dict:
        workspace_dict[session_id] = workspace
        return f"当前目录: /"
    current_dir = workspace_dict[session_id]
    rel_path = current_dir.relative_to(workspace).as_posix()
    return f"当前目录: {"/" if rel_path == '.' else f"/{rel_path}"}"


@toolkit.tool(
    "list_dir",
    "列出沙盒环境内的所有文件",
    None,
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event):
    session_id = agent.session_id(event)
    workspace = WORKSPACE / session_id
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    return await list_dir(workspace)


@toolkit.tool(
    "ch_dir",
    "切换工作目录，会自动创建不存在的目录，路径均相对于沙盒环境。支持绝对路径（以'/'开头，代表沙盒根目录）,相对路径（相对于当前工作目录）和'~'（返回沙盒根目录）。",
    {"path": {"type": "string", "description": "目标路径"}},
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, path: str):
    session_id = agent.session_id(event)
    workspace = WORKSPACE / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    path = path.strip()
    if path == "~":
        workspace_dict[session_id] = workspace
        return f"切换目录成功。\n当前目录 /"
    elif path.startswith("/"):
        target = workspace / path[1:]
    else:
        if session_id not in workspace_dict:
            workspace_dict[session_id] = workspace
        target = workspace_dict[session_id] / path
    if not target.is_relative_to(workspace):
        return f"切换目录失败：禁止访问沙盒外部位置。"
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    elif not target.is_dir():
        return f"切换目录失败：目录是一个文件。"
    workspace_dict[session_id] = target
    rel_path = target.relative_to(workspace).as_posix()
    return f"切换目录成功。\n当前目录: {"/" if rel_path == '.' else f"/{rel_path}"}"


@toolkit.tool(
    "read_file",
    "读取文件",
    {"file_path": {"type": "string", "description": "需要读取的文件名"}},
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, file_path: str):
    file = current_dir(agent.session_id(event)) / file_path
    if not file.exists():
        return f"文件不存在"
    try:
        raw_data = file.read_bytes()
        try:
            return raw_data.decode("utf-8")
        except UnicodeDecodeError:
            import chardet

            return raw_data.decode(encoding=chardet.detect(raw_data)["encoding"] or "ascii")
    except Exception as e:
        return f"读取文件失败：{e}"


@toolkit.tool(
    "write_file",
    "写入文件",
    {
        "file_path": {"type": "string", "description": "需要写入的文件名"},
        "file_content": {"type": "string", "description": "需要写入到文件的内容"},
    },
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, file_path: str, file_content: str):
    file = current_dir(agent.session_id(event)) / file_path
    try:
        file.write_text(file_content, encoding="utf-8")
        return f"文件写入成功。"
    except Exception as e:
        return f"文件写入失败：{e}"


# @toolkit.tool(
#     "shell",
#     f"执行命令。使用当前系统({platform.system()})执行一条命令。注意：本方法只能运行一条非交互式命令。请务必优先使用其他工具。",
#     {"command": {"type": "string", "description": "shell命令"}},
#     ["工作区工具"],
# )
# async def _(agent: CloversAgent, event: Event, command: str):
#     await init_workspace(agent.async_client)
#     session_id = agent.session_id(event)
#     if session_id not in workspace_dict:
#         workspace = WORKSPACE / session_id
#         workspace_dict[session_id] = workspace
#     else:
#         workspace = workspace_dict[session_id]
#     workspace.mkdir(parents=True, exist_ok=True)
#     process = await asyncio.create_subprocess_shell(
#         command,
#         cwd=str(workspace),
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE,
#         stdin=asyncio.subprocess.DEVNULL,
#     )
#     try:
#         stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
#         return f"OUT: {stdout.decode()}\nERR: {stderr.decode()}"
#     except asyncio.TimeoutError:
#         process.kill()
#         await process.wait()
#         return "错误：命令执行超时（>60 秒），进程已终止。"
#     except Exception as e:
#         return f"错误：{e}"
