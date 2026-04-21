from pathlib import Path
from clovers_agent import Event
from clovers_agent.core import CloversAgent
from clovers.logger import logger
from .docker import WORKSPACE, Shell
from ..toolkit import TOOLS, CONFIG

README = WORKSPACE / "README.md"
shell_dict: dict[str, Shell] = {}


def get_session_id(agent: CloversAgent, event: Event) -> str: ...


if CONFIG.session_workspace:
    get_session_id = lambda agent, event: agent.session_id(event)
else:
    get_session_id = lambda agent, event: "public"


@TOOLS.on_skill("工作区工具")
async def _(agent: CloversAgent, event: Event):
    if not WORKSPACE.exists():
        WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not README.exists():
        README.write_text("Clovers Agent Workspace")
    if CONFIG.use_shell:
        session_id = get_session_id(agent, event)
        if session_id in shell_dict:
            shell_dict[session_id].workdir = "/workspace"
        else:
            try:
                shell_dict[session_id] = Shell(session_id)
            except Exception as e:
                logger.error(e)
                return f"workspace 已初始化, shell 初始化失败:{e}\n当前工作目录: /workspace"
        return f"workspace 已初始化，当前系统：Debian\n当前工作目录: /workspace"
    else:
        return f"workspace 已初始化\n当前工作目录: /workspace"


if CONFIG.use_shell:

    @TOOLS.register(
        "shell",
        "在工作区环境下执行命令",
        {"command": {"type": "string", "description": "需要执行的命令，如需要执行多条命令，请使用 `&&` 或 `;`隔开"}},
        ["工作区工具"],
    )
    async def _(agent: CloversAgent, event: Event, command: str):
        session_id = get_session_id(agent, event)
        if session_id not in shell_dict:
            return f"Error: shell 初始化失败，请返回故障原因。在故障排除前不要重复调用此方法。"
        shell = shell_dict[session_id]
        logger.info(f"[CloversAgentShell][{session_id}]> {command if (idx := command.find("\n")) == -1 else f"{command[:idx]}..."}")
        output = await shell.execute(command)
        return f"{output}\n当前工作目录: {shell.workdir}"


def read_text(file: Path):
    for encoding in ["utf-8", None, "ansi"]:
        try:
            return file.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            return ""


@TOOLS.register(
    "read_files",
    "读取并查看指定文件的内容。支持同时传入多个路径以一次性查看多个文件上下文。"
    "在需要分析代码、检查配置文件时，尤其是需要查看多个文件时，应优先使用此工具以提高效率。",
    {"filepaths": {"type": "array", "description": "包含一个或多个文件路径的数组", "items": {"type": "string"}}},
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, filepaths: list[str]):
    session_id = get_session_id(agent, event)
    workspace = WORKSPACE / session_id
    md = []
    for file_path in filepaths:
        if file_path.startswith("/workspace"):
            file = workspace / f"./{file_path[10:]}"
        else:
            file = workspace / file_path
        if not file.is_relative_to(workspace):
            md.append(f"{file_path} 非工作区文件")
        if not file.exists():
            md.append(f"{file_path} 文件不存在")
            continue
        md.append(f"```{file_path}\n{read_text(file)}\n```")
    return "\n\n".join(md)


@TOOLS.register(
    "write_file",
    "写入文件",
    {
        "file_path": {"type": "string", "description": "需要写入的文件路径"},
        "file_content": {"type": "string", "description": "需要写入到文件的内容"},
    },
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, file_path: str, file_content: str):
    session_id = get_session_id(agent, event)
    workspace = WORKSPACE / session_id
    if file_path.startswith("/workspace"):
        file = workspace / f"./{file_path[10:]}"
    else:
        file = workspace / file_path
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        file.write_text(file_content, encoding="utf-8")
        return f"文件写入成功。"
    except Exception as e:
        logger.error(e)
        return f"文件写入失败：{e}"


@TOOLS.register(
    "upload_file",
    "把文件上传给用户。工作区对用户透明，如用户要求助手发送文件则必须使用此工具。",
    {"file_path": {"type": "string", "description": "需要上传的的文件路径"}},
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, file_path: str):
    session_id = get_session_id(agent, event)
    workspace = WORKSPACE / session_id
    if file_path.startswith("/workspace"):
        file = workspace / f"./{file_path[10:]}"
    else:
        file = workspace / file_path
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
