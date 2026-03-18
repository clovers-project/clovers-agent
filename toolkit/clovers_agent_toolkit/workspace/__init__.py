from clovers_agent import Event, CloversAgent
from clovers.logger import logger
from .docker import WORKSPACE, Shell
from ..toolkit import toolkit


README = WORKSPACE / "README.md"
shell_dict: dict[str, Shell] = {}


@toolkit.on_skill("工作区工具")
async def _(agent: CloversAgent, event: Event):
    if not WORKSPACE.exists():
        WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not README.exists():
        README.write_text("Clovers Agent Workspace")
    session_id = agent.session_id(event)
    if session_id in shell_dict:
        shell_dict[session_id].workdir = "/workspace"
    else:
        try:
            shell_dict[session_id] = Shell(session_id)
        except Exception as e:
            logger.error(e)
            return f"workspace 已初始化, shell 初始化失败:{e}\n当前工作目录: /workspace"
    return f"workspace 已初始化，当前系统：Debian\n当前工作目录: /workspace"


@toolkit.tool(
    "shell",
    "在工作区环境下执行命令",
    {"command": {"type": "string", "description": "需要执行的命令，如需要执行多条命令，请使用 `&&` 或 `;`隔开"}},
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, command: str):
    session_id = agent.session_id(event)
    if session_id not in shell_dict:
        return f"Error: shell 初始化失败，请返回故障原因。在故障排除前不要重复调用此方法。"
    shell = shell_dict[session_id]
    logger.info(f"[CloversAgentTookit][{session_id}]执行命令: {command}")
    output = await shell.execute(command)
    return f"{output}\n当前工作目录: {shell.workdir}"


@toolkit.tool(
    "read_file",
    "读取文件",
    {"file_path": {"type": "string", "description": "需要读取的文件名"}},
    ["工作区工具"],
)
async def _(agent: CloversAgent, event: Event, file_path: str):
    session_id = agent.session_id(event)
    if session_id in shell_dict:
        workdir = shell_dict[session_id].workdir.lstrip("/")
    else:
        workdir = "workspace"
    workspace = WORKSPACE / session_id
    file = workspace / workdir / file_path
    if not file.is_relative_to(workspace):
        return """status:error
message:不可访问工作区外文件"""
    try:
        return f"""status:success
file:\n{file.read_text()}"""
    except Exception as e:
        return f"""status:error
message:{e}"""


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
    session_id = agent.session_id(event)
    if session_id in shell_dict:
        workdir = shell_dict[session_id].workdir.lstrip("/")
    else:
        workdir = "workspace"
    workspace = WORKSPACE / session_id
    file = workspace / workdir / file_path
    try:
        file.write_text(file_content, encoding="utf-8")
        return f"文件写入成功。"
    except Exception as e:
        return f"文件写入失败：{e}"
