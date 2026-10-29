from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from clovers_agent import CloversAgent, Turn, Event
from clovers_agent.session import Session
from clovers_agent.api import ALLOWED_IMAGE_SUBTYPES
from clovers_agent.utils import decode_bytes, to_b64url
from clovers_agent.config import CONFIG as AGENT_CONFIG
from clovers_agent.constants import HIDDEN_CATEGORY, INTRINSIC_CATEGORY
from clovers.logger import logger
from ..base import TOOLS, CONFIG, CATEGORIES
from ..utils import retry_request, format_path, format_size, md5, list_entries, render_listing
from .constants import WORKSPACE, SUBAGENT_PROMPT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .docker import Shell

SUCCESS_TOOLIST = "execute_command", "subagent"
FALLBACK_TOOLIST = "fetch_url", "ls", "read_file", "write_file"
WORKSPACE_PATH = Path(AGENT_CONFIG.path) / "workspace"

workspace_config = CONFIG.workspace

DOCKER_IMAGE = workspace_config.docker_image
MAX_SUBAGENT_COUNT = workspace_config.max_subagent_count
MAX_DOWNLOAD_RETURN_SIZE = workspace_config.max_download_return_size
USE_SHELL = workspace_config.use_shell
SHARED_DOCKER_CONTAINER = workspace_config.shared_workspace
if SHARED_DOCKER_CONTAINER:
    shared_shell = None
    SHARED_ID = "shared"

    def session_workspace(session_id: str) -> Path:
        return WORKSPACE_PATH / SHARED_ID

    def session_shell(session: Session) -> Shell:
        from .docker import Shell

        global shared_shell
        if not isinstance(shared_shell, Shell):
            shared_shell = Shell(SHARED_ID, WORKSPACE_PATH, DOCKER_IMAGE)
            extra = session.extra

            async def cleanup():
                global shared_shell
                assert shared_shell
                logger.info("Cleaning up shared workspace...")
                await shared_shell.cleanup()
                shared_shell = None

            if "cleanup" not in extra:
                extra["cleanup"] = [cleanup()]
            else:
                extra["cleanup"].append(cleanup())
        return shared_shell

else:

    def session_workspace(session_id: str) -> Path:
        return WORKSPACE_PATH / session_id

    def session_shell(session: Session) -> Shell:
        from .docker import Shell

        if "shell" in session.extra:
            shell = session.extra["shell"]
        else:
            shell = None
        if not isinstance(shell, Shell):
            shell = Shell(session.session_id, WORKSPACE_PATH, DOCKER_IMAGE)
            extra = session.extra
            extra["shell"] = shell
            if "cleanup" not in extra:
                extra["cleanup"] = [shell.cleanup()]
            else:
                extra["cleanup"].append(shell.cleanup())
        return shell


async def enter_workspace(agent: CloversAgent, turn: Turn):
    prompts = ["# Workspace initialized"]
    if USE_SHELL:
        try:
            await session_shell(agent.current_session(turn.session_id)).activate()
            turn.extend_tools(agent.tool_manifest[x] for x in SUCCESS_TOOLIST)
            prompts.append("Current system: Debian\nCurrent working directory: /workspace")
            return "\n\n".join(prompts)
        except Exception as e:
            logger.error("enter_workspace failed: %s: %s", type(e).__name__, e)
    turn.extend_tools(agent.tool_manifest[x] for x in FALLBACK_TOOLIST)
    prompts.append("Current working directory: /workspace")
    return "\n\n".join(prompts)


@TOOLS.register("on_task", "Call this when the current user request is a task that requires tools.")
async def on_task(agent: CloversAgent, turn: Turn, event: Event):
    turn.create_payload(agent.api("on_task"), agent.style_prompt)
    workspace = session_workspace(turn.session_id)
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    if (readme := workspace / "README.md").exists():
        turn.unit_prompts.append(decode_bytes(readme.read_bytes()))
    elif (readme := WORKSPACE_PATH / "README.md").exists():
        turn.unit_prompts.append(decode_bytes(readme.read_bytes()))
    turn.payload["tools"] = [
        *agent.select_tools(CATEGORIES.WORKSPACE),
        *agent.select_tools(CATEGORIES.NETWORK),
    ]
    turn.unit_prompts.append(await enter_workspace(agent, turn))
    return ""


@TOOLS.create_category(WORKSPACE, "Tools for workspace file and command operations.")
def workspace_category(agent: CloversAgent, session: Turn, event: Event):
    return enter_workspace(agent, session)


@TOOLS.register(
    "read_image",
    "Load an image for visual inspection. Accepts an HTTP(S) URL or a local path.",
    {"path": {"type": "string", "description": 'Image URL (e.g. "https://example.com/a.png") or local path (e.g. "logo.png").'}},
    INTRINSIC_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, path: str):
    path = path.strip()
    if not path:
        return "path must not be empty."
    if path.lower().startswith("http"):
        b64url = await turn.api.download_url(path)
        if not b64url:
            return "Resource not found or unsupported format."
    else:
        file = session_workspace(turn.session_id) / format_path(path)
        if not file.is_file():
            return f'File not found: "{path}"'
        b64url = to_b64url(file.read_bytes(), ALLOWED_IMAGE_SUBTYPES)
        if not b64url:
            return f'Not a recognized image format: "{path}"'
    turn.pending_content.append({"type": "image_url", "image_url": {"url": b64url}})
    return "OK"


@TOOLS.register(
    "upload_file",
    "Send a file to the user. The user may not be able to see the workspace, so a path in your reply is not enough — call this to hand over the file itself.",
    {"file_path": {"type": "string", "description": "Path to the file to send."}},
    WORKSPACE,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, file_path: str):
    workspace = session_workspace(turn.session_id)
    file = workspace / format_path(file_path)
    if not file.is_relative_to(workspace):
        return f'Not a workspace file: "{file_path}"'
    if not file.exists():
        return f'File not found: "{file_path}"'
    if (coro := event.send("file", file)) is None:
        return "File upload is not available in this environment."
    try:
        await coro
    except Exception as e:
        return f"Upload failed: {e}"
    return "File sent to the user."


@TOOLS.register(
    "execute_command",
    "Run a shell command in the workspace.",
    {"command": {"type": "string", "description": "Command to run. Chain multiple commands with `&&` or `;`."}},
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, command: str):
    try:

        return await session_shell(agent.current_session(turn.session_id)).execute(command)
    except Exception:
        return "Error: shell initialization failed. Report this to the user; do not call this tool again until it is resolved."


@TOOLS.register(
    "subagent",
    f"Delegate a self-contained task to a subagent (a separate agent that executes in its own context) to offload focused, independent work — research, a scoped implementation, or analysis — without consuming this conversation's context. The subagent runs asynchronously; its result is not returned immediately but delivered through a wakeup message after the current call chain ends. After delegating, continue with the remaining work or end the current call chain to allow the result to be delivered. You can run up to {MAX_SUBAGENT_COUNT} subagents per call chain.",
    {
        "prompt": {
            "type": "string",
            "description": "The complete, self-contained task for the subagent, including the goal, relevant files or paths, any constraints, and the expected output format.",
        }
    },
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, prompt: str):
    session = agent.current_session(turn.session_id)
    if session.wakeup_context and len(session.wakeup_context.tasks) > MAX_SUBAGENT_COUNT:
        return f"Subagent limit reached. Slots are released only when the current call chain ends — do not create more subagents in this chain."
    sub_turn = turn.create_turn()
    sub_turn.create_payload(turn.api, SUBAGENT_PROMPT)
    tools = sub_turn.payload["tools"] = [
        *agent.select_tools(INTRINSIC_CATEGORY),
        *agent.select_tools(CATEGORIES.WORKSPACE),
        *agent.select_tools(CATEGORIES.NETWORK),
    ]
    sub_turn.unit_prompts.append(await enter_workspace(agent, sub_turn))
    if tool := next((x for x in tools if x["function"]["name"] == "subagent"), None):
        sub_turn.payload["tools"].remove(tool)
    sub_turn.input_text(prompt)
    agent.submit_turn_for_wakeup(sub_turn, event)
    return "OK"


@TOOLS.register(
    "fetch_url",
    "Fetch a URL using an HTTP GET or POST request. Set file_path to save the response to the workspace; otherwise the response body is returned as the tool output.",
    {
        "method": {"type": "string", "description": "HTTP method: GET to fetch a resource, POST to submit data.", "enum": ["GET", "POST"]},
        "url": {"type": "string", "description": "Absolute HTTP or HTTPS URL, e.g. https://example.com/path."},
        "headers": {"type": "object", "description": 'Custom request headers, e.g. {"Authorization": "Bearer xxx"}. Usually omitted.'},
        "data": {"type": "object", "description": "Request data: query parameters for get, JSON body for post."},
        "file_path": {"type": "string", "description": 'Path to save the response to, e.g. "logo.png".'},
    },
    HIDDEN_CATEGORY,
    required=["method", "url"],
)
async def _(
    agent: CloversAgent,
    turn: Turn,
    event: Event,
    method: str,
    url: str,
    headers: dict | None = None,
    data: dict | None = None,
    file_path: str | None = None,
):
    if method.upper() == "POST":
        resp = await retry_request(agent.async_client.post(url, json=data, headers=headers))
    else:
        resp = await retry_request(agent.async_client.get(url, params=data, headers=headers))
    content = resp.content
    if not content:
        return f"Request succeeded (HTTP {resp.status_code}), but the response body was empty."
    r_size = len(content)
    s_size = format_size(r_size)
    # 指定了保存路径：写入工作区
    if file_path:
        file = session_workspace(turn.session_id) / format_path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
        return f'Saved to "{file_path}" ({s_size}, HTTP {resp.status_code}).'
    # 未指定保存路径，但内容过大：拦截并落盘
    if r_size > MAX_DOWNLOAD_RETURN_SIZE:
        saved_name = f"{md5(content)}.bin"
        file = session_workspace(turn.session_id) / saved_name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
        return f'Response too large to return ({s_size}); saved to the workspace as "{saved_name}".'
    return decode_bytes(content)


@TOOLS.register(
    "ls",
    "List files and directories under a workspace path. Lists direct children by default; set depth to recurse.",
    {
        "path": {"type": "string", "description": 'Defaults to "./".'},
        "depth": {"type": "integer", "description": "Levels to list: 1 lists direct children only. Defaults to 1."},
    },
    HIDDEN_CATEGORY,
    [],
)
async def _(agent: CloversAgent, turn: Turn, event: Event, path: str = "./", depth: int = 1):
    workspace = session_workspace(turn.session_id)
    relative = format_path(path)
    folder = workspace / relative
    if not folder.is_relative_to(workspace):
        return f'Not a workspace folder: "{path}"'
    depth = max(1, depth)
    try:
        entries = list_entries(folder, depth)
    except FileNotFoundError:
        return f'Path not found: "{path}"'
    except NotADirectoryError:
        return f'Not a directory: "{path}"'
    except OSError as exc:
        return f'Cannot list "{path}": {exc.strerror or exc}'
    return render_listing(path, depth, entries)


@TOOLS.register(
    "read_file",
    "Read a text file from the workspace.",
    {"file_path": {"type": "string"}},
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, file_path: str):
    workspace = session_workspace(turn.session_id)
    file = workspace / format_path(file_path)
    if not file.is_relative_to(workspace):
        return f'Not a workspace file: "{file_path}"'
    if not file.exists():
        return f'File not found: "{file_path}"'
    content = file.read_bytes()
    if b"\x00" in content or not (text := decode_bytes(content)):
        return f'Unsupported file type: "{file_path}"'
    return text


@TOOLS.register(
    "write_file",
    "Write text to a file",
    {
        "file_path": {"type": "string"},
        "content": {"type": "string", "description": "Full content to write"},
    },
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, turn: Turn, event: Event, file_path: str, content: str):
    workspace = session_workspace(turn.session_id)
    file = workspace / format_path(file_path)
    if not file.is_relative_to(workspace):
        file_name = f"{file.name}.{uuid4().hex}"
        file = workspace / file_name
        workspace.mkdir(parents=True, exist_ok=True)
        tip = f'File written to the workspace as "{file_name}" — the requested path is outside the workspace.'
    else:
        file.parent.mkdir(parents=True, exist_ok=True)
        tip = "File written."
    try:
        file.write_text(content, encoding="utf-8")
        return tip
    except Exception as e:
        return f"Failed to write file: {type(e).__name__}: {e}"
