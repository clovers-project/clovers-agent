from pathlib import Path
from uuid import uuid4
from clovers_agent import CloversAgent, Turn, Event
from clovers_agent.api import ALLOWED_IMAGE_SUBTYPES
from clovers_agent.session import Session
from clovers_agent.utils import decode_bytes, to_b64url
from clovers_agent.config import CONFIG as AGENT_CONFIG
from clovers_agent.constants import HIDDEN_CATEGORY, INTRINSIC_CATEGORY
from clovers.logger import logger
from ..base import TOOLS, CONFIG, CATEGORIES
from ..utils import retry_request, format_path, format_size, md5, list_entries, render_listing
from .turn_batch import TurnBatch
from .constants import WORKSPACE, KEY_SUBAGENTS, KEY_SUBAGENTS, SUBAGENT_PROMPT

from typing import cast

workspace_config = CONFIG.workspace

DOCKER_IMAGE = workspace_config.docker_image
MAX_SUBAGENT_COUNT = workspace_config.max_subagent_count
MAX_DOWNLOAD_RETURN_SIZE = workspace_config.max_download_return_size
USE_SHELL = workspace_config.use_shell

SUCCESS_TOOLIST = "execute_command", "subagent"
FALLBACK_TOOLIST = "fetch_url", "ls", "read_file", "write_file"
WORKSPACE_PATH = Path(AGENT_CONFIG.path) / "workspace"


async def enter_workspace(agent: CloversAgent, session: Turn):
    prompts = ["# Workspace initialized"]
    if USE_SHELL:
        try:
            from .docker import Shell

            if "shell" in session.extra:
                shell = session.extra["shell"]
            else:
                shell = None
            if not isinstance(shell, Shell):
                shell = Shell(session.session_id, WORKSPACE_PATH, DOCKER_IMAGE)
                session.extra["shell"] = shell
            await shell.activate()
            session.extra["shell"].workdir = "/workspace"
            session.extend_tools(agent.manifest[x] for x in SUCCESS_TOOLIST)
            prompts.append("Current system: Debian\nCurrent working directory: /workspace")
            return "\n\n".join(prompts)
        except Exception as e:
            logger.error("enter_workspace failed: %s: %s", type(e).__name__, e)
    session.extend_tools(agent.manifest[x] for x in FALLBACK_TOOLIST)
    prompts.append("Current working directory: /workspace")
    return "\n\n".join(prompts)


@TOOLS.register("on_task", "Call this when the current user request is a task that requires tools.")
async def on_task(agent: CloversAgent, session: Turn, event: Event):
    session.create_payload(agent.api("on_task"), agent.style_prompt)
    workspace = WORKSPACE_PATH / session.session_id
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    if (readme := workspace / "README.md").exists():
        session.unit_prompts.append(decode_bytes(readme.read_bytes()))
    elif (readme := WORKSPACE_PATH / "README.md").exists():
        session.unit_prompts.append(decode_bytes(readme.read_bytes()))
    session.payload["tools"] = [
        *agent.select_tools(CATEGORIES.WORKSPACE),
        *agent.select_tools(CATEGORIES.NETWORK),
    ]
    await enter_workspace(agent, session)
    return ""


@TOOLS.create_category(WORKSPACE, "Tools for workspace file and command operations.")
async def workspace_category(agent: CloversAgent, session: Turn, event: Event):
    return await enter_workspace(agent, session)


@TOOLS.register(
    "read_image",
    "Load an image for visual inspection. Accepts an HTTP(S) URL or a local path.",
    {"path": {"type": "string", "description": 'Image URL (e.g. "https://example.com/a.png") or local path (e.g. "logo.png").'}},
    INTRINSIC_CATEGORY,
)
async def _(agent: CloversAgent, session: Turn, event: Event, path: str):
    path = path.strip()
    if not path:
        return "path must not be empty."
    if path.lower().startswith("http"):
        b64url = await session.api.download_url(path)
        if not b64url:
            return "Resource not found or unsupported format."
    else:
        file = WORKSPACE_PATH / session.session_id / format_path(path)
        if not file.exists() or not file.is_file():
            return f'File not found: "{path}"'
        b64url = to_b64url(file.read_bytes(), ALLOWED_IMAGE_SUBTYPES)
        if not b64url:
            return f'Not a recognized image format: "{path}"'
    session.input_image(b64url)
    return "Image attached to the current user input."


@TOOLS.register(
    "upload_file",
    "Send a file to the user. The user may not be able to see the workspace, so a path in your reply is not enough — call this to hand over the file itself.",
    {"file_path": {"type": "string", "description": "Path to the file to send."}},
    WORKSPACE,
)
async def _(agent: CloversAgent, session: Turn, event: Event, file_path: str):
    workspace = WORKSPACE_PATH / session.session_id
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
    "Run a shell command in the workspace. The shell persists between calls, so the working directory carries over — each result ends with the current working directory.",
    {"command": {"type": "string", "description": "Command to run. Chain multiple commands with `&&` or `;`."}},
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, session: Turn, event: Event, command: str):
    from .docker import Shell

    extra = session.extra
    if "shell" not in extra or not isinstance(shell := extra["shell"], Shell):
        return "Error: shell initialization failed. Report this to the user; do not call this tool again until it is resolved."
    output = await shell.execute(command)
    return f"{output}\nCurrent working directory: {shell.workdir}"


@TOOLS.register(
    "subagent",
    f"Delegate a self-contained task to a subagent (a separate agent that executes in its own context) to offload focused, independent work — research, a scoped implementation, an analysis — so it does not consume this conversation's context. The subagent returns its result, not its intermediate steps. The result is delivered after the current call chain ends, waking you to continue. You can run up to {MAX_SUBAGENT_COUNT} subagents per call chain.",
    {
        "prompt": {
            "type": "string",
            "description": "The complete, self-contained task for the subagent: the goal, the relevant files or paths, any constraints, and the expected output format. The subagent does not share this conversation's context, so include everything it needs to finish unaided",
        }
    },
    HIDDEN_CATEGORY,
)
async def _(agent: CloversAgent, session: Turn, event: Event, prompt: str):
    extra = session.extra
    if not KEY_SUBAGENTS in extra:
        extra[KEY_SUBAGENTS] = {}
    batchs = cast(dict[str, TurnBatch], extra[KEY_SUBAGENTS])
    session_id = session.session_id
    if session_id in batchs:
        batch = batchs[session_id]
        if len(batch.tasklist) > MAX_SUBAGENT_COUNT:
            return f"Subagent limit reached. Slots are released only when the current call chain ends — do not create more subagents in this chain."
    else:
        if not isinstance(session, Session):
            return "Subagents can only be created from the main session."
        batch = batchs[session_id] = TurnBatch(agent, session, event)
        batch.start().add_done_callback(lambda _: batchs.pop(session_id, None))

    s = session.create_turn()
    s.create_payload(session.api, SUBAGENT_PROMPT)
    tools = s.payload["tools"] = [
        *agent.select_tools(INTRINSIC_CATEGORY),
        *agent.select_tools(CATEGORIES.WORKSPACE),
        *agent.select_tools(CATEGORIES.NETWORK),
    ]
    s.unit_prompts.append(await enter_workspace(agent, s))
    if tool := next((x for x in tools if x["function"]["name"] == "subagent"), None):
        s.payload["tools"].remove(tool)
    s.input_text(prompt)
    batch.submit(s, event)
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
    session: Turn,
    event: Event,
    method: str,
    url: str,
    headers: dict | None = None,
    data: dict | None = None,
    file_path: str | None = None,
):
    if method.upper() == "POST":
        resp = await retry_request(agent.async_client.post, url, json=data, headers=headers)
    else:
        resp = await retry_request(agent.async_client.get, url, params=data, headers=headers)
    content = resp.content
    if not content:
        return f"Request succeeded (HTTP {resp.status_code}), but the response body was empty."
    r_size = len(content)
    s_size = format_size(r_size)
    # 指定了保存路径：写入工作区
    if file_path:
        file = WORKSPACE_PATH / session.session_id / format_path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
        return f'Saved to "{file_path}" ({s_size}, HTTP {resp.status_code}).'
    # 未指定保存路径，但内容过大：拦截并落盘
    if r_size > MAX_DOWNLOAD_RETURN_SIZE:
        saved_name = f"{md5(content)}.bin"
        file = WORKSPACE_PATH / session.session_id / saved_name
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
async def _(agent: CloversAgent, session: Turn, event: Event, path: str = "./", depth: int = 1):
    workspace = WORKSPACE_PATH / session.session_id
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
async def _(agent: CloversAgent, session: Turn, event: Event, file_path: str):
    workspace = WORKSPACE_PATH / session.session_id
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
async def _(agent: CloversAgent, session: Turn, event: Event, file_path: str, content: str):
    workspace = WORKSPACE_PATH / session.session_id
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
