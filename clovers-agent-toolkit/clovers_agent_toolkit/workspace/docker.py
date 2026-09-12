import asyncio
import docker
import shlex
from docker.models.containers import Container
from docker.errors import NotFound
from pathlib import Path

client: docker.DockerClient | None = None


class Shell:

    def __init__(self, session_id: str, workspace: Path, docker_image: str):
        self.lock = asyncio.Lock()
        self.session_id = session_id
        self.workspace = workspace / session_id
        self.workdir = "/workspace"
        self.container: Container | None = None
        self.docker_image = docker_image
        global client
        if client is None:
            client = docker.from_env()
        self.client = client
        self.stdout_log = self.workspace / "stdout.log"

    async def execute(self, command: str):
        async with self.lock:
            if self.container is None:
                self.workdir = "/workspace"
                if not self.workspace.exists():
                    self.workspace.mkdir(parents=True, exist_ok=True)
                container_name = f"CloversAgentSandbox-{self.session_id}"
                try:
                    self.container = await asyncio.to_thread(self.client.containers.get, container_name)
                except NotFound:
                    self.container = await asyncio.to_thread(
                        self.client.containers.run,
                        self.docker_image,
                        name=container_name,
                        detach=True,
                        tty=True,
                        command="sleep infinity",
                        volumes={self.workspace.resolve().as_posix(): {"bind": "/workspace", "mode": "rw"}},
                    )
            else:
                self.container
                self.container.reload()
            if self.container.status != "running":
                await asyncio.to_thread(self.container.start)
            assert self.container is not None
            # result = await asyncio.to_thread(self.container.exec_run, wrapped_command, workdir=self.workdir)
            # stdout: str = result.output.decode("utf-8")
            output, workdir = await asyncio.to_thread(self.execute_thread, command)
            self.workdir = workdir.strip()
            return output

    async def cleanup(self):
        async with self.lock:
            if self.container is None:
                return
            await asyncio.to_thread(self.container.remove, force=True)
            self.workdir = "/workspace"
            self.container = None

    def execute_thread(self, command: str):
        """运行命令,不输出被回车覆盖的行"""
        wrapped_command = f"bash -c {shlex.quote(f"{command}\necho '___CWD_MARKER___'\npwd")} < /dev/null"
        exec_id = self.client.api.exec_create(self.container.id, wrapped_command, workdir=self.workdir)  # type: ignore
        output_gen = self.client.api.exec_start(exec_id["Id"], stream=True)
        outputs: list[str] = []
        buffer: bytearray = bytearray()
        with self.stdout_log.open("w", buffering=1, encoding="utf-8", newline="") as f:
            f.write(f"[EXEC:{self.session_id}]:~{self.workdir}$ {command}\n")
            for chunk in output_gen:
                for byte in chunk:
                    buffer.append(byte)
                    if byte == 10:
                        line = buffer.decode("utf-8", errors="replace")
                        f.write(line)
                        outputs.append(line)
                        buffer.clear()
                    elif byte == 13:
                        line = buffer.decode("utf-8", errors="replace")
                        f.write(line)
                        buffer.clear()
        if buffer:
            outputs.append(buffer.decode("utf-8"))
        output, workdir = "".join(outputs).rsplit("___CWD_MARKER___", 1)
        return output, workdir.strip()
