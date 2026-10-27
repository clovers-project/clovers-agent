import asyncio
import docker
import shlex
from docker.models.containers import Container
from docker.errors import NotFound
from pathlib import Path

DOCKER_CLIENT: docker.DockerClient = docker.from_env()


class Shell:
    def __init__(self, session_id: str, workspace: Path, docker_image: str):
        self.session_id = session_id
        self.workspace = workspace / session_id
        self.container: Container | None = None
        self.docker_image = docker_image
        self.client = DOCKER_CLIENT

    async def activate(self):
        if self.container is None:
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
            self.container.reload()
        if self.container.status != "running":
            await asyncio.to_thread(self.container.start)

    async def execute(self, command: str):
        return await asyncio.to_thread(self.execute_thread, command)

    async def cleanup(self):
        if self.container is None:
            return
        await asyncio.to_thread(self.container.remove, force=True)
        self.container = None

    def execute_thread(self, command: str):
        """运行命令,不输出被回车覆盖的行"""
        wrapped_command = f"bash -c {shlex.quote(command)} < /dev/null"
        if self.container is None:
            raise RuntimeError("Container not initialized")
        exec_id = self.client.api.exec_create(self.container.id, wrapped_command, workdir="/workspace")
        output_gen = self.client.api.exec_start(exec_id["Id"], stream=True)
        outputs: list[str] = []
        buffer: bytearray = bytearray()
        for chunk in output_gen:
            for byte in chunk:
                buffer.append(byte)
                if byte == 10:
                    line = buffer.decode("utf-8", errors="replace")
                    outputs.append(line)
                    buffer.clear()
                elif byte == 13:
                    line = buffer.decode("utf-8", errors="replace")
                    buffer.clear()
        if buffer:
            outputs.append(buffer.decode("utf-8"))
        return "".join(outputs)
