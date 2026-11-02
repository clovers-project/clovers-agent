import asyncio
import docker
import shlex
from pathlib import Path
from collections import deque
from docker.models.containers import Container
from docker.errors import APIError, NotFound

from collections.abc import Iterable

DOCKER_CLIENT: docker.DockerClient = docker.from_env()


class Shell:
    def __init__(self, session_id: str, workspace: Path, docker_image: str):
        self.session_id = session_id
        self.workspace = workspace / session_id
        self.container: Container | None = None
        self.docker_image = docker_image
        self.client = DOCKER_CLIENT
        self._activate_lock = asyncio.Lock()

    async def activate(self):
        async with self._activate_lock:
            if self.container is not None:
                return
            self.workspace.mkdir(parents=True, exist_ok=True)
            container_name = f"CloversAgentSandbox-{self.session_id}"
            try:
                container = await asyncio.to_thread(self.client.containers.get, container_name)
            except NotFound:
                container = await asyncio.to_thread(
                    self.client.containers.run,
                    self.docker_image,
                    name=container_name,
                    detach=True,
                    tty=True,
                    command="sleep infinity",
                    volumes={self.workspace.resolve().as_posix(): {"bind": "/workspace", "mode": "rw"}},
                )
            else:
                if container.status != "running":
                    await asyncio.to_thread(container.start)
            self.container = container

    async def execute(self, command: str, timeout: int, max_output: int) -> str:
        await self.activate()
        wrapped = f"timeout -k 5 {timeout} bash -c {shlex.quote(command)} < /dev/null"
        return await asyncio.to_thread(self.execute_thread, wrapped, max_output)

    async def cleanup(self):
        async with self._activate_lock:
            if self.container is None:
                return
            await asyncio.to_thread(self.container.remove, force=True)
            self.container = None

    def execute_thread(self, command: str, max_output: int) -> str:
        if self.container is None or (container_id := self.container.id) is None:
            raise RuntimeError("Container not initialized")
        try:
            exec_id = self.client.api.exec_create(container_id, command, workdir="/workspace")["Id"]
            exec_stream = self.client.api.exec_start(exec_id, stream=True)
        except APIError:
            self.container.reload()
            if self.container.status != "running":
                self.container.start()
            exec_id = self.client.api.exec_create(container_id, command, workdir="/workspace")["Id"]
            exec_stream = self.client.api.exec_start(exec_id, stream=True)
        return self.stdout(exec_stream, max_output)

    @staticmethod
    def stdout(output_gen: Iterable[bytes], max_output: int) -> str:
        max_output = max_output + 2
        outputs = deque[str](maxlen=max_output)
        buffer = bytearray()
        for chunk in output_gen:
            for byte in chunk:
                if byte == 13:
                    buffer.clear()
                else:
                    buffer.append(byte)
                    if byte == 10:
                        outputs.extend(buffer.decode("utf-8", errors="replace"))
                        buffer.clear()
        if buffer:
            outputs.extend(buffer.decode("utf-8", errors="replace"))
        if len(outputs) == max_output:
            outputs.popleft()
            outputs.popleft()
            outputs.popleft()
            outputs.appendleft("...")
        return "".join(outputs)
