import sys
import asyncio
import docker
import shlex
from docker.models.containers import Container
from docker.types import DeviceRequest
from docker.errors import NotFound
from pathlib import Path


WORKSPACE = Path("workspace")
client: docker.DockerClient | None = None


class Shell:

    def __init__(self, session_id: str):
        self.lock = asyncio.Lock()
        self.session_id = session_id
        self.workdir = "/workspace"
        self.container: Container | None = None
        global client
        if client is None:
            client = docker.from_env()
        self.client = client

    async def execute(self, command: str):
        async with self.lock:
            if self.container is None:
                self.workdir = "/workspace"
                workspace = WORKSPACE / self.session_id
                if not workspace.exists():
                    workspace.mkdir(parents=True, exist_ok=True)
                container_name = f"CloversAgentSandbox-{self.session_id}"
                try:
                    self.container = await asyncio.to_thread(self.client.containers.get, container_name)
                except NotFound:
                    self.container = await asyncio.to_thread(
                        self.client.containers.run,
                        "nikolaik/python-nodejs:python3.12-nodejs20",
                        name=container_name,
                        detach=True,
                        tty=True,
                        command="sleep infinity",
                        volumes={workspace.resolve().as_posix(): {"bind": "/workspace", "mode": "rw"}},
                        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                    )
            else:
                self.container
                self.container.reload()
            if self.container.status != "running":
                await asyncio.to_thread(self.container.start)
            assert self.container is not None
            wrapped_command = f"bash -c {shlex.quote(f"{command}\necho '___CWD_MARKER___'\npwd")}"
            # result = await asyncio.to_thread(self.container.exec_run, wrapped_command, workdir=self.workdir)
            # stdout: str = result.output.decode("utf-8")
            stdout = await asyncio.to_thread(self.execute_thread, wrapped_command)
            output, workdir = stdout.rsplit("___CWD_MARKER___", 1)
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
        exec_id = self.client.api.exec_create(self.container.id, command, workdir=self.workdir)  # type:ignore
        output_gen = self.client.api.exec_start(exec_id["Id"], stream=True)
        outputs: list[str] = []
        buffer: bytearray = bytearray()
        for chunk in output_gen:
            for byte in chunk:
                buffer.append(byte)
                if byte == 10:
                    line = buffer.decode("utf-8")
                    # print(line)
                    outputs.append(line)
                    buffer.clear()
                elif byte == 13:
                    # sys.stdout.buffer.write(buffer)
                    # sys.stdout.flush()
                    buffer.clear()
        if buffer:
            outputs.append(buffer.decode("utf-8"))
        return "\n".join(outputs)
