import asyncio
from clovers_agent import CloversAgent, Event
from clovers.base import Coro


class Wakeup:

    def __init__(self, agent: CloversAgent, event: Event) -> None:
        self.agent = agent
        self.event = event
        session = agent.current_session(agent.session_id(event))
        s = session.create_turn()
        s.payload = session.payload
        self.session = s
        self.tasks: list[asyncio.Task[str | None]] = []
        self.watcher: asyncio.Task | None = None

    def append_task(self, coro: Coro[str | None]) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    def wait_all(self):
        if self.watcher is None:
            watcher = asyncio.create_task(self._wait_all())
            self.watcher = watcher
            return watcher

    async def _wait_all(self) -> None:
        try:
            seen: set[asyncio.Task] = set()
            while True:
                pending = [t for t in self.tasks if t not in seen]
                if not pending:
                    break
                seen.update(pending)
                await asyncio.gather(*pending, return_exceptions=True)
            self.session.input_text("本条信息为子代理唤醒")
            self.session.input_text("---\n".join(task.result() or "None" for task in self.tasks))
            asyncio.create_task(self.agent.wakeup(self.session, self.event))
        finally:
            self.running = False
            self._watcher = None

    def cancel(self) -> None:
        for t in self.tasks:
            t.cancel()
        if self.watcher:
            self.watcher.cancel()
