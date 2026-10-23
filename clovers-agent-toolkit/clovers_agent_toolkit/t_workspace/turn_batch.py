import asyncio
import contextlib
from clovers_agent import CloversAgent, Turn, Event


class TurnBatch:
    def __init__(self, agent: CloversAgent, event: Event) -> None:
        self.tasklist: list[asyncio.Task[str | None]] = []
        self.wakeup_task = None
        self._agent = agent
        self._event = event
        main_session = agent.current_session(agent.session_id(event))
        self._idle_event = main_session.idle_event
        s = main_session.create_turn()
        s.usage_counter = {}
        s.api = main_session.api
        s.payload = main_session.payload
        self._session = s

    def submit(self, session: Turn, event: Event):
        session.usage_counter = self._session.usage_counter
        self.tasklist.append(asyncio.create_task(self._run_turn(session, event)))

    def start(self):
        if self.wakeup_task is not None:
            raise RuntimeError("Wakeup is single-use; create a new instance per batch.")
        task = asyncio.create_task(self._wakeup())
        self.wakeup_task = task
        return task

    async def _run_turn(self, session: Turn, event: Event):
        with contextlib.suppress(Exception):
            last = None
            async for result in self._agent.call_turn(session, event):
                last = result
            return last

    async def _wakeup(self) -> None:
        await self._idle_event.wait()
        seen = set()
        while True:
            # 保证主调用链返回但子代理还在时还能继续添加子代理
            pending = tuple(t for t in self.tasklist if t not in seen)
            seen.update(pending)
            if not pending:
                break
            await asyncio.gather(*pending)
        self._session.unit_prompts.append("本条信息为子代理唤醒")
        self._session.input_text("---\n".join(task.result() or "None" for task in self.tasklist))
        asyncio.create_task(self._agent.wakeup(self._session, self._event))

    def cancel(self) -> None:
        for t in self.tasklist:
            t.cancel()
        if self.wakeup_task:
            self.wakeup_task.cancel()
