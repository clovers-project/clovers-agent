import asyncio
from clovers_agent import CloversAgent, Turn, Event
from clovers_agent.session import Session


class TurnBatch:
    def __init__(self, agent: CloversAgent, session: Session, event: Event) -> None:
        if (cursor := session.cursor) is None:
            raise RuntimeError("Cannot create turn_batch when session is not active")
        self.tasklist: list[asyncio.Task[str | None]] = []
        self.wakeup_task = None
        self._agent = agent
        self._event = event
        self._idle_event = session.idle_event
        self._m_idx = cursor
        self._last_rt = session.last_active_time
        s = session.create_turn()
        s.usage_counter = {}
        s.api = session.api
        s.payload = session.payload
        self._session = s

    def submit(self, session: Turn, event: Event):
        session.usage_counter = self._session.usage_counter
        self.tasklist.append(asyncio.create_task(self._agent.resolve_turn(session, event)))

    def start(self):
        if self.wakeup_task is not None:
            raise RuntimeError("Wakeup is single-use; create a new instance per batch.")
        task = asyncio.create_task(self._wakeup())
        self.wakeup_task = task
        return task

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
        self._session.unit_prompts.append("本条信息为子代理唤醒\n")
        self._session.unit_prompts.extend("\n---\n".join(task.result() or "None" for task in self.tasklist))
        asyncio.create_task(self._agent.on_wakeup(self._session, self._event, self._m_idx, self._last_rt))

    def cancel(self) -> None:
        for t in self.tasklist:
            t.cancel()
        if self.wakeup_task:
            self.wakeup_task.cancel()
