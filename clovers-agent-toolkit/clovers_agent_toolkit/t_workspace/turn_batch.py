import asyncio
from clovers_agent import CloversAgent, Turn, Event


class TurnBatch:
    def __init__(self, agent: CloversAgent, event: Event) -> None:
        self._turn_tasks: list[asyncio.Task[str | None]] = []
        self._wakeup_task = None
        self._task_added_event = asyncio.Event()
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
        self._turn_tasks.append(asyncio.create_task(self._run_turn(session, event)))
        self._task_added_event.set()

    def start(self):
        if self._wakeup_task is not None or self._task_added_event.is_set():
            raise RuntimeError("Wakeup is single-use; create a new instance per batch.")
        task = asyncio.create_task(self._wakeup())
        self._wakeup_task = task
        return task

    async def _run_turn(self, session: Turn, event: Event):
        last = None
        async for result in self._agent.call_turn(session, event):
            last = result
        return last

    async def _wakeup(self) -> None:
        await self._task_added_event.wait()
        await self._idle_event.wait()
        await asyncio.gather(*self._turn_tasks, return_exceptions=True)
        self._session.unit_prompts.append("本条信息为子代理唤醒")
        print("wakeup 唤醒时任务长度为：", len(self._turn_tasks))
        self._session.input_text("---\n".join(task.result() or "None" for task in self._turn_tasks))
        await self._agent.wakeup(self._session, self._event)

    def cancel(self) -> None:
        for t in self._turn_tasks:
            t.cancel()
        if self._wakeup_task:
            self._wakeup_task.cancel()
