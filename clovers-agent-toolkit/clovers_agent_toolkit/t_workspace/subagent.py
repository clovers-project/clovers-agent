import contextlib
from clovers_agent import CloversAgent, Event
from clovers_agent.api import OpenAIAPI
from clovers_agent.session import extend_tools
from clovers_agent.typing import Payload


class SubAgent:
    def __init__(self, api: OpenAIAPI, payload: Payload):
        self.api = api
        self.payload = payload
        self.usage_counter = {}
        self.running = False

    async def call(self, agent: CloversAgent, event: Event, command: str):
        self.running = True
        self.payload["messages"].append({"role": "user", "content": command})
        with contextlib.suppress(Exception):
            result = await agent.call_turn(self.api, self.payload, self.usage_counter, event)
        self.running = False
        return result
