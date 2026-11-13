"""Model reasoning ("thought") parts must never reach an agent's output."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import run_agent_once, run_agent_stream


def part(text: str, thought: bool = False) -> SimpleNamespace:
    return SimpleNamespace(text=text, thought=thought)


def event(parts: list[SimpleNamespace], *, partial: bool, final: bool = False) -> SimpleNamespace:
    return SimpleNamespace(error_code=None, error_message=None, partial=partial, usage_metadata=None,
                           content=SimpleNamespace(parts=parts), is_final_response=lambda: final)


# Shape of a Gemma 4 streamed reply: reasoning parts first, then the answer.
EVENTS = [
    event([part("*   Task: extract facts.\n", thought=True)], partial=True),
    event([part("*   Output: JSON list.\n", thought=True)], partial=True),
    event([part('["blue", ')], partial=True),
    event([part('"green"]')], partial=True),
    event([part("*   Task: extract facts.\n*   Output: JSON list.\n", thought=True), part('["blue", "green"]')],
          partial=False, final=True),
]


class FakeRunner:
    def __init__(self, agent, app_name):
        self.session_service = SimpleNamespace(create_session=self._create_session)

    async def _create_session(self, **_kwargs):
        return None

    async def run_async(self, **_kwargs):
        for item in EVENTS:
            yield item


class AnswerTextTests(unittest.TestCase):
    def test_stream_skips_thought_parts_in_deltas_and_result(self) -> None:
        deltas: list[str] = []
        with patch("google.adk.runners.InMemoryRunner", FakeRunner):
            result = asyncio.run(run_agent_stream(SimpleNamespace(model=None), state={}, message="m",
                                                  on_text=lambda delta, _acc: deltas.append(delta)))
        self.assertEqual(result, '["blue", "green"]')
        self.assertEqual("".join(deltas), '["blue", "green"]')

    def test_once_skips_thought_parts(self) -> None:
        with patch("google.adk.runners.InMemoryRunner", FakeRunner):
            result = asyncio.run(run_agent_once(SimpleNamespace(model=None), state={}, message="m"))
        self.assertEqual(result, '["blue", "green"]')


if __name__ == "__main__":
    unittest.main()
