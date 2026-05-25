"""Conversation execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_graph import End

from pa import history

T = TypeVar("T")


async def run_with_incremental_history(
    agent: Agent[Any, Any],
    prompt: str,
    message_history: Sequence[ModelMessage],
    history_path: Path,
) -> AgentRunResult[Any]:
    """Run an agent and persist conversation history after each graph step."""
    prior = history.normalize_for_replay(list(message_history))
    history.save(history.append_user_prompt(prior, prompt), history_path)

    async with agent.iter(prompt, message_history=prior) as agent_run:

        def save_current() -> None:
            current = agent_run.all_messages()
            if current != prior:
                history.save(current, history_path)

        try:
            node = agent_run.next_node
            while not isinstance(node, End):
                if agent_run.result is not None:
                    break
                node = await agent_run.next(node)
                save_current()
        finally:
            save_current()

    if agent_run.result is None:
        raise RuntimeError("The graph run did not finish properly")
    history.save(agent_run.result.all_messages(), history_path)
    return agent_run.result


def run_coro_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async operation from sync CLI code, even if a loop already exists."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()
