"""Conversation execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_graph import End

from pa import history
from pa.progress import (
    HistorySavedEvent,
    ProgressEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    event_from_agent_stream_event,
    events_from_message,
)

T = TypeVar("T")
ProgressCallback = Callable[[ProgressEvent], None]


async def run_with_incremental_history(
    agent: Agent[Any, Any],
    prompt: str,
    message_history: Sequence[ModelMessage],
    history_path: Path,
    progress: ProgressCallback | None = None,
) -> AgentRunResult[Any]:
    """Run an agent and persist conversation history after each graph step."""
    prior = history.normalize_for_replay(list(message_history))
    last_saved = history.append_user_prompt(prior, prompt)
    history.save(last_saved, history_path)
    if progress is not None:
        progress(RunStartedEvent(message="run started", prompt=prompt, history_messages=len(prior)))
        progress(
            HistorySavedEvent(
                message="history saved",
                path=str(history_path),
                messages=len(last_saved),
                phase="pending_user_prompt",
            )
        )
    try:
        async with agent.iter(prompt, message_history=prior) as agent_run:

            def save_current(
                phase: Literal["step", "final"] = "step",
                pending_request: ModelRequest | None = None,
            ) -> None:
                nonlocal last_saved
                current = agent_run.all_messages()
                if pending_request is not None and (not current or current[-1] != pending_request):
                    current = [*current, pending_request]
                saved = current != last_saved
                if saved:
                    history.save(current, history_path)
                    last_saved = list(current)
                if progress is not None and saved:
                    progress(
                        HistorySavedEvent(
                            message="history saved",
                            path=str(history_path),
                            messages=len(current),
                            phase=phase,
                        )
                    )

            try:
                node = agent_run.next_node
                while not isinstance(node, End):
                    if agent_run.result is not None:
                        break
                    if agent.is_call_tools_node(node):
                        async with node.stream(agent_run.ctx) as stream:
                            async for stream_event in stream:
                                if progress_event := event_from_agent_stream_event(stream_event):
                                    if progress is not None:
                                        progress(progress_event)
                        node = await agent_run.next(node)
                        pending_request = node.request if agent.is_model_request_node(node) else None
                        save_current(pending_request=pending_request)
                    else:
                        node = await agent_run.next(node)
                        save_current()
            finally:
                save_current()

        if agent_run.result is None:
            raise RuntimeError("The graph run did not finish properly")
        final_messages = agent_run.result.all_messages()
        history.save(final_messages, history_path)
        if progress is not None:
            progress(
                HistorySavedEvent(
                    message="history saved",
                    path=str(history_path),
                    messages=len(final_messages),
                    phase="final",
                )
            )
            progress(
                RunCompletedEvent(
                    message="run completed",
                    output=str(agent_run.result.output),
                    history_messages=len(final_messages),
                )
            )
        return agent_run.result
    except Exception as e:
        if progress is not None:
            progress(RunFailedEvent(message="run failed", error_type=type(e).__name__, error=str(e)))
        raise


def summarize_progress(message: ModelMessage) -> list[str]:
    """Return compact human-readable progress lines for one message."""
    return [event.message for event in events_from_message(message)]


def run_coro_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async operation from sync CLI code, even if a loop already exists."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()
