"""Conversation execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_graph import End

from pa import history

T = TypeVar("T")
ProgressCallback = Callable[[str], None]


async def run_with_incremental_history(
    agent: Agent[Any, Any],
    prompt: str,
    message_history: Sequence[ModelMessage],
    history_path: Path,
    progress: ProgressCallback | None = None,
) -> AgentRunResult[Any]:
    """Run an agent and persist conversation history after each graph step."""
    prior = history.normalize_for_replay(list(message_history))
    history.save(history.append_user_prompt(prior, prompt), history_path)
    emitted_count = len(prior)

    async with agent.iter(prompt, message_history=prior) as agent_run:

        def save_current() -> None:
            nonlocal emitted_count
            current = agent_run.all_messages()
            if current != prior:
                history.save(current, history_path)
            if progress is not None:
                if len(current) < emitted_count:
                    emitted_count = 0
                for message in current[emitted_count:]:
                    for line in summarize_progress(message):
                        progress(line)
                emitted_count = len(current)

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


def summarize_progress(message: ModelMessage) -> list[str]:
    """Return compact human-readable progress lines for one message."""
    lines: list[str] = []
    if isinstance(message, ModelResponse):
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                lines.append(f"-> {part.tool_name} {_summarize_args(part.args)}")
    elif isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                lines.append(f"<- {part.tool_name} {part.outcome}: {_summarize_value(part.content)}")
            elif isinstance(part, RetryPromptPart):
                tool_name = part.tool_name or "model"
                lines.append(f"retry {tool_name}: {_summarize_value(part.content)}")
    return lines


def _summarize_args(args: str | dict[str, Any] | None) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return _clip(args)
    if code := args.get("code"):
        return "code=" + _clip(str(code))
    if command := args.get("command"):
        return "command=" + _clip(str(command))
    if path := args.get("path"):
        return "path=" + _clip(str(path))
    return _clip(_json_dump(args))


def _summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        if "returncode" in value:
            parts.append(f"returncode={value['returncode']}")
        if stdout := value.get("stdout"):
            parts.append("stdout=" + _clip(str(stdout)))
        if stderr := value.get("stderr"):
            parts.append("stderr=" + _clip(str(stderr)))
        if parts:
            return " ".join(parts)
        if "body" in value:
            status = f"status={value.get('status')} " if "status" in value else ""
            return status + "body=" + _clip(str(value["body"]))
    return _clip(_json_dump(value) if isinstance(value, (dict, list)) else str(value))


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def _clip(value: str, limit: int = 240) -> str:
    one_line = " ".join(value.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1] + "…"


def run_coro_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async operation from sync CLI code, even if a loop already exists."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()
