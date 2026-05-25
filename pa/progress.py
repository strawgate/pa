"""Structured progress events for agent runs."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultEvent,
    ToolReturnPart,
)


@dataclass(frozen=True, kw_only=True)
class ProgressEvent:
    """Base class for structured agent progress events."""

    kind: ClassVar[str]
    message: str


@dataclass(frozen=True, kw_only=True)
class RunStartedEvent(ProgressEvent):
    kind: ClassVar[str] = "run_started"
    prompt: str
    history_messages: int


@dataclass(frozen=True, kw_only=True)
class HistorySavedEvent(ProgressEvent):
    kind: ClassVar[str] = "history_saved"
    path: str
    messages: int
    phase: Literal["pending_user_prompt", "step", "final"]


@dataclass(frozen=True, kw_only=True)
class ToolCallStartedEvent(ProgressEvent):
    kind: ClassVar[str] = "tool_call_started"
    tool_name: str
    tool_call_id: str | None
    args_summary: str


@dataclass(frozen=True, kw_only=True)
class ToolCallFinishedEvent(ProgressEvent):
    kind: ClassVar[str] = "tool_call_finished"
    tool_name: str
    tool_call_id: str | None
    outcome: str | None
    result_summary: str


@dataclass(frozen=True, kw_only=True)
class RetryRequestedEvent(ProgressEvent):
    kind: ClassVar[str] = "retry_requested"
    tool_name: str
    tool_call_id: str | None
    reason_summary: str


@dataclass(frozen=True, kw_only=True)
class RunCompletedEvent(ProgressEvent):
    kind: ClassVar[str] = "run_completed"
    output: str
    history_messages: int


@dataclass(frozen=True, kw_only=True)
class RunFailedEvent(ProgressEvent):
    kind: ClassVar[str] = "run_failed"
    error_type: str
    error: str


def events_from_message(message: ModelMessage) -> list[ProgressEvent]:
    """Return structured progress events represented by one model message."""
    events: list[ProgressEvent] = []
    if isinstance(message, ModelResponse):
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                args_summary = summarize_args(part.args)
                events.append(
                    ToolCallStartedEvent(
                        message=f"-> {part.tool_name} {args_summary}".rstrip(),
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                        args_summary=args_summary,
                    )
                )
    elif isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                result_summary = summarize_value(part.content)
                outcome = str(part.outcome) if part.outcome is not None else None
                events.append(
                    ToolCallFinishedEvent(
                        message=f"<- {part.tool_name} {outcome}: {result_summary}",
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                        outcome=outcome,
                        result_summary=result_summary,
                    )
                )
            elif isinstance(part, RetryPromptPart):
                tool_name = part.tool_name or "model"
                reason_summary = summarize_value(part.content)
                events.append(
                    RetryRequestedEvent(
                        message=f"retry {tool_name}: {reason_summary}",
                        tool_name=tool_name,
                        tool_call_id=part.tool_call_id,
                        reason_summary=reason_summary,
                    )
                )
    return events


def event_from_agent_stream_event(event: AgentStreamEvent) -> ProgressEvent | None:
    """Return a progress event for one live Pydantic AI stream event."""
    if isinstance(event, ToolCallEvent):
        part = event.part
        args_summary = summarize_args(part.args)
        return ToolCallStartedEvent(
            message=f"-> {part.tool_name} {args_summary}".rstrip(),
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            args_summary=args_summary,
        )
    if isinstance(event, ToolResultEvent):
        part = event.part
        if isinstance(part, RetryPromptPart):
            tool_name = part.tool_name or "model"
            reason_summary = summarize_value(part.content)
            return RetryRequestedEvent(
                message=f"retry {tool_name}: {reason_summary}",
                tool_name=tool_name,
                tool_call_id=part.tool_call_id,
                reason_summary=reason_summary,
            )
        if isinstance(part, ToolReturnPart):
            result_summary = summarize_value(part.content)
            outcome = str(part.outcome) if part.outcome is not None else None
            return ToolCallFinishedEvent(
                message=f"<- {part.tool_name} {outcome}: {result_summary}",
                tool_name=part.tool_name,
                tool_call_id=part.tool_call_id,
                outcome=outcome,
                result_summary=result_summary,
            )
    return None


def event_to_dict(event: ProgressEvent) -> dict[str, Any]:
    """Serialize a progress event to a JSON-compatible dict."""
    data: dict[str, Any] = {"type": event.kind}
    for field in fields(event):
        data[field.name] = _jsonable(getattr(event, field.name))
    return data


def event_to_json(event: ProgressEvent) -> str:
    """Serialize a progress event as one JSON line."""
    return json.dumps(event_to_dict(event), ensure_ascii=False, sort_keys=True)


def summarize_args(args: str | dict[str, Any] | None) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return clip(args)
    if code := args.get("code"):
        return "code=" + clip(str(code))
    if command := args.get("command"):
        return "command=" + clip(str(command))
    if path := args.get("path"):
        return "path=" + clip(str(path))
    return clip(json_dump(args))


def summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        if "returncode" in value:
            parts.append(f"returncode={value['returncode']}")
        if stdout := value.get("stdout"):
            parts.append("stdout=" + clip(str(stdout)))
        if stderr := value.get("stderr"):
            parts.append("stderr=" + clip(str(stderr)))
        if parts:
            return " ".join(parts)
        if "body" in value:
            status = f"status={value.get('status')} " if "status" in value else ""
            return status + "body=" + clip(str(value["body"]))
    return clip(json_dump(value) if isinstance(value, (dict, list)) else str(value))


def json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def clip(value: str, limit: int = 240) -> str:
    one_line = " ".join(value.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
