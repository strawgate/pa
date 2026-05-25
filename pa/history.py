"""Persistent conversation history.

The CLI stores history in pa's resolved state directory under ``~/.pa`` by
default. The path remains injectable so tests and low-level callers can use
their own storage.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

from pa.state import pa_home

# Keep at most this many message objects (≈ MAX_MESSAGES/2 turns).
MAX_MESSAGES = 40

HISTORY_PATH_DEFAULT = pa_home() / "history.json"


def _history_path(path: Path | str | None) -> Path:
    return Path(path) if path is not None else pa_home() / "history.json"


def _safe_truncate(messages: list, max_messages: int) -> list:
    """Truncate to at most max_messages, starting at a clean UserPromptPart boundary.

    Never starts on a ToolReturnPart or RetryPromptPart — those reference tool
    call IDs from earlier messages that would no longer be present, causing
    "tool id not found" errors on the next run.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    if len(messages) <= max_messages:
        candidates = messages
    else:
        candidates = messages[-max_messages:]

    # Find the first ModelRequest whose parts are solely UserPromptPart(s)
    for i, msg in enumerate(candidates):
        if isinstance(msg, ModelRequest) and all(isinstance(p, UserPromptPart) for p in msg.parts):
            return list(candidates[i:])

    # No clean boundary found (e.g. all messages are tool turns) — drop all.
    # This is safer than sending orphaned tool results.
    return []


def _strip_persisted_instructions(messages: list) -> list:
    """Remove old dynamic instructions before persisting or replaying history."""
    return [
        replace(msg, instructions=None) if isinstance(msg, ModelRequest) and msg.instructions is not None else msg
        for msg in messages
    ]


def normalize_for_replay(messages: list) -> list:
    """Return messages safe to replay into Pydantic AI."""
    return _strip_persisted_instructions(_safe_truncate(list(messages), MAX_MESSAGES))


def load(path: Path | str | None = None) -> list:
    """Load history. Returns [] if absent or corrupt."""
    history_path = _history_path(path)
    if not history_path.exists():
        return []
    try:
        raw = history_path.read_bytes()
        messages = ModelMessagesTypeAdapter.validate_json(raw)
        return normalize_for_replay(list(messages))
    except Exception:
        return []


def save(messages: list, path: Path | str | None = None) -> None:
    """Serialize and write messages."""
    history_path = _history_path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    # Trim before saving so the file never grows unbounded
    trimmed = normalize_for_replay(list(messages))
    raw = ModelMessagesTypeAdapter.dump_json(trimmed)
    history_path.write_bytes(raw)


def append_user_prompt(messages: list, content: str) -> list:
    """Return history with a pending user prompt appended."""
    return _safe_truncate(
        [
            *normalize_for_replay(list(messages)),
            ModelRequest(parts=[UserPromptPart(content=content)]),
        ],
        MAX_MESSAGES,
    )


def clear(path: Path | str | None = None) -> None:
    """Delete the history file."""
    history_path = _history_path(path)
    if history_path.exists():
        history_path.unlink()
