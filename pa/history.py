"""Persistent per-project conversation history.

History is stored in ``pa/history.json`` in the current working directory
(same location as ``pa/registrations.yaml``).  On each ``pa run`` the last
``MAX_MESSAGES`` messages are loaded, passed as ``message_history`` to the
agent, and the updated history is written back after the run.  This gives the
agent memory of recent work without unbounded growth.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

# Keep at most this many message objects (≈ MAX_MESSAGES/2 turns).
MAX_MESSAGES = 40

_HISTORY_PATH = Path("pa") / "history.json"


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


def load() -> list:
    """Load history from pa/history.json. Returns [] if absent or corrupt."""
    if not _HISTORY_PATH.exists():
        return []
    try:
        raw = _HISTORY_PATH.read_bytes()
        messages = ModelMessagesTypeAdapter.validate_json(raw)
        return _safe_truncate(list(messages), MAX_MESSAGES)
    except Exception:
        return []


def save(messages: list) -> None:
    """Serialize and write messages to pa/history.json."""
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Trim before saving so the file never grows unbounded
    trimmed = _safe_truncate(list(messages), MAX_MESSAGES)
    raw = ModelMessagesTypeAdapter.dump_json(trimmed)
    _HISTORY_PATH.write_bytes(raw)


def clear() -> None:
    """Delete the history file."""
    if _HISTORY_PATH.exists():
        _HISTORY_PATH.unlink()
