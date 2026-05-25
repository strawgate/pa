"""Persistent conversation history.

The CLI stores history in pa's resolved state directory under ``~/.pa`` by
default. The path remains injectable so tests and low-level callers can use
their own storage.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

# Keep at most this many message objects (≈ MAX_MESSAGES/2 turns).
MAX_MESSAGES = 40

HISTORY_PATH_DEFAULT = Path("pa") / "history.json"


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


def load(path: Path | str = HISTORY_PATH_DEFAULT) -> list:
    """Load history. Returns [] if absent or corrupt."""
    history_path = Path(path)
    if not history_path.exists():
        return []
    try:
        raw = history_path.read_bytes()
        messages = ModelMessagesTypeAdapter.validate_json(raw)
        return _safe_truncate(list(messages), MAX_MESSAGES)
    except Exception:
        return []


def save(messages: list, path: Path | str = HISTORY_PATH_DEFAULT) -> None:
    """Serialize and write messages."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    # Trim before saving so the file never grows unbounded
    trimmed = _safe_truncate(list(messages), MAX_MESSAGES)
    raw = ModelMessagesTypeAdapter.dump_json(trimmed)
    history_path.write_bytes(raw)


def clear(path: Path | str = HISTORY_PATH_DEFAULT) -> None:
    """Delete the history file."""
    history_path = Path(path)
    if history_path.exists():
        history_path.unlink()
