"""Compatibility shims for released Pydantic AI beta edges."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any


def apply_pydantic_ai_v2_harness_compat() -> None:
    """Apply local compatibility shims for the current Pydantic AI V2 beta.

    These can be removed once the upstream beta edges are fixed and released.
    """
    _patch_harness_parallel_execution_mode()
    _patch_run_sync_event_loop_warning()


def _patch_harness_parallel_execution_mode() -> None:
    """Keep pydantic-ai-harness CodeMode working with Pydantic AI V2 beta."""
    from pydantic_ai.tool_manager import ToolManager

    method = ToolManager.get_parallel_execution_mode
    if getattr(method, "_pa_accepts_legacy_tool_defs", False):
        return

    if len(signature(method).parameters) != 1:
        return

    @wraps(method)
    def _compat(self: ToolManager, *_tool_defs: Any) -> Any:
        return method(self)

    setattr(_compat, "_pa_accepts_legacy_tool_defs", True)
    setattr(ToolManager, "get_parallel_execution_mode", _compat)


def _patch_run_sync_event_loop_warning() -> None:
    """Avoid Python 3.13's `asyncio.get_event_loop()` deprecation in run_sync()."""
    import asyncio

    import pydantic_ai._utils as utils

    current = utils.get_event_loop
    if getattr(current, "_pa_avoids_get_event_loop_warning", False):
        return

    @wraps(current)
    def _compat() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass

        event_loop = getattr(_compat, "_pa_event_loop", None)
        if event_loop is not None and not event_loop.is_closed():
            return event_loop

        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        setattr(_compat, "_pa_event_loop", event_loop)
        return event_loop

    setattr(_compat, "_pa_avoids_get_event_loop_warning", True)
    setattr(utils, "get_event_loop", _compat)
