"""Compatibility shims for released Pydantic AI beta edges."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any


def apply_pydantic_ai_v2_harness_compat() -> None:
    """Keep pydantic-ai-harness CodeMode working with Pydantic AI V2 beta.

    pydantic-ai-harness 0.3.0 calls ToolManager.get_parallel_execution_mode([])
    using the V1 signature. Pydantic AI 2.0.0b3 changed that method to accept
    only self. Patch only when the installed core has the V2-shaped signature.
    """
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
