from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

import pydantic_monty as pm
from pydantic_core import to_jsonable_python

from pa.slots import SLOTS, SlotName

DEFAULT_LIMITS = pm.ResourceLimits(max_duration_secs=2.0)

_STD_STUBS = """\
import asyncio
from typing import Any, Optional, Literal
"""

try:
    import logfire as _logfire  # type: ignore
    _LOGFIRE_SPAN = _logfire.span if _logfire.DEFAULT_LOGFIRE_INSTANCE.config.send_to_logfire else None
except (ImportError, AttributeError):
    _LOGFIRE_SPAN = None


def _span_ctx(**kw: Any) -> Any:
    if _LOGFIRE_SPAN is not None:
        return _LOGFIRE_SPAN("pa.registration.execute", **kw)
    return nullcontext()


class MontyBridgeError(Exception):
    ...


class MontySyntaxBridgeError(MontyBridgeError):
    ...


class MontyRuntimeBridgeError(MontyBridgeError):
    ...


class MontyReturnShapeError(MontyBridgeError):
    ...


@dataclass
class BridgeResult:
    value: Any
    stdout: str
    duration_ms: float


async def execute_registration(
    *,
    slot: SlotName,
    name: str,
    code: str,
    inputs: dict[str, Any],
    external_functions: dict[str, Callable[..., Awaitable[Any] | Any]] | None = None,
    extra_stubs: str | None = None,
    limits: pm.ResourceLimits | None = None,
) -> BridgeResult:
    slot_def = SLOTS[slot]
    if set(inputs) != set(slot_def.inputs):
        raise MontyBridgeError(f"slot {slot!r} expects inputs {slot_def.inputs}; got {tuple(inputs)}")
    safe_inputs = {k: to_jsonable_python(v) for k, v in inputs.items()}
    ext = external_functions or {}
    stubs = _build_stubs(slot_def, ext, extra_stubs)
    script_name = f"pa/registrations/{slot}/{name}.py"

    try:
        m = pm.Monty(
            code,
            inputs=list(safe_inputs.keys()),
            script_name=script_name,
            type_check=True,
            type_check_stubs=stubs,
        )
    except pm.MontySyntaxError as e:
        raise MontySyntaxBridgeError(f"{name}: syntax error: {e}") from e
    except pm.MontyTypingError as e:
        raise MontySyntaxBridgeError(f"{name}: type error: {e}") from e

    stdout_buf: list[str] = []

    def print_callback(stream: str, text: str) -> None:
        stdout_buf.append(text)

    t0 = asyncio.get_event_loop().time()
    with _span_ctx(slot=slot, name=name):
        try:
            result = await m.run_async(
                inputs=safe_inputs,
                external_functions=ext,
                limits=limits or DEFAULT_LIMITS,
                print_callback=print_callback,
            )
        except pm.MontyRuntimeError as e:
            raise MontyRuntimeBridgeError(f"{name}: runtime error: {e}") from e
        except pm.MontyTypingError as e:
            raise MontySyntaxBridgeError(f"{name}: type error: {e}") from e
    duration_ms = (asyncio.get_event_loop().time() - t0) * 1000.0

    return BridgeResult(
        value=_validate_return_shape(slot, result),
        stdout="".join(stdout_buf),
        duration_ms=duration_ms,
    )


def _build_stubs(slot_def, ext, extra) -> str:
    lines: list[str] = [_STD_STUBS]
    for inp in slot_def.inputs:
        lines.append(f"{inp}: Any = ...  # injected by pa")
    for fn_name in ext:
        lines.append(f"def {fn_name}(*args: Any, **kwargs: Any) -> Any: ...")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _validate_return_shape(slot: SlotName, value: Any) -> Any:
    if slot == "instruction":
        if not isinstance(value, str):
            raise MontyReturnShapeError(f"instruction must return str, got {type(value).__name__}")
        return value
    if slot == "compaction":
        if not isinstance(value, list) or not all(isinstance(i, int) for i in value):
            raise MontyReturnShapeError("compaction must return list[int]")
        return value
    if slot == "guard":
        if not isinstance(value, dict) or "action" not in value:
            raise MontyReturnShapeError("guard must return dict with 'action' key")
        if value["action"] not in ("allow", "deny", "modify"):
            raise MontyReturnShapeError(f"guard action must be allow|deny|modify, got {value['action']!r}")
        return value
    if slot == "tool_filter":
        if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
            raise MontyReturnShapeError("tool_filter must return list[str]")
        return value
    if slot == "tool":
        return value  # tools can return anything JSON-serializable
    raise MontyBridgeError(f"unknown slot {slot!r}")
