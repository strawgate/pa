from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

import pydantic_monty as pm

from pa.manifest import MANIFEST_PATH_DEFAULT, Manifest, Registration
from pa.monty_bridge import BridgeResult, MontyBridgeError, execute_registration


def stringify_error(e: BaseException) -> str:
    text = str(e)
    return text if len(text) <= 500 else text[:500] + "...[truncated]"


def normalize_compaction_indices(indices: list[int], message_count: int) -> list[int]:
    return sorted({idx for idx in indices if 0 <= idx < message_count})


def compaction_policy_error(indices: list[int], message_count: int) -> str:
    normalized = normalize_compaction_indices(indices, message_count)
    if normalized != indices:
        return "compaction returned duplicate, unordered, or out-of-range indices; pa will repair history"
    if message_count > 0 and message_count - 1 not in normalized:
        return "compaction must preserve the current request; pa will restore it"
    return ""


@dataclass
class RegistrationExecutionError(Exception):
    registration: Registration
    error: MontyBridgeError

    def __str__(self) -> str:
        return stringify_error(self.error)


async def run_registration(
    reg: Registration,
    *,
    inputs: dict[str, Any],
    manifest: Manifest | None = None,
    manifest_path: Path | str | None = MANIFEST_PATH_DEFAULT,
    external_functions: dict[str, Callable[..., Awaitable[Any] | Any]] | None = None,
    extra_stubs: str | None = None,
    limits: pm.ResourceLimits | None = None,
) -> BridgeResult:
    """Run one registration and persist standard health fields."""
    try:
        result = await execute_registration(
            slot=reg.slot,
            name=reg.name,
            code=reg.code,
            inputs=inputs,
            external_functions=external_functions,
            extra_stubs=extra_stubs,
            limits=limits,
        )
    except MontyBridgeError as e:
        record_registration_result(reg, ok=False, error=stringify_error(e), manifest=manifest, path=manifest_path)
        raise RegistrationExecutionError(reg, e) from e

    record_registration_result(
        reg,
        ok=True,
        duration_ms=result.duration_ms,
        manifest=manifest,
        path=manifest_path,
    )
    return result


def record_registration_result(
    reg: Registration,
    *,
    ok: bool,
    error: str = "",
    duration_ms: float | None = None,
    manifest: Manifest | None = None,
    path: Path | str | None = MANIFEST_PATH_DEFAULT,
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    updates: dict[str, Any] = {
        "last_run_status": "ok" if ok else "error",
        "last_run_at": now,
        "last_duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
        "last_error": "" if ok else error,
    }
    if ok:
        updates["last_ok_at"] = now

    changed = False
    for key, value in updates.items():
        if getattr(reg, key) != value:
            setattr(reg, key, value)
            changed = True

    if changed and manifest is not None and path is not None:
        manifest.save(path)
