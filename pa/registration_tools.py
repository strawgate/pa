from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic_ai.toolsets import FunctionToolset
from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pa.manifest import (
    MANIFEST_PATH_DEFAULT,
    CardinalityError,
    Manifest,
    ManifestError,
    Registration,
    default_tool_schema,
)
from pa.monty_bridge import MontyBridgeError, compile_registration, execute_registration
from pa.registration_runtime import (
    RegistrationExecutionError,
    compaction_policy_error,
    record_registration_result,
    run_registration,
    stringify_error,
)
from pa.slots import SlotName

_SCALAR_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _load(path: Path | str = MANIFEST_PATH_DEFAULT) -> Manifest:
    return Manifest.load(path)


def _save(m: Manifest, path: Path | str = MANIFEST_PATH_DEFAULT) -> None:
    m.save(path)


def _normalize_tool_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if schema is None:
        return default_tool_schema()
    if schema.get("type", "object") != "object":
        raise ValueError("tool parameters_json_schema must be an object schema")
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    normalized.setdefault("additionalProperties", False)
    if not isinstance(normalized["properties"], dict):
        raise ValueError("tool parameters_json_schema.properties must be a mapping")
    if not isinstance(normalized.get("required", []), list):
        raise ValueError("tool parameters_json_schema.required must be a list")
    return normalized


def validate_args_against_schema(schema: dict[str, Any], args: dict[str, Any]) -> None:
    """Small JSON-schema subset validator for registered tool arguments."""
    if schema.get("type", "object") != "object":
        raise ValueError("registered tool schema must be an object schema")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("registered tool schema properties must be a mapping")
    for required in schema.get("required", []):
        if required not in args:
            raise ValueError(f"missing required argument {required!r}")
    if schema.get("additionalProperties") is False:
        allowed = set(properties)
        extra = sorted(set(args) - allowed)
        if extra:
            raise ValueError(f"unexpected argument(s): {', '.join(extra)}")
    for key, value in args.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        if "enum" in prop and value not in prop["enum"]:
            raise ValueError(f"argument {key!r} must be one of {prop['enum']!r}")
        typ = prop.get("type")
        if isinstance(typ, list):
            if "null" in typ and value is None:
                continue
            typ = next((t for t in typ if t != "null"), None)
        if typ == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"argument {key!r} must be integer, got {type(value).__name__}")
            continue
        if typ == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"argument {key!r} must be number, got {type(value).__name__}")
            continue
        expected = _SCALAR_JSON_TYPES.get(typ) if isinstance(typ, str) else None
        if expected is not None and not isinstance(value, expected):
            raise ValueError(f"argument {key!r} must be {typ}, got {type(value).__name__}")


def _stringify_error(e: Exception) -> str:
    return stringify_error(e)


def _run_coro_sync(fn: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run async validation for sync callers without depending on the caller's event loop state."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(fn())).result()


def _register(
    slot: SlotName,
    name: str,
    code: str,
    *,
    path: Path | str = MANIFEST_PATH_DEFAULT,
) -> str:
    try:
        compile_registration(slot=slot, name=name, code=code)
        reg = Registration(slot=slot, name=name, code=code)
    except Exception as e:
        return f"ERROR: invalid registration: {e}"
    m = _load(path)
    try:
        m.add(reg)
    except CardinalityError as e:
        return f"ERROR: cardinality: {e}"
    except ManifestError as e:
        return f"ERROR: {e}"
    _save(m, path)
    return f"OK: registered {slot}/{name}. It will be active on next agent run."


def register_instruction(name: str, code: str) -> str:
    """Register a Monty snippet returning a string to append to the system prompt."""
    return _register("instruction", name, code)


def register_compaction(name: str, code: str) -> str:
    """Register THE (single) Monty snippet that compacts history.
    Receives `messages: list[dict]`, returns `list[int]` indices to keep."""
    return _register("compaction", name, code)


def register_guard(name: str, code: str) -> str:
    """Register a Monty guard. Receives `tool_name: str`, `args: dict`. Returns
    {'action': 'allow' | 'deny' | 'modify', ...}. First deny wins."""
    return _register("guard", name, code)


def register_tool_filter(name: str, code: str) -> str:
    """Register a Monty tool-filter. Receives `tool_names: list[str]`,
    returns the filtered list."""
    return _register("tool_filter", name, code)


async def _run_tool_validation(reg: Registration, example_args: dict[str, Any]) -> Any:
    validate_args_against_schema(reg.parameters_json_schema, example_args)
    result = await execute_registration(
        slot="tool",
        name=reg.name,
        code=reg.code,
        inputs={"args": example_args},
    )
    return result.value


def _validate_tool_sync(reg: Registration, example_args: dict[str, Any]) -> Any:
    return _run_coro_sync(lambda: _run_tool_validation(reg, example_args))


def register_tool(
    name: str,
    description: str,
    code: str,
    parameters_json_schema: dict[str, Any] | None = None,
    example_args: dict[str, Any] | None = None,
) -> str:
    """Register a reusable Monty tool.

    Tools are draft-only until validated with an example argument object. Pass
    `example_args` here, or call `validate_tool(name, example_args)` later.
    """
    return _run_coro_sync(
        lambda: _register_tool(
            name, description, code, parameters_json_schema, example_args, path=MANIFEST_PATH_DEFAULT
        )
    )


async def _register_tool(
    name: str,
    description: str,
    code: str,
    parameters_json_schema: dict[str, Any] | None,
    example_args: dict[str, Any] | None,
    *,
    path: Path | str,
) -> str:
    try:
        schema = _normalize_tool_schema(parameters_json_schema)
        compile_registration(slot="tool", name=name, code=code)
        reg = Registration(
            slot="tool",
            name=name,
            code=code,
            description=description,
            parameters_json_schema=schema,
            status="draft",
        )
    except Exception as e:
        return f"ERROR: invalid registration: {e}"

    if example_args is not None:
        try:
            await _run_tool_validation(reg, example_args)
        except (MontyBridgeError, ValueError) as e:
            reg.status = "disabled"
            reg.last_error = _stringify_error(e)
            status_note = f"validation failed; saved disabled draft: {e}"
        else:
            reg.status = "active"
            reg.validated_example_args = example_args
            status_note = "validated and activated"
    else:
        status_note = "saved as draft; call validate_tool before it becomes callable"

    m = _load(path)
    try:
        m.add(reg)
    except CardinalityError as e:
        return f"ERROR: cardinality: {e}"
    except ManifestError as e:
        return f"ERROR: {e}"
    _save(m, path)
    return f"OK: registered tool/{name} ({reg.status}). {status_note}."


def validate_tool(name: str, example_args: dict[str, Any]) -> str:
    """Validate a registered tool with example args and activate it on success."""
    return _run_coro_sync(lambda: _validate_tool(name, example_args, path=MANIFEST_PATH_DEFAULT))


async def _validate_tool(name: str, example_args: dict[str, Any], *, path: Path | str) -> str:
    m = _load(path)
    reg = m.find(name)
    if reg is None:
        return f"ERROR: no registration named {name!r}"
    if reg.slot != "tool":
        return f"ERROR: registration {name!r} is a {reg.slot}, not a tool"
    try:
        value = await _run_tool_validation(reg, example_args)
    except (MontyBridgeError, ValueError) as e:
        reg.status = "disabled"
        reg.last_error = _stringify_error(e)
        _save(m, path)
        return f"ERROR: validation failed for tool/{name}: {e}"
    reg.status = "active"
    reg.validated_example_args = example_args
    reg.last_error = ""
    _save(m, path)
    preview = json.dumps(value, default=str)[:160]
    return f"OK: validated and activated tool/{name}. Example result: {preview}"


def disable_tool(name: str, reason: str = "") -> str:
    """Disable a registered tool without deleting its source."""
    return _disable_tool(name, reason, path=MANIFEST_PATH_DEFAULT)


def disable_registration(name: str, reason: str = "") -> str:
    """Disable any registration without deleting its source."""
    return _disable_registration(name, reason, path=MANIFEST_PATH_DEFAULT)


def _disable_tool(name: str, reason: str = "", *, path: Path | str) -> str:
    m = _load(path)
    reg = m.find(name)
    if reg is None:
        return f"ERROR: no registration named {name!r}"
    if reg.slot != "tool":
        return f"ERROR: registration {name!r} is a {reg.slot}, not a tool"
    _disable_loaded_registration(m, reg, reason, path=path)
    return f"OK: disabled tool/{name}."


def _disable_registration(name: str, reason: str = "", *, path: Path | str) -> str:
    m = _load(path)
    reg = m.find(name)
    if reg is None:
        return f"ERROR: no registration named {name!r}"
    _disable_loaded_registration(m, reg, reason, path=path)
    return f"OK: disabled {reg.slot}/{name}."


def _disable_loaded_registration(m: Manifest, reg: Registration, reason: str, *, path: Path | str) -> None:
    reg.status = "disabled"
    reg.last_error = reason
    _save(m, path)


def list_registrations() -> str:
    """Return a JSON-encoded list of registration entries."""
    return _list_registrations(path=MANIFEST_PATH_DEFAULT)


def _registration_summary(r: Registration) -> dict[str, Any]:
    health = r.last_run_status
    if r.status == "disabled":
        health = "disabled"
    elif r.last_error:
        health = "error"
    elif r.status == "draft":
        health = "draft"
    return {
        "slot": r.slot,
        "name": r.name,
        "status": r.status,
        "health": health,
        "description": r.description,
        "lines": r.code.count("\n") + 1,
        "preview": r.code[:120] + ("..." if len(r.code) > 120 else ""),
        "last_error": r.last_error,
        "last_run_status": r.last_run_status,
        "last_run_at": r.last_run_at,
        "last_ok_at": r.last_ok_at,
        "last_duration_ms": r.last_duration_ms,
    }


def _list_registrations(*, path: Path | str) -> str:
    m = _load(path)
    out = [_registration_summary(r) for r in m.registrations]
    return json.dumps(out, indent=2)


def check_registrations() -> str:
    """Smoke-check active registrations and return a JSON health report."""
    return _run_coro_sync(lambda: _check_registrations(path=MANIFEST_PATH_DEFAULT))


async def _check_registrations(*, path: Path | str) -> str:
    m = _load(path)
    out: list[dict[str, Any]] = []
    for reg in m.registrations:
        entry = _registration_summary(reg)
        inputs = _smoke_inputs(reg)
        if reg.status == "disabled":
            entry.update({"check": "skipped", "reason": "disabled"})
        elif reg.slot == "tool" and reg.status != "active":
            entry.update({"check": "skipped", "reason": f"tool is {reg.status}"})
        elif inputs is None:
            entry.update({"check": "skipped", "reason": "no smoke-test inputs available"})
        else:
            try:
                if reg.slot == "tool":
                    validate_args_against_schema(reg.parameters_json_schema, inputs["args"])
                result = await run_registration(reg, inputs=inputs, manifest=m, manifest_path=path)
                policy_error = ""
                if reg.slot == "compaction":
                    policy_error = compaction_policy_error(result.value, len(inputs["messages"]))
                if policy_error:
                    record_registration_result(reg, ok=False, error=policy_error, manifest=m, path=path)
                    entry.update({"check": "error", "error": policy_error})
                else:
                    entry.update({"check": "ok"})
            except RegistrationExecutionError as e:
                entry.update({"check": "error", "error": str(e)})
            except ValueError as e:
                record_registration_result(
                    reg,
                    ok=False,
                    error=f"invalid smoke-test args: {e}",
                    manifest=m,
                    path=path,
                )
                entry.update({"check": "error", "error": str(e)})
        entry.update(_registration_summary(reg))
        out.append(entry)
    return json.dumps(out, indent=2)


def _smoke_inputs(reg: Registration) -> dict[str, Any] | None:
    if reg.slot == "instruction":
        return {"ctx_summary": {"agent_name": "pa-doctor", "run_step": 0}}
    if reg.slot == "compaction":
        return {
            "messages": [
                to_jsonable_python(ModelRequest(parts=[UserPromptPart(content="health check")])),
                to_jsonable_python(ModelResponse(parts=[TextPart(content="ok")])),
            ]
        }
    if reg.slot == "guard":
        return {"tool_name": "read_file", "args": {"path": "README.md"}}
    if reg.slot == "tool_filter":
        return {"tool_names": ["read_file", "write_file", "bash", "http_get", "complete"]}
    if reg.slot == "tool":
        if reg.validated_example_args is not None:
            return {"args": reg.validated_example_args}
        if reg.parameters_json_schema.get("required"):
            return None
        return {"args": {}}
    return None


def remove_registration(name: str) -> str:
    """Remove the registration with the given name."""
    return _remove_registration(name, path=MANIFEST_PATH_DEFAULT)


def _remove_registration(name: str, *, path: Path | str) -> str:
    m = _load(path)
    try:
        removed = m.remove(name)
    except ManifestError as e:
        return f"ERROR: {e}"
    _save(m, path)
    return f"OK: removed {removed.slot}/{removed.name}."


def make_registration_toolset(manifest_path: str | Path, *, include_advanced: bool = False) -> FunctionToolset[Any]:
    """Create native registration-management tools bound to a manifest path."""
    toolset: FunctionToolset[Any] = FunctionToolset(id="pa-registration-management")
    path = Path(manifest_path)

    def register_instruction_bound(name: str, code: str) -> str:
        """Register a Monty snippet returning a string to append to the system prompt."""
        return _register("instruction", name, code, path=path)

    def register_compaction_bound(name: str, code: str) -> str:
        """Register the single Monty snippet that compacts history."""
        return _register("compaction", name, code, path=path)

    def register_guard_bound(name: str, code: str) -> str:
        """Register a Monty guard that can allow, deny, or modify tool calls."""
        return _register("guard", name, code, path=path)

    def register_tool_filter_bound(name: str, code: str) -> str:
        """Register a Monty snippet that filters available primitive tools."""
        return _register("tool_filter", name, code, path=path)

    async def register_tool_bound(
        name: str,
        description: str,
        code: str,
        parameters_json_schema: dict[str, Any] | None = None,
        example_args: dict[str, Any] | None = None,
    ) -> str:
        """Register a draft reusable tool; provide example_args to activate it immediately."""
        return await _register_tool(name, description, code, parameters_json_schema, example_args, path=path)

    async def validate_tool_bound(name: str, example_args: dict[str, Any]) -> str:
        """Validate a registered tool with example args and activate it on success."""
        return await _validate_tool(name, example_args, path=path)

    def disable_tool_bound(name: str, reason: str = "") -> str:
        """Disable a registered tool without deleting its source."""
        return _disable_tool(name, reason, path=path)

    def disable_registration_bound(name: str, reason: str = "") -> str:
        """Disable any registration without deleting its source."""
        return _disable_registration(name, reason, path=path)

    def list_registrations_bound() -> str:
        """Return a JSON-encoded list of registration entries."""
        return _list_registrations(path=path)

    async def check_registrations_bound() -> str:
        """Smoke-check active registrations and return a JSON health report."""
        return await _check_registrations(path=path)

    def remove_registration_bound(name: str) -> str:
        """Remove the registration with the given name."""
        return _remove_registration(name, path=path)

    toolset.tool_plain(name="register_instruction", description=register_instruction_bound.__doc__)(
        register_instruction_bound
    )
    toolset.tool_plain(name="register_guard", description=register_guard_bound.__doc__)(register_guard_bound)
    toolset.tool_plain(name="register_tool", description=register_tool_bound.__doc__)(register_tool_bound)
    toolset.tool_plain(name="validate_tool", description=validate_tool_bound.__doc__)(validate_tool_bound)
    toolset.tool_plain(name="disable_registration", description=disable_registration_bound.__doc__)(
        disable_registration_bound
    )
    toolset.tool_plain(name="list_registrations", description=list_registrations_bound.__doc__)(
        list_registrations_bound
    )
    toolset.tool_plain(name="check_registrations", description=check_registrations_bound.__doc__)(
        check_registrations_bound
    )
    toolset.tool_plain(name="remove_registration", description=remove_registration_bound.__doc__)(
        remove_registration_bound
    )
    if include_advanced:
        toolset.tool_plain(name="register_compaction", description=register_compaction_bound.__doc__)(
            register_compaction_bound
        )
        toolset.tool_plain(name="register_tool_filter", description=register_tool_filter_bound.__doc__)(
            register_tool_filter_bound
        )
        toolset.tool_plain(name="disable_tool", description=disable_tool_bound.__doc__)(disable_tool_bound)
    return toolset
