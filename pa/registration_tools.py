from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import FunctionToolset
from pydantic import Field
from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pa import primitives
from pa.manifest import (
    DEFAULT_REGISTERED_TOOL_TIMEOUT_S,
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
    limits_for_registration,
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

_COMPLETE_CALL_RE = re.compile(r"\bcomplete\s*\(")

SELF_EVOLUTION_TOOL_MAX_RETRIES = 15

REGISTERED_TOOL_EXTERNAL_FUNCTIONS: dict[str, Callable[..., Awaitable[Any] | Any]] = {
    "read_file": primitives.read_file,
    "write_file": primitives.write_file,
    "list_dir": primitives.list_dir,
    "bash": primitives.bash,
    "http_get": primitives.http_get,
    "complete": primitives.complete,
}

REGISTERED_TOOL_EXTRA_STUBS = """\
async def read_file(*, path: str) -> str: ...
async def write_file(*, path: str, content: str) -> str: ...
async def list_dir(*, path: str = ".") -> list[dict[str, Any]]: ...
async def bash(*, command: str, timeout_s: float = 30.0) -> dict[str, Any]: ...
async def http_get(*, url: str, timeout_s: float = 30.0) -> dict[str, Any]: ...
async def complete(
    *,
    prompt: str,
    system: str = "",
    data: str = "",
    output_schema: dict[str, Any] | None = None,
    output_mode: str = "native",
) -> Any: ...
"""

JsonToolValue = dict[str, object] | list[object] | str | int | float | bool | None

_REGISTRATION_TOOLSET_INSTRUCTIONS = """\
Registration code is Monty Python. Inputs are injected as variables and the
registration returns the value of the final expression. Do not wrap snippets in
`def ...` and do not use a top-level `return`.

Native tool hooks see Pydantic AI tool calls such as `run_code`,
`register_tool`, `list_registrations`, and active registered tools. Sandboxed
primitives (`read_file`, `write_file`, `list_dir`, `bash`, `http_get`,
`complete`) run inside the native `run_code` tool. Registered tools may also
call these primitives from Monty. To govern primitives, inspect
`tool_name == "run_code"` / registered tool arguments, or use
`register_tool_filter` to hide primitives before CodeMode exposes `run_code`.
Tool hooks see the outer registered tool call, not each primitive call made
inside that registered tool, so sensitive registered tools should enforce their
own skip rules too.

Working snippets:
- instruction: `"Call check_registrations after changing registrations."`
- before_run_hook: `"Checklist: inspect registration health before editing."`
- before_tool_hook: `{"action": "deny", "reason": "no writes"} if tool_name == "run_code" and "write_file(" in args.get("code", "") else {"action": "allow"}`
- after_tool_hook: `{"action": "modify", "result": {**result, "pa_note": "nonzero return"}} if isinstance(result, dict) and result.get("returncode", 0) != 0 else {"action": "allow"}`
- compaction: `list(range(max(0, len(messages) - 8), len(messages)))`
- tool_filter: `[name for name in tool_names if name != "bash"]`
- tool: `args["text"].strip().lower()` with a description, JSON schema, and
  `example_args` so it can be validated before activation. Tools can also use
  `await read_file(path=...)` and `await list_dir(path=...)` when they need
  filesystem context. Use `timeout_s` up to 60 seconds for legitimately slower
  tools that call `complete`, run bounded shell commands, or fetch network data.
  Always `await complete(...)`; it already returns the full text or structured
  dict, so do not wrap an un-awaited `complete(...)` call in another object. For
  structured LLM sub-tasks, pass `output_schema` to `complete`; it uses
  provider-native JSON schema output when supported and falls back to prompted
  structured output when native support is unavailable. Invalid structured
  sub-completion results are retried locally before the surrounding tool call
  fails.

Pydantic AI tool retries are fatal when exhausted. A validation error, denied
tool call, bad hook result, or after-tool `retry` response counts against that
tool's retry budget. If a call fails repeatedly, change strategy, inspect
registrations, or disable the broken registration instead of burning the budget.

After adding or changing registrations, call `check_registrations()`.
"""

_DESC_REGISTER_INSTRUCTION = """\
Register durable guidance that is injected into future model requests.

Use when you learned a lasting user preference, project convention, workflow rule,
or reminder that should shape future reasoning. The Monty code receives
`ctx_summary: dict` and must return a string as its final expression. Do not use
this for one-off facts that only matter in the current answer.

Example code: `"Always call check_registrations after registration changes."`
"""

_DESC_REGISTER_BEFORE_RUN_HOOK = """\
Register a hook that runs once at the start of each future agent run.

Use for run-local setup or reminders, such as checking project state, surfacing an
active checklist, or choosing a working mode. The Monty code receives
`ctx_summary: dict` and must return a string as its final expression. That
string is injected as run-local guidance for the model request.

Example code: `"Checklist: call check_registrations before editing registrations."`
"""

_DESC_REGISTER_AFTER_RUN_HOOK = """\
Register a hook that runs once after each future agent run completes.

Use for end-of-run policy such as normalizing final output, adding a required
signoff, or preserving a lightweight summary. The Monty code receives
`ctx_summary: dict` and `output`, and must return `{"action": "allow"}` or
`{"action": "replace_output", "output": str}` as its final expression.

Example code: `{"action": "allow"}`
"""

_DESC_REGISTER_BEFORE_TOOL_HOOK = """\
Register a hook that runs before every future tool call.

Use to enforce durable tool-call policy: block risky commands, prevent writes
outside the repo, or normalize arguments. This hook sees native Pydantic AI tool
calls; sandboxed primitives such as `bash` are inside `run_code`, so inspect the
`run_code` code string or use `register_tool_filter` to hide primitives. The
Monty code receives `tool_name: str` and `args: dict`, and must return
`{"action": "allow"}`, `{"action": "deny", "reason": str}`, or
`{"action": "modify", "args": dict}` as its final expression.

Example code: `{"action": "deny", "reason": "no writes"} if tool_name == "run_code" and "write_file(" in args.get("code", "") else {"action": "allow"}`
"""

_DESC_REGISTER_AFTER_TOOL_HOOK = """\
Register a hook that runs after every future tool call.

Use to enforce durable result policy: redact secrets, truncate noisy output,
rewrite confusing failures, or ask the model to retry a bad call. This hook sees
native Pydantic AI tool results; if `run_code` returns a primitive result, that
result is visible as the `run_code` result. The Monty code receives
`tool_name: str`, `args: dict`, and `result`, and must return
`{"action": "allow"}`, `{"action": "modify", "result": Any}`, or
`{"action": "retry", "reason": str}` as its final expression.

Use `retry` sparingly: each retry consumes the native tool retry budget, and
exhausting that budget aborts the whole agent run.

Example code: `{"action": "modify", "result": {**result, "pa_note": "nonzero return"}} if isinstance(result, dict) and result.get("returncode", 0) != 0 else {"action": "allow"}`
"""

_DESC_REGISTER_COMPACTION = """\
Register the single history-compaction hook for future model requests.

Use when conversation history needs a durable retention policy. The Monty code
receives `messages: list[dict]` and must return a list of message indices to
keep as its final expression. pa repairs unsafe output, but a bad compaction can
still make future runs less useful, so call `check_registrations()` after
registering.

Example code: `list(range(max(0, len(messages) - 8), len(messages)))`
"""

_DESC_REGISTER_TOOL_FILTER = """\
Register a hook that filters primitive tools before CodeMode exposes run_code.

Use to apply durable capability policy, such as read-only mode or hiding network
access. The Monty code receives `tool_names: list[str]` and must return the
subset to keep as its final expression. Tool filters can hide capabilities from
future runs, so prefer clear names and call `check_registrations()` after
registering.

Example code: `[name for name in tool_names if name != "bash"]`
"""

_DESC_REGISTER_TOOL = """\
Register a reusable native tool backed by a Monty snippet.

Use only after proving a repeatable operation in `run_code`. The code receives
`args: dict` and returns any JSON-serializable value as its final expression.
Registered tools may also await `read_file`, `write_file`, `list_dir`, `bash`,
`http_get`, and `complete` with keyword-only arguments. Always provide
`description`. Provide a JSON object schema and `example_args` whenever
possible; without `example_args`, the tool is saved as a draft and is not
callable until `validate_tool` succeeds. Set `timeout_s` when validation and
future calls need more than the default sandbox time; the maximum is 60 seconds.
If the tool calls `complete`, it must use `await complete(..., output_schema=schema)`.
For structured tool results, provide `output_json_schema` too; pa validates that
result when activating, health-checking, and running the tool.
For deterministic structured tools, also provide `expected_example_output` when
the model can reliably supply it; pa compares the validation result exactly so a
shape-correct but wrong tool does not become active.
When validation succeeds, the tool is exposed as a native tool on the next
agent run. It is not available later in the same run and cannot be called from
inside `run_code`.

Example code: `args["text"].strip().lower()`
"""

_DESC_VALIDATE_TOOL = """\
Validate a draft registered tool with concrete example arguments.

Use after `register_tool` saved a draft, or after repairing a disabled tool. The
example args must satisfy the tool schema. On success, the tool becomes active
and will be exposed as a native tool on the next agent run. It is not available
later in the same run and cannot be called from inside `run_code`. Pass
`output_json_schema` to add or replace result validation while validating. Pass
`expected_example_output` for deterministic tools when you want an exact
semantic check for the example args.
"""

_DESC_DISABLE_REGISTRATION = """\
Disable any registration without deleting its source.

Use to quarantine broken, risky, or no-longer-wanted self-evolution behavior
while keeping the code available for inspection or repair. Disabled
registrations stay in the agent's registrations manifest but do not run.
"""

_DESC_LIST_REGISTRATIONS = """\
List all registrations as JSON.

Use to inspect what the agent has learned, including slot, status, health,
description, preview, last error, and last run timing. This does not execute
registrations; use `check_registrations` when you need a smoke test.
"""

_DESC_CHECK_REGISTRATIONS = """\
Smoke-check registrations and return a JSON health report.

Use after adding or changing registrations, or when behavior looks surprising.
This executes registrations with safe sample inputs where possible, records
health fields, and reports which registrations are ok, skipped, or failing.
"""

_DESC_REMOVE_REGISTRATION = """\
Delete a registration from the manifest.

Use when a registration should be permanently removed rather than quarantined.
Prefer `disable_registration` first when you may want to inspect, repair, or
reuse the code later.
"""


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


def _normalize_output_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    if not isinstance(schema, dict):
        raise ValueError("tool output_json_schema must be a JSON schema object")
    normalized = dict(schema)
    if (normalized.get("type") == "object" or "properties" in normalized) and "properties" in normalized:
        normalized.setdefault("additionalProperties", False)
    return normalized


def _validate_tool_code_policy(code: str) -> None:
    if _COMPLETE_CALL_RE.search(code) and "output_schema" not in code:
        raise ValueError("registered tools that call complete must pass output_schema=... for reliable validation")


def validate_registered_tool_output(reg: Registration, value: Any) -> None:
    primitives.reject_unresolved_async_placeholders(value)
    if reg.output_json_schema is not None:
        primitives.validate_json_schema_subset(reg.output_json_schema, value)


def validate_expected_example_output(reg: Registration, value: Any) -> None:
    if reg.expected_example_output is None:
        return
    actual = to_jsonable_python(value)
    expected = to_jsonable_python(reg.expected_example_output)
    if actual != expected:
        raise ValueError(f"example output mismatch: expected {expected!r}, got {actual!r}")


def _schema_only_validation_note(reg: Registration) -> str:
    if reg.output_json_schema is None or reg.expected_example_output is not None:
        return ""
    return " Shape was validated, but no expected_example_output is stored; exact example semantics were not checked."


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


def _return_or_retry_for_agent(result: str, *, retry_disabled_validation: bool = False) -> str:
    if result.startswith("ERROR:") or (retry_disabled_validation and "validation failed" in result):
        raise ModelRetry(result)
    return result


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


def register_before_tool_hook(name: str, code: str) -> str:
    """Register a before-tool hook that can allow, deny, or modify tool args."""
    return _register("before_tool_hook", name, code)


def register_after_tool_hook(name: str, code: str) -> str:
    """Register an after-tool hook that can allow, retry, or modify tool results."""
    return _register("after_tool_hook", name, code)


def register_before_run_hook(name: str, code: str) -> str:
    """Register a start-of-run hook returning run-local guidance text."""
    return _register("before_run_hook", name, code)


def register_after_run_hook(name: str, code: str) -> str:
    """Register an end-of-run hook that can allow or replace final output."""
    return _register("after_run_hook", name, code)


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
        external_functions=REGISTERED_TOOL_EXTERNAL_FUNCTIONS,
        extra_stubs=REGISTERED_TOOL_EXTRA_STUBS,
        limits=limits_for_registration(reg),
    )
    validate_registered_tool_output(reg, result.value)
    validate_expected_example_output(reg, result.value)
    return result.value


def _validate_tool_sync(reg: Registration, example_args: dict[str, Any]) -> Any:
    return _run_coro_sync(lambda: _run_tool_validation(reg, example_args))


def register_tool(
    name: str,
    description: str,
    code: str,
    parameters_json_schema: dict[str, Any] | None = None,
    example_args: dict[str, Any] | None = None,
    output_json_schema: dict[str, Any] | None = None,
    expected_example_output: Any | None = None,
    timeout_s: float = DEFAULT_REGISTERED_TOOL_TIMEOUT_S,
) -> str:
    """Register a reusable Monty tool.

    Tools are draft-only until validated with an example argument object. Pass
    `example_args` here, or call `validate_tool(name, example_args)` later.
    Set `timeout_s` for tools that legitimately need longer validation or
    runtime execution. The maximum is 60 seconds.
    """
    return _run_coro_sync(
        lambda: _register_tool(
            name,
            description,
            code,
            parameters_json_schema,
            example_args,
            output_json_schema,
            expected_example_output,
            timeout_s,
            path=MANIFEST_PATH_DEFAULT,
        )
    )


async def _register_tool(
    name: str,
    description: str,
    code: str,
    parameters_json_schema: dict[str, Any] | None,
    example_args: dict[str, Any] | None,
    output_json_schema: dict[str, Any] | None,
    expected_example_output: Any | None,
    timeout_s: float,
    *,
    path: Path | str,
) -> str:
    try:
        schema = _normalize_tool_schema(parameters_json_schema)
        output_schema = _normalize_output_schema(output_json_schema)
        _validate_tool_code_policy(code)
        compile_registration(slot="tool", name=name, code=code, extra_stubs=REGISTERED_TOOL_EXTRA_STUBS)
        reg = Registration(
            slot="tool",
            name=name,
            code=code,
            description=description,
            parameters_json_schema=schema,
            output_json_schema=output_schema,
            expected_example_output=expected_example_output,
            timeout_s=timeout_s,
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
            status_note = (
                "validated and activated; it will be exposed as a native tool on the next agent run, "
                "not inside run_code or later in this run" + _schema_only_validation_note(reg)
            )
    else:
        status_note = "saved as draft; call validate_tool before it becomes callable on a future run"

    m = _load(path)
    try:
        m.add(reg)
    except CardinalityError as e:
        return f"ERROR: cardinality: {e}"
    except ManifestError as e:
        return f"ERROR: {e}"
    _save(m, path)
    return f"OK: registered tool/{name} ({reg.status}). {status_note}."


def validate_tool(
    name: str,
    example_args: dict[str, Any],
    output_json_schema: dict[str, Any] | None = None,
    expected_example_output: Any | None = None,
) -> str:
    """Validate a registered tool with example args and activate it on success."""
    return _run_coro_sync(
        lambda: _validate_tool(
            name,
            example_args,
            output_json_schema,
            expected_example_output,
            path=MANIFEST_PATH_DEFAULT,
        )
    )


async def _validate_tool(
    name: str,
    example_args: dict[str, Any],
    output_json_schema: dict[str, Any] | None = None,
    expected_example_output: Any | None = None,
    *,
    path: Path | str,
) -> str:
    m = _load(path)
    reg = m.find(name)
    if reg is None:
        return f"ERROR: no registration named {name!r}"
    if reg.slot != "tool":
        return f"ERROR: registration {name!r} is a {reg.slot}, not a tool"
    try:
        if output_json_schema is not None:
            reg.output_json_schema = _normalize_output_schema(output_json_schema)
        if expected_example_output is not None:
            reg.expected_example_output = expected_example_output
        value = await _run_tool_validation(reg, example_args)
    except (MontyBridgeError, ValueError) as e:
        reg.status = "disabled"
        reg.last_error = _stringify_error(e)
        _save(m, path)
        return f"ERROR: validation failed for tool/{name}: {e}"
    reg.status = "active"
    reg.validated_example_args = example_args
    if expected_example_output is not None:
        reg.expected_example_output = expected_example_output
    reg.last_error = ""
    _save(m, path)
    preview = json.dumps(value, default=str)[:160]
    return (
        f"OK: validated and activated tool/{name}. It will be exposed as a native tool on the next agent run, "
        f"not inside run_code or later in this run. Example result: {preview}.{_schema_only_validation_note(reg)}"
    )


def disable_registration(name: str, reason: str = "") -> str:
    """Disable any registration without deleting its source."""
    return _disable_registration(name, reason, path=MANIFEST_PATH_DEFAULT)


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


def list_registrations_at(path: Path | str) -> str:
    """Return registrations from a specific manifest path as JSON."""
    return _list_registrations(path=path)


def _registration_summary(r: Registration) -> dict[str, Any]:
    health = r.last_run_status
    if r.status == "disabled":
        health = "disabled"
    elif r.last_error:
        health = "error"
    elif r.status == "draft":
        health = "draft"
    summary = {
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
    if r.slot == "tool":
        summary["timeout_s"] = r.timeout_s
        summary["has_output_schema"] = r.output_json_schema is not None
        summary["has_expected_example_output"] = r.expected_example_output is not None
    return summary


def _list_registrations(*, path: Path | str) -> str:
    m = _load(path)
    out = [_registration_summary(r) for r in m.registrations]
    return json.dumps(out, indent=2)


def check_registrations() -> str:
    """Smoke-check active registrations and return a JSON health report."""
    return _run_coro_sync(lambda: _check_registrations(path=MANIFEST_PATH_DEFAULT))


def check_registrations_at(path: Path | str) -> str:
    """Smoke-check active registrations from a specific manifest path."""
    return _run_coro_sync(lambda: _check_registrations(path=path))


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
                    result = await run_registration(
                        reg,
                        inputs=inputs,
                        manifest=m,
                        manifest_path=path,
                        external_functions=REGISTERED_TOOL_EXTERNAL_FUNCTIONS,
                        extra_stubs=REGISTERED_TOOL_EXTRA_STUBS,
                    )
                else:
                    result = await run_registration(reg, inputs=inputs, manifest=m, manifest_path=path)
                policy_error = ""
                if reg.slot == "tool":
                    validate_registered_tool_output(reg, result.value)
                    if inputs == {"args": reg.validated_example_args}:
                        validate_expected_example_output(reg, result.value)
                else:
                    primitives.reject_unresolved_async_placeholders(result.value)
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
                    error=f"smoke test failed: {e}",
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
    if reg.slot == "before_tool_hook":
        return {"tool_name": "read_file", "args": {"path": "README.md"}}
    if reg.slot == "after_tool_hook":
        return {"tool_name": "read_file", "args": {"path": "README.md"}, "result": "README contents"}
    if reg.slot == "before_run_hook":
        return {"ctx_summary": {"agent_name": "pa-doctor", "run_step": 0}}
    if reg.slot == "after_run_hook":
        return {"ctx_summary": {"agent_name": "pa-doctor", "run_step": 0}, "output": "health check"}
    if reg.slot == "tool_filter":
        return {"tool_names": ["read_file", "write_file", "list_dir", "bash", "http_get", "complete"]}
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


def make_registration_toolset(manifest_path: str | Path, *, include_advanced: bool = True) -> FunctionToolset[Any]:
    """Create native registration-management tools bound to a manifest path."""
    toolset: FunctionToolset[Any] = FunctionToolset(
        id="pa-registration-management",
        max_retries=SELF_EVOLUTION_TOOL_MAX_RETRIES,
        sequential=True,
        instructions=_REGISTRATION_TOOLSET_INSTRUCTIONS,
    )
    path = Path(manifest_path)

    def register_instruction_bound(name: str, code: str) -> str:
        return _return_or_retry_for_agent(_register("instruction", name, code, path=path))

    def register_compaction_bound(name: str, code: str) -> str:
        return _return_or_retry_for_agent(_register("compaction", name, code, path=path))

    def register_before_tool_hook_bound(name: str, code: str) -> str:
        return _return_or_retry_for_agent(_register("before_tool_hook", name, code, path=path))

    def register_after_tool_hook_bound(name: str, code: str) -> str:
        return _return_or_retry_for_agent(_register("after_tool_hook", name, code, path=path))

    def register_before_run_hook_bound(name: str, code: str) -> str:
        return _return_or_retry_for_agent(_register("before_run_hook", name, code, path=path))

    def register_after_run_hook_bound(name: str, code: str) -> str:
        return _return_or_retry_for_agent(_register("after_run_hook", name, code, path=path))

    def register_tool_filter_bound(name: str, code: str) -> str:
        return _return_or_retry_for_agent(_register("tool_filter", name, code, path=path))

    async def register_tool_bound(
        name: str,
        description: str,
        code: str,
        parameters_json_schema: dict[str, Any] | None = None,
        example_args: dict[str, Any] | None = None,
        output_json_schema: Annotated[
            dict[str, Any] | None,
            Field(description="Optional JSON schema for validating the tool result."),
        ] = None,
        expected_example_output: Annotated[
            JsonToolValue,
            Field(
                description=(
                    "Exact expected result for example_args. Required for deterministic tools when "
                    "output_json_schema is provided. Pass the object/value itself, not null."
                )
            ),
        ] = None,
        timeout_s: float = DEFAULT_REGISTERED_TOOL_TIMEOUT_S,
    ) -> str:
        result = await _register_tool(
            name,
            description,
            code,
            parameters_json_schema,
            example_args,
            output_json_schema,
            expected_example_output,
            timeout_s,
            path=path,
        )
        return _return_or_retry_for_agent(result, retry_disabled_validation=True)

    async def validate_tool_bound(
        name: str,
        example_args: dict[str, Any],
        output_json_schema: Annotated[
            dict[str, Any] | None,
            Field(description="Optional JSON schema for validating the tool result."),
        ] = None,
        expected_example_output: Annotated[
            JsonToolValue,
            Field(
                description=(
                    "Exact expected result for example_args. Required for deterministic tools when "
                    "output_json_schema is provided. Pass the object/value itself, not null."
                )
            ),
        ] = None,
    ) -> str:
        result = await _validate_tool(name, example_args, output_json_schema, expected_example_output, path=path)
        return _return_or_retry_for_agent(result)

    def disable_registration_bound(name: str, reason: str = "") -> str:
        return _return_or_retry_for_agent(_disable_registration(name, reason, path=path))

    def list_registrations_bound() -> str:
        return _list_registrations(path=path)

    async def check_registrations_bound() -> str:
        return await _check_registrations(path=path)

    def remove_registration_bound(name: str) -> str:
        return _return_or_retry_for_agent(_remove_registration(name, path=path))

    toolset.tool_plain(name="register_instruction", description=_DESC_REGISTER_INSTRUCTION)(register_instruction_bound)
    toolset.tool_plain(name="register_before_tool_hook", description=_DESC_REGISTER_BEFORE_TOOL_HOOK)(
        register_before_tool_hook_bound
    )
    toolset.tool_plain(name="register_after_tool_hook", description=_DESC_REGISTER_AFTER_TOOL_HOOK)(
        register_after_tool_hook_bound
    )
    toolset.tool_plain(name="register_before_run_hook", description=_DESC_REGISTER_BEFORE_RUN_HOOK)(
        register_before_run_hook_bound
    )
    toolset.tool_plain(name="register_after_run_hook", description=_DESC_REGISTER_AFTER_RUN_HOOK)(
        register_after_run_hook_bound
    )
    toolset.tool_plain(name="register_tool", description=_DESC_REGISTER_TOOL)(register_tool_bound)
    toolset.tool_plain(name="validate_tool", description=_DESC_VALIDATE_TOOL)(validate_tool_bound)
    toolset.tool_plain(name="disable_registration", description=_DESC_DISABLE_REGISTRATION)(disable_registration_bound)
    toolset.tool_plain(name="list_registrations", description=_DESC_LIST_REGISTRATIONS)(list_registrations_bound)
    toolset.tool_plain(name="check_registrations", description=_DESC_CHECK_REGISTRATIONS)(check_registrations_bound)
    toolset.tool_plain(name="remove_registration", description=_DESC_REMOVE_REGISTRATION)(remove_registration_bound)
    if include_advanced:
        toolset.tool_plain(name="register_compaction", description=_DESC_REGISTER_COMPACTION)(register_compaction_bound)
        toolset.tool_plain(name="register_tool_filter", description=_DESC_REGISTER_TOOL_FILTER)(
            register_tool_filter_bound
        )
    return toolset
