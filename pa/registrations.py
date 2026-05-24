from __future__ import annotations

from typing import Any, Awaitable, Callable

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai.exceptions import ModelRetry
from pydantic_core import to_jsonable_python

from pa.manifest import Manifest, Registration
from pa.registration_runtime import (
    RegistrationExecutionError,
    compaction_policy_error,
    normalize_compaction_indices,
    record_registration_result,
    run_registration,
)
from pa.registration_tools import validate_args_against_schema


def make_before_run_hook(reg: Registration, *, manifest: Manifest | None = None, manifest_path: str = ""):
    async def _before_run(ctx: RunContext[Any]) -> str:
        ctx_summary = {
            "agent_name": getattr(ctx.agent, "name", None) if ctx.agent else None,
            "run_step": ctx.run_step,
        }
        try:
            res = await run_registration(
                reg,
                inputs={"ctx_summary": ctx_summary},
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except RegistrationExecutionError:
            return ""
        return res.value

    _before_run.__name__ = f"pa_before_run_hook_{reg.name}"
    return _before_run


def make_after_run_hook(reg: Registration, *, manifest: Manifest | None = None, manifest_path: str = ""):
    async def _after_run(ctx: RunContext[Any], output: Any) -> Any:
        ctx_summary = {
            "agent_name": getattr(ctx.agent, "name", None) if ctx.agent else None,
            "run_step": ctx.run_step,
        }
        try:
            res = await run_registration(
                reg,
                inputs={"ctx_summary": ctx_summary, "output": output},
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except RegistrationExecutionError:
            return output
        action = res.value["action"]
        if action == "allow":
            return output
        if action == "replace_output":
            return res.value.get("output", output)
        return output

    _after_run.__name__ = f"pa_after_run_hook_{reg.name}"
    return _after_run


def make_instruction_fn(
    reg: Registration,
    *,
    manifest: Manifest | None = None,
    manifest_path: str = "",
) -> Callable[[RunContext[Any]], Awaitable[str | None]]:
    async def _instruction(ctx: RunContext[Any]) -> str | None:
        ctx_summary = {
            "agent_name": getattr(ctx.agent, "name", None) if ctx.agent else None,
            "run_step": ctx.run_step,
        }
        try:
            res = await run_registration(
                reg,
                inputs={"ctx_summary": ctx_summary},
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except RegistrationExecutionError as e:
            return f"[pa: instruction {reg.name!r} failed: {e}]"
        return res.value or None

    _instruction.__name__ = f"pa_instruction_{reg.name}"
    return _instruction


def make_compaction_fn(reg: Registration, *, manifest: Manifest | None = None, manifest_path: str = ""):
    async def _compact(messages: list[ModelMessage]) -> list[ModelMessage]:
        if not messages:
            return messages
        jsonable = [to_jsonable_python(m) for m in messages]
        try:
            res = await run_registration(
                reg,
                inputs={"messages": jsonable},
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except RegistrationExecutionError:
            return messages  # fail-safe: do not drop history on bridge error
        out: list[ModelMessage] = []
        n = len(messages)
        indices = normalize_compaction_indices(res.value, n)
        policy_error = compaction_policy_error(res.value, n)
        if policy_error:
            record_registration_result(reg, ok=False, error=policy_error, manifest=manifest, path=manifest_path)
        if n - 1 not in indices:
            indices.append(n - 1)
        for idx in indices:
            if 0 <= idx < n:
                out.append(messages[idx])
        return out or messages

    _compact.__name__ = f"pa_compaction_{reg.name}"
    return _compact


def make_before_tool_hook(reg: Registration, *, manifest: Manifest | None = None, manifest_path: str = ""):
    async def _before_tool(
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            res = await run_registration(
                reg,
                inputs={"tool_name": call.tool_name, "args": args},
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except RegistrationExecutionError as e:
            raise ModelRetry(f"before_tool_hook {reg.name!r} crashed: {e}") from e
        action = res.value["action"]
        if action == "allow":
            return args
        if action == "deny":
            reason = res.value.get("reason", "denied by before-tool hook")
            raise ModelRetry(f"before_tool_hook {reg.name!r} denied {call.tool_name!r}: {reason}")
        if action == "modify":
            new_args = res.value.get("args", args)
            if not isinstance(new_args, dict):
                raise ModelRetry(f"before_tool_hook {reg.name!r} produced non-dict args")
            return new_args
        raise ModelRetry(f"before_tool_hook {reg.name!r}: unknown action {action!r}")

    _before_tool.__name__ = f"pa_before_tool_hook_{reg.name}"
    return _before_tool


def make_guard_hook(reg: Registration, *, manifest: Manifest | None = None, manifest_path: str = ""):
    """Compatibility wrapper for legacy guard registrations."""
    return make_before_tool_hook(reg, manifest=manifest, manifest_path=manifest_path)


def make_after_tool_hook(reg: Registration, *, manifest: Manifest | None = None, manifest_path: str = ""):
    async def _after_tool(
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        try:
            res = await run_registration(
                reg,
                inputs={"tool_name": call.tool_name, "args": args, "result": result},
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except RegistrationExecutionError as e:
            raise ModelRetry(f"after_tool_hook {reg.name!r} crashed: {e}") from e
        action = res.value["action"]
        if action == "allow":
            return result
        if action == "modify":
            return res.value.get("result", result)
        if action == "retry":
            reason = res.value.get("reason", "rejected by after-tool hook")
            raise ModelRetry(f"after_tool_hook {reg.name!r} rejected {call.tool_name!r}: {reason}")
        raise ModelRetry(f"after_tool_hook {reg.name!r}: unknown action {action!r}")

    _after_tool.__name__ = f"pa_after_tool_hook_{reg.name}"
    return _after_tool


def make_registered_toolset(manifest: Manifest, *, manifest_path: str = "") -> FunctionToolset[Any]:
    """Create native Pydantic AI tools for active tool registrations."""
    toolset: FunctionToolset[Any] = FunctionToolset(id="pa-registered-tools", max_retries=2)

    for reg in manifest.active_by_slot("tool"):
        toolset.add_tool(_make_registered_tool(reg, manifest=manifest, manifest_path=manifest_path))

    return toolset


def _make_registered_tool(reg: Registration, *, manifest: Manifest, manifest_path: str) -> Tool[Any]:
    async def _tool(**kwargs: Any) -> Any:
        try:
            res = await run_registration(
                reg,
                inputs={"args": kwargs},
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except RegistrationExecutionError as e:
            raise ModelRetry(f"registered tool {reg.name!r} failed: {e}") from e
        return res.value

    def _validate_args(ctx: RunContext[Any], **kwargs: Any) -> None:
        try:
            validate_args_against_schema(reg.parameters_json_schema, kwargs)
        except ValueError as e:
            record_registration_result(
                reg,
                ok=False,
                error=f"invalid arguments: {e}",
                manifest=manifest,
                path=manifest_path,
            )
            raise ModelRetry(f"invalid arguments for registered tool {reg.name!r}: {e}") from e

    _tool.__name__ = reg.name
    _tool.__doc__ = reg.description or f"User-defined tool: {reg.name}"
    return Tool.from_schema(
        _tool,
        name=reg.name,
        description=reg.description or f"User-defined tool: {reg.name}",
        json_schema=reg.parameters_json_schema,
        takes_ctx=False,
        sequential=True,
        args_validator=_validate_args,
    )
