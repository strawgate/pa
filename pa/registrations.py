from __future__ import annotations

from typing import Any, Awaitable, Callable

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai.exceptions import ModelRetry
from pydantic_core import to_jsonable_python

from pa.manifest import Manifest, Registration
from pa.monty_bridge import execute_registration, MontyBridgeError
from pa.registration_tools import validate_args_against_schema


def make_instruction_fn(reg: Registration) -> Callable[[RunContext[Any]], Awaitable[str | None]]:
    async def _instruction(ctx: RunContext[Any]) -> str | None:
        ctx_summary = {
            "agent_name": getattr(ctx.agent, "name", None) if ctx.agent else None,
            "run_step": ctx.run_step,
        }
        try:
            res = await execute_registration(
                slot="instruction",
                name=reg.name,
                code=reg.code,
                inputs={"ctx_summary": ctx_summary},
            )
        except MontyBridgeError as e:
            return f"[pa: instruction {reg.name!r} failed: {e}]"
        return res.value or None

    _instruction.__name__ = f"pa_instruction_{reg.name}"
    return _instruction


def make_compaction_fn(reg: Registration):
    async def _compact(messages: list[ModelMessage]) -> list[ModelMessage]:
        if not messages:
            return messages
        jsonable = [to_jsonable_python(m) for m in messages]
        try:
            res = await execute_registration(
                slot="compaction",
                name=reg.name,
                code=reg.code,
                inputs={"messages": jsonable},
            )
        except MontyBridgeError:
            return messages  # fail-safe: do not drop history on bridge error
        out: list[ModelMessage] = []
        n = len(messages)
        for idx in res.value:
            if 0 <= idx < n:
                out.append(messages[idx])
        return out or messages

    _compact.__name__ = f"pa_compaction_{reg.name}"
    return _compact


def make_guard_hook(reg: Registration):
    async def _guard(
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            res = await execute_registration(
                slot="guard",
                name=reg.name,
                code=reg.code,
                inputs={"tool_name": call.tool_name, "args": args},
            )
        except MontyBridgeError as e:
            raise ModelRetry(f"guard {reg.name!r} crashed: {e}") from e
        action = res.value["action"]
        if action == "allow":
            return args
        if action == "deny":
            reason = res.value.get("reason", "denied by guard")
            raise ModelRetry(f"guard {reg.name!r} denied {call.tool_name!r}: {reason}")
        if action == "modify":
            new_args = res.value.get("args", args)
            if not isinstance(new_args, dict):
                raise ModelRetry(f"guard {reg.name!r} produced non-dict args")
            return new_args
        raise ModelRetry(f"guard {reg.name!r}: unknown action {action!r}")

    _guard.__name__ = f"pa_guard_{reg.name}"
    return _guard


def make_registered_toolset(manifest: Manifest) -> FunctionToolset[Any]:
    """Create native Pydantic AI tools for active tool registrations."""
    toolset: FunctionToolset[Any] = FunctionToolset(id="pa-registered-tools", max_retries=2)

    for reg in manifest.by_slot("tool"):
        if reg.status != "active":
            continue
        toolset.add_tool(_make_registered_tool(reg))

    return toolset


def _make_registered_tool(reg: Registration) -> Tool[Any]:
    async def _tool(**kwargs: Any) -> Any:
        try:
            res = await execute_registration(
                slot="tool",
                name=reg.name,
                code=reg.code,
                inputs={"args": kwargs},
            )
        except MontyBridgeError as e:
            raise ModelRetry(f"registered tool {reg.name!r} failed: {e}") from e
        return res.value

    def _validate_args(ctx: RunContext[Any], **kwargs: Any) -> None:
        try:
            validate_args_against_schema(reg.parameters_json_schema, kwargs)
        except ValueError as e:
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
