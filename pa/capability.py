from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Awaitable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.tools import ToolDefinition

from pa.manifest import MANIFEST_PATH_DEFAULT, Manifest
from pa.registration_runtime import RegistrationExecutionError, run_registration
from pa.registration_tools import make_registration_toolset
from pa.registrations import (
    make_after_run_hook,
    make_after_tool_hook,
    make_before_run_hook,
    make_before_tool_hook,
    make_compaction_fn,
    make_guard_hook,
    make_instruction_fn,
    make_registered_toolset,
)

_FILTERABLE_PRIMITIVES = frozenset({"read_file", "write_file", "list_dir", "bash", "http_get", "complete"})
_CANCELLED_TOOL_MESSAGE = "Tool call was cancelled before it returned."


@dataclass
class PaRegistrations(AbstractCapability[Any]):
    """Loads pa's registrations manifest and wires entries through native Pydantic AI hooks."""

    manifest_path: str = str(MANIFEST_PATH_DEFAULT)
    expose_advanced_registration_tools: bool = True
    _manifest: Manifest = field(init=False, repr=False)
    _compaction: Callable[[list[ModelMessage]], Awaitable[list[ModelMessage]]] | None = field(
        default=None, init=False, repr=False
    )
    _run_notes: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._manifest = Manifest.load(self.manifest_path)
        comp = self._manifest.active_by_slot("compaction")
        if comp:
            self._compaction = make_compaction_fn(
                comp[0],
                manifest=self._manifest,
                manifest_path=self.manifest_path,
            )

    async def for_run(self, ctx: RunContext[Any]) -> "PaRegistrations":
        return type(self)(
            manifest_path=self.manifest_path,
            expose_advanced_registration_tools=self.expose_advanced_registration_tools,
        )

    def get_toolset(self) -> AbstractToolset[Any] | None:
        toolset = make_registration_toolset(
            self.manifest_path,
            include_advanced=self.expose_advanced_registration_tools,
        )
        registered = make_registered_toolset(self._manifest, manifest_path=self.manifest_path)
        for tool in registered.tools.values():
            toolset.add_tool(tool)
        return toolset

    def get_instructions(self):
        instructions = [
            make_instruction_fn(r, manifest=self._manifest, manifest_path=self.manifest_path)
            for r in self._manifest.active_by_slot("instruction")
        ]

        async def _registration_errors(ctx: RunContext[Any]) -> str | None:
            sections = []
            if self._run_notes:
                sections.append("[pa: run hooks]\n" + "\n".join(self._run_notes))
            errors = [
                f"{r.slot}/{r.name}: {r.last_error}"
                for r in self._manifest.registrations
                if r.status == "active" and r.last_error
            ]
            if errors:
                sections.append("[pa: registration issues]\n" + "\n".join(errors))
            if not sections:
                return None
            return "\n\n".join(sections)

        instructions.append(_registration_errors)
        return instructions

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Run self-authored start-of-run hooks and collect run-local guidance."""
        self._run_notes = []
        for reg in self._manifest.active_by_slot("before_run_hook"):
            hook = make_before_run_hook(reg, manifest=self._manifest, manifest_path=self.manifest_path)
            note = await hook(ctx)
            if note:
                self._run_notes.append(f"{reg.name}: {note}")

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        """Run self-authored end-of-run hooks."""
        output = result.output
        for reg in self._manifest.active_by_slot("after_run_hook"):
            hook = make_after_run_hook(reg, manifest=self._manifest, manifest_path=self.manifest_path)
            output = await hook(ctx, output)
        result.output = output
        return result

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Filter primitive tools through registered tool_filter snippets."""
        filters = self._manifest.active_by_slot("tool_filter")
        if not filters:
            return tool_defs

        candidate_names = [td.name for td in tool_defs if td.name in _FILTERABLE_PRIMITIVES]
        allowed = list(candidate_names)
        for reg in filters:
            try:
                res = await run_registration(
                    reg,
                    inputs={"tool_names": allowed},
                    manifest=self._manifest,
                    manifest_path=self.manifest_path,
                )
            except RegistrationExecutionError:
                continue
            allowed = [name for name in res.value if name in allowed]

        allowed_set = set(allowed)
        return [td for td in tool_defs if td.name not in _FILTERABLE_PRIMITIVES or td.name in allowed_set]

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Run registered guards through native before-tool hooks."""
        for reg in self._manifest.active_by_slot("before_tool_hook"):
            hook = make_before_tool_hook(reg, manifest=self._manifest, manifest_path=self.manifest_path)
            args = await hook(ctx, call=call, tool_def=tool_def, args=args)
        for reg in self._manifest.active_by_slot("guard"):
            hook = make_guard_hook(reg, manifest=self._manifest, manifest_path=self.manifest_path)
            args = await hook(ctx, call=call, tool_def=tool_def, args=args)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Run registered after-tool hooks through native tool-execution hooks."""
        for reg in self._manifest.active_by_slot("after_tool_hook"):
            hook = make_after_tool_hook(reg, manifest=self._manifest, manifest_path=self.manifest_path)
            result = await hook(ctx, call=call, tool_def=tool_def, args=args, result=result)
        return result

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: Any,
    ) -> Any:
        """Repair orphaned tool calls/results before provider validation sees history."""
        messages = patch_orphaned_tool_parts(request_context.messages)
        if self._compaction is not None:
            compacted = await self._compaction(messages)
            request_context.messages = patch_orphaned_tool_parts(compacted)
        else:
            request_context.messages = messages
        return request_context


def patch_orphaned_tool_parts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Return messages with synthetic returns for orphaned calls and orphaned returns removed."""
    if not messages:
        return messages
    messages = _patch_orphaned_calls(messages)
    patched = _remove_orphaned_returns(messages)
    return patched or messages


def _patch_orphaned_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    patched: list[ModelMessage] = []
    skip_next = False
    for i, msg in enumerate(messages):
        if skip_next:
            skip_next = False
            continue
        patched.append(msg)
        if not isinstance(msg, ModelResponse):
            continue
        calls = [part for part in msg.parts if isinstance(part, ToolCallPart)]
        if not calls:
            continue
        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        answered = set()
        if isinstance(next_msg, ModelRequest):
            for part in next_msg.parts:
                if isinstance(part, (RetryPromptPart, ToolReturnPart)) and part.tool_call_id:
                    answered.add(part.tool_call_id)
        missing = [call for call in calls if call.tool_call_id not in answered]
        if not missing:
            continue
        synthetic = [
            ToolReturnPart(
                tool_name=call.tool_name,
                content=_CANCELLED_TOOL_MESSAGE,
                tool_call_id=call.tool_call_id,
            )
            for call in missing
        ]
        if isinstance(next_msg, ModelRequest):
            patched.append(ModelRequest(parts=[*synthetic, *next_msg.parts], timestamp=next_msg.timestamp))
            skip_next = True
        else:
            patched.append(ModelRequest(parts=synthetic))
    return patched


def _remove_orphaned_returns(messages: list[ModelMessage]) -> list[ModelMessage]:
    patched: list[ModelMessage] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, ModelRequest):
            patched.append(msg)
            continue
        prev_msg = messages[i - 1] if i > 0 else None
        call_ids: set[str] = set()
        if isinstance(prev_msg, ModelResponse):
            call_ids = {
                part.tool_call_id
                for part in prev_msg.parts
                if isinstance(part, ToolCallPart) and part.tool_call_id is not None
            }
        parts = [
            part
            for part in msg.parts
            if not (isinstance(part, ToolReturnPart) and part.tool_call_id and part.tool_call_id not in call_ids)
        ]
        if parts:
            patched.append(ModelRequest(parts=parts, timestamp=msg.timestamp, instructions=msg.instructions))
    return patched
