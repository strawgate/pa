from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory, Hooks
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from pa.manifest import Manifest, MANIFEST_PATH_DEFAULT
from pa.registrations import (
    make_instruction_fn,
    make_compaction_fn,
    make_guard_hook,
)


@dataclass
class PaRegistrations(AbstractCapability[Any]):
    """Loads ./pa/registrations.yaml and wires every entry into the agent."""

    manifest_path: str = str(MANIFEST_PATH_DEFAULT)
    _sub: list[AbstractCapability[Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        manifest = Manifest.load(self.manifest_path)
        self._sub = self._build(manifest)

    @classmethod
    def from_spec(cls, args: Any = (), kwargs: Any = {}) -> "PaRegistrations":
        if isinstance(kwargs, dict):
            return cls(**kwargs)
        return cls()

    def apply(self, visitor: Callable[[AbstractCapability[Any]], None]) -> None:
        """Expose sub-capabilities (Hooks, ProcessHistory, etc.) to the framework."""
        for cap in self._sub:
            cap.apply(visitor)

    def get_instructions(self):
        """Delegate instruction collection to the sub-capability."""
        for cap in self._sub:
            if isinstance(cap, _InstructionCapability):
                return cap.get_instructions()
        return None

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate before_tool_execute to sub-capabilities (guards)."""
        for cap in self._sub:
            args = await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        return args

    def _build(self, manifest: Manifest) -> list[AbstractCapability[Any]]:
        caps: list[AbstractCapability[Any]] = []

        inst_fns = [make_instruction_fn(r) for r in manifest.by_slot("instruction")]
        if inst_fns:
            caps.append(_InstructionCapability(fns=inst_fns))

        comp = manifest.by_slot("compaction")
        if comp:
            caps.append(ProcessHistory(make_compaction_fn(comp[0])))

        guards = manifest.by_slot("guard")
        if guards:
            hooks = Hooks()
            for r in guards:
                hooks.on.before_tool_execute(make_guard_hook(r))
            caps.append(hooks)

        # tool_filter registrations are applied at build time in runtime.py,
        # not as runtime capabilities, so nothing to do here.

        return caps


@dataclass
class _InstructionCapability(AbstractCapability[Any]):
    fns: list[Callable[..., Any]] = field(default_factory=list)

    def get_instructions(self):
        return list(self.fns)
