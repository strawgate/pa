from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from pa import primitives


@dataclass
class PaPrimitiveTools(AbstractCapability[Any]):
    """Provide pa's sandbox primitives as a native Pydantic AI capability."""

    def get_toolset(self) -> AbstractToolset[Any]:
        toolset: FunctionToolset[Any] = FunctionToolset(id="pa-primitives")
        for name, tool in PRIMITIVES.items():
            toolset.tool_plain(name=name)(tool)
        return toolset


@dataclass
class PaRuntimeContext(AbstractCapability[Any]):
    """Inject current project context as dynamic instructions."""

    working_dir: Path
    project_root: Path

    def get_instructions(self):
        def _instructions(ctx: RunContext[Any]) -> str:
            return _build_context_instructions(
                ctx,
                working_dir=self.working_dir,
                project_root=self.project_root,
            )

        return _instructions


PRIMITIVES = {
    "read_file": primitives.read_file,
    "write_file": primitives.write_file,
    "list_dir": primitives.list_dir,
    "bash": primitives.bash,
    "http_get": primitives.http_get,
    "complete": primitives.complete,
}


def _build_context_instructions(
    ctx: RunContext[Any],
    *,
    working_dir: Path | None = None,
    project_root: Path | None = None,
) -> str:
    """Dynamic instruction fragment injected at runtime.

    Appended after the static instructions so it is always current. Mirrors pi's
    pattern of injecting date + cwd last, plus project context from AGENTS.md if
    present.
    """
    import datetime

    date = datetime.date.today().isoformat()
    resolved_working_dir = _path_from_ctx(ctx, ("working_dir", "project_dir"), working_dir or Path.cwd())
    resolved_project_root = _path_from_ctx(ctx, ("project_root",), project_root or resolved_working_dir)
    cwd = str(resolved_working_dir)

    parts = ["\nCurrent date: " + date, "\nCurrent working directory: " + cwd]

    agents_md = resolved_project_root / "AGENTS.md"
    if not agents_md.exists() and resolved_working_dir != resolved_project_root:
        agents_md = resolved_working_dir / "AGENTS.md"
    if agents_md.exists():
        content = agents_md.read_text(encoding="utf-8").strip()
        if content:
            parts.append("\n<project_context>\n" + content + "\n</project_context>")

    return "".join(parts)


def _path_from_ctx(ctx: RunContext[Any], names: tuple[str, ...], fallback: Path) -> Path:
    for name in names:
        value = getattr(ctx, name, None)
        if isinstance(value, Path):
            return value.resolve()
        if isinstance(value, str) and value:
            return Path(value).expanduser().resolve()
    return fallback.resolve()
