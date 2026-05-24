# pa — A Self-Evolving Pydantic-AI Agent Harness: Technical Specification (v0.1)

**Audience**: a junior Python developer who is fluent in Python but has not used Pydantic-AI before. **Date**: 2026-05-23. **License of the produced artifact**: AGPL-3.0.

This is a complete, file-by-file specification for `pa`, the smallest possible self-evolving Pydantic-AI agent runtime. It is split into two parts:

- **Part 1 — Research synthesis (~2,500 words).** Grounded findings about Monty, Pydantic-AI capabilities/hooks, the pydantic-ai-harness CodeMode bridge, Agent specs, and Logfire — the load-bearing technologies behind the design.
- **Part 2 — Implementation brief (~5,500 words).** A file-by-file design any competent Python developer can execute over two focused days, with public APIs, key code snippets, acceptance criteria, the demo script, and an hour-by-hour plan.

Every decision is locked. There is no "developer chooses" anywhere in this document.

---

## Part 1 — Research synthesis

### 1.1 Monty internals

Monty (github.com/pydantic/monty) is a **Python interpreter written entirely in Rust** — not a CPython sandbox, not a WASM compile of CPython, and not a restricted-`exec` wrapper. It parses a subset of Python 3.14 syntax with Ruff's parser, compiles to a custom bytecode, and runs that bytecode in a sandboxed VM with strict resource limits and zero default access to the host (filesystem, env, network). All side-effects must go through host-provided **external functions**.

The crate layout is a Cargo workspace: the core interpreter lives in `crates/monty`, with PyO3 bindings in `crates/monty-python` (published to PyPI as **`pydantic-monty`**, version 0.0.17, released April 22, 2026 per upload timestamps on every wheel on pypi.org/project/pydantic-monty/), napi-rs bindings in `crates/monty-js`, and an internal `monty-cli`. Wheels are built for Python 3.12–3.14 across the usual manylinux / macOS / Windows matrix.

**Performance.** Per the pydantic/monty GitHub README: "Startup extremely fast (<1μs to go from code to execution result)". Talk Python To Me episode 541 with Samuel Colvin gives the practical numbers: "roughly 6 microseconds cold and under 1 microsecond in a hot loop, compared to over a second for container-based sandboxes and nearly 3 seconds for Pyodide." Steady-state runtime, per the README verbatim, "has runtime performance that is similar to CPython (generally between 5x faster and 5x slower)." Per Samuel Colvin's pydantic.dev article *Pydantic Monty: A Minimal Python Sandbox for AI Agents* (Feb 27, 2026): "A Monty snapshot is single-digit kilobytes. If you're building agents that pause and resume — say, waiting for human approval — this difference is not academic."

**The Python entry point.** The canonical class is `pydantic_monty.Monty`. The README example, copied verbatim, is:

```python
from typing import Any
import pydantic_monty

code = """
async def agent(prompt: str, messages: Messages):
    while True:
        output = await call_llm(prompt, messages)
        if isinstance(output, str):
            return output
        messages.extend(output)
await agent(prompt, [])
"""

m = pydantic_monty.Monty(
    code,
    inputs=['prompt'],
    script_name='agent.py',
    type_check=True,
    type_check_stubs=type_definitions,
)

result = await pydantic_monty.run_monty_async(
    m,
    inputs={'prompt': 'testing'},
    external_functions={'call_llm': call_llm},
)
```

**Return values.** Monty captures the **last expression** of the snippet as its return value. The pydantic-ai-harness CodeMode docs confirm: "The last expression in the code snippet is automatically captured as the return value — the model does not need to print()." `print()` calls are captured separately via an optional `print_callback`/stdout sink.

**Argument injection (FFI in).** `inputs=[name, ...]` declares the names; `run(inputs={name: value, ...})` provides values. Cross-FFI conversion is bidirectional via PyO3. Supported scalar types are int, float, bool, str, bytes, None; supported containers are list, dict, tuple. Dataclasses cross the boundary as their attribute dict; classes are not yet definable *inside* Monty, but *can* be received from external functions.

**Allowed Python subset (the constraint we sell to the agent).** As of monty 0.0.17, the language does not support class definitions, match statements, context managers, or generators. Standard-library access is limited to a few approved modules — the README states: "Use the standard library (except a few select modules: sys, typing, asyncio, dataclasses (soon), json (soon))." `os`, `pathlib`, `re`, `datetime`, `math` are mentioned across articles but staggered in availability. An agent that tries an unsupported import receives a parse-time error, which it can recover from on a retry — a deliberate design point (Pydantic blog and Talk Python). **No `import *`**, no third-party packages, no network or filesystem unless an external function provides it.

**Async support.** Monty natively supports `async def` and `await`; you call `pydantic_monty.run_monty_async(...)` and provide async external functions.

**Iterative execution and snapshots.** For interactive REPL-style use, `Monty.start(...)` returns either a `FunctionSnapshot` (pause at an external call), an `OsCallSnapshot`, or a `MontyComplete`. Snapshots are serializable via `.dump()`/`load_snapshot()` (postcard-based, single-digit-kB). We **do not** need this in v0.1 — registrations are one-shot — but the snapshot API is the natural path to v0.2 worktrees.

**Errors.** Monty exposes `MontySyntaxError`, `MontyRuntimeError`, and `MontyTypingError` (the names are stable across both monty-js and monty-python bindings per `crates/*/README.md`). Resource-limit violations raise a `MontyRuntimeError`, after which **the heap is undefined** — Monty's DeepWiki Resource-Limits section says bluntly that the execution context must be discarded after a limit hit. For our purposes: catch, discard the Monty instance, return an error string to the agent.

**Surprises that drive the implementation:**

1. **No class declarations.** Registrations are functions or expressions only. The framework cannot expect Pydantic models *inside* Monty; instead, we serialize at the boundary.
2. **Type stubs are separate from inputs.** Type-checking requires `type_check_stubs` to declare both injected inputs and external functions with their signatures. CodeMode uses this; we copy it.
3. **External functions are the *only* I/O.** Every primitive must be passed as an external function — they are NOT importable from inside the snippet.
4. **Heap unsafe after a limit hit.** Never reuse a `Monty` object after a resource error.

### 1.2 Pydantic-AI capabilities and hooks API

Per Kacperwłodarczyk's Medium article (March 2026): "Pydantic AI just shipped the biggest API change since launch. Capabilities, hooks, and agent specs landed in v1.71+." The capabilities abstraction replaces the prior middleware pattern with one composable interface: a *capability* is a class subclassing `AbstractCapability[AgentDepsT]` that the agent calls at construction to gather instructions/tools/settings, and at runtime to fire hooks. The base class lives at `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py`.

**`AbstractCapability` surface** (verified from `abstract.py` per the rendered API docs at pydantic.dev/docs/ai/api/pydantic-ai/capabilities):

```python
class AbstractCapability(ABC, Generic[AgentDepsT]):
    # Spec serialization
    @classmethod
    def get_serialization_name(cls) -> str | None: ...
    @classmethod
    def from_spec(cls, args: Any = (), kwargs: Any = {}) -> AbstractCapability[Any]: ...

    # Ordering and per-run resolution
    def get_ordering(self) -> CapabilityOrdering | None: ...
    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]: ...

    # Static configuration (called once at agent construction)
    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None: ...
    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None: ...
    def get_toolset(self) -> AgentToolset[AgentDepsT] | None: ...
    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]: ...

    # Per-run toolset wrapping
    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None: ...

    # Tool filtering
    async def prepare_tools(self, ctx, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]: ...
    async def prepare_output_tools(self, ctx, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]: ...

    # Run lifecycle
    async def before_run(self, ctx) -> None: ...
    async def after_run(self, ctx, result) -> AgentRunResult[Any]: ...
    async def wrap_run(self, ctx, handler) -> AgentRunResult[Any]: ...
    async def on_run_error(self, ctx, error) -> AgentRunResult[Any]: ...

    # Model-request lifecycle
    async def before_model_request(self, ctx, request_context: ModelRequestContext) -> ModelRequestContext: ...
    async def after_model_request(self, ctx, *, request_context, response: ModelResponse) -> ModelResponse: ...
    # plus wrap_model_request, on_model_request_error, and equivalents for node, tool validate,
    # tool execute, output validate, output process, plus handle_deferred_tool_calls.
```

**The `Hooks()` capability** is the recommended lightweight way to register hooks without subclassing (pydantic.dev/docs/ai/core-concepts/hooks):

```python
from pydantic_ai.capabilities import Hooks
hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx, request_context: ModelRequestContext) -> ModelRequestContext: ...

@hooks.on.before_tool_execute(tools=['send_email'])
async def audit(ctx, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]) -> dict[str, Any]: ...
```

Decorator namespace, verified against the live Hooks documentation: `before_run`, `after_run`, `run`, `run_error`; `before_node_run`, `after_node_run`, `node_run`, `node_run_error`; `before_model_request`, `after_model_request`, `model_request`, `model_request_error`; `before_tool_validate`, `after_tool_validate`, `tool_validate`, `tool_validate_error`; `before_tool_execute`, `after_tool_execute`, `tool_execute`, `tool_execute_error`; `before_output_validate`/`after_output_validate`/`output_validate`/`before_output_process`/`after_output_process`/`output_process`; `prepare_tools`, `prepare_output_tools`; `deferred_tool_calls`; `event`, `run_event_stream`. Decorators accept `timeout=float` and (for tool execute/validate) `tools=Sequence[str]` for name-filtered firing.

**Composition rules.** Per the v1.71 migration article: "Before-hooks chain, after-hooks reverse, wrap-hooks nest. Done." For `before_model_request`, the output of one hook becomes the `request_context` input of the next — so emitting one capability per registration gives us free stacking.

**`ProcessHistory` for compaction.** `pydantic_ai.capabilities.ProcessHistory` wraps a callable `(list[ModelMessage]) -> list[ModelMessage]` (sync or async, optionally accepting `RunContext`) as a `before_model_request` hook (pydantic.dev/docs/ai/core-concepts/message-history). Two warnings: (a) the returned list **replaces** the state's history; (b) the input list includes the *current* run's messages too, so you must keep tool-call/tool-return pairs balanced — issue #2050 against pydantic-ai documents the failure mode.

The `ModelMessage` type is a discriminated union of `ModelRequest` and `ModelResponse`, each with `parts: Sequence[...]`. Parts include `UserPromptPart`, `SystemPromptPart`, `ToolReturnPart`, `TextPart`, `ToolCallPart`, `ThinkingPart`. Pydantic-AI ships `ModelMessagesTypeAdapter = TypeAdapter(list[ModelMessage])`; round-trip is `to_jsonable_python(messages)` → `ModelMessagesTypeAdapter.validate_python(...)`. **We will serialize each message with `to_jsonable_python` and pass `list[dict]` into Monty.** The compaction snippet returns a list of indices to keep — the simplest possible cross-FFI contract.

**Dynamic instructions.** A capability's `get_instructions()` may return a string (static), a `TemplateStr`, or a callable `(RunContext[Deps]) -> str | None` (sync or async). Multiple registered instructions are *concatenated*, with static placed before dynamic in registration order (pydantic-ai docs, agent.md: "Static instructions are always sorted before dynamic ones"). An instruction registration becomes a capability whose `get_instructions()` returns a callable that runs Monty.

**Tool filtering.** `PrepareTools(prepare_func)` accepts `async (ctx, tool_defs) -> list[ToolDefinition] | None`. Returning `None` is "no change"; multiple `PrepareTools` capabilities pipeline naturally.

**Guards.** The cleanest hook is `before_tool_execute(call: ToolCallPart, tool_def: ToolDefinition, args: dict)` returning a `dict[str, Any]` (modified args) or raising `ModelRetry("denied: ...")` to short-circuit. Multiple before-hooks chain; **first to raise wins** — our "first-deny-wins" semantics for free.

### 1.3 The CodeMode bridge (the pattern we will copy)

The CodeMode capability ships in **pydantic-ai-harness 0.2.0**, released April 25, 2026 per the package's PyPI page (pypi.org/project/pydantic-ai-harness/), installable via `uv add "pydantic-ai-harness[code-mode]"` (the harness README and AGENTS.md both confirm `code-mode` is the canonical extra and `codemode` is an alias).

CodeMode wraps every selected tool into a single `run_code` tool. The model never calls tools directly; it writes Monty Python that calls them as functions. The bridge is the load-bearing piece we mirror in `pa/monty_bridge.py`.

Verified-from-source facts about the bridge (`pydantic_ai_harness/code_mode/_toolset.py`, indexed `main` snapshot 763df599, 768 lines):

1. **Names are sanitized.** Non-identifier chars become underscores; leading-digit names get `_` prefix; keywords get `_` suffix. A `sanitized_to_original` dict maps back. Code lives in `_sanitize_tool_name` (lines ~126–137).
2. **Tool signatures are rendered as Python stubs** in two places: (a) injected into the `run_code` tool *description* via `_build_description` (string the model sees), and (b) passed to Monty as `type_check_stubs` via `_build_type_check_stubs`, which always prepends `import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal`.
3. **Dispatch is via Monty's snapshot API**, not a registered `external_functions` map. The harness uses `MontyRepl.feed_start(code, print_callback=capture)` and resolves each `FunctionSnapshot` by looking the name up in `callable_defs`, mapping through `sanitized_to_original`, and invoking through a `ToolManager`. This gives CodeMode REPL-style state preservation between `run_code` calls. For `pa`, the simpler `pydantic_monty.run_monty_async(..., external_functions={name: callable})` is sufficient — registrations are stateless one-shots.
4. **The cross-FFI serializer is `TypeAdapter(ToolReturnContent)`** (`_TOOL_RETURN_CONTENT_TA: TypeAdapter[Any] = TypeAdapter(ToolReturnContent)`), called as `.dump_python(result)` outbound and `.validate_python(result)` inbound. Critically: `dump_python`, not `to_jsonable_python`. Our spec uses `pydantic_core.to_jsonable_python` instead because `ToolReturnContent` is private to pydantic-ai; we restrict our registrations to JSON-compatible scalars/dicts/lists at the FFI boundary.
5. **The `run_code` system prompt** lists every available function as an `async def` (or `def` for `sequential=True` tools) with full signatures, plus a header explaining `await` semantics. The model is told the last expression is the return value.
6. **Errors are surfaced via UserError / ModelRetry.** A deferred tool inside CodeMode raises `UserError("Tool approval and deferral are not supported in code mode")`. We will mirror this pattern: bridge errors map to clean Python exceptions our hook layer converts to `ModelRetry`.

### 1.4 Agent specs and `Agent.from_file`

Pydantic-AI exposes `Agent.from_file('agent.yaml')` (returns a fully wired `Agent`) and `Agent.from_spec(dict_or_AgentSpec, custom_capability_types=[...], deps_type=...)` (pydantic.dev/docs/ai/core-concepts/agent-spec). YAML shape:

```yaml
model: anthropic:claude-sonnet-4-6
instructions: You are a helpful assistant.
capabilities:
  - WebSearch
  - Thinking: {effort: high}
  - CodeMode: {}
  - PaRegistrations: {}
```

Custom capabilities must implement `get_serialization_name()` (the YAML key) and `from_spec(args, kwargs)` classmethod. Dataclass-style capabilities with serializable arguments use the defaults. We pass our custom capability via `custom_capability_types=[PaRegistrations, CodeMode]`.

### 1.5 pydantic-deep comparison (one paragraph)

vstorm-co's `pydantic-deep` / `pydantic-deepagents` ships a `SkillsToolset` that loads `SKILL.md` files from a directory tree (`./skills/<name>/SKILL.md`) with progressive disclosure (list/load/run_script). Skills are *authored by humans, deployed alongside the agent, and discovered at runtime*. This is the opposite of `pa`'s model: **`pa` registrations are authored by the agent, at runtime, as Monty code snippets bound to specific framework hook points.** We are not building skills.

### 1.6 Logfire integration

`logfire.configure()` followed by `logfire.instrument_pydantic_ai()` gives us per-run, per-model-request, and per-tool-call spans (pydantic.dev/docs/ai/integrations/logfire). CodeMode already emits a `run_code` span and nests inner tool-call spans under it (harness README: "Each run_code span fans out into the tool calls the model issued from inside the sandbox"). For per-registration execution, we wrap each `monty_bridge.execute(...)` call in `logfire.span('pa.registration.execute', slot=..., name=...)`. Logfire is *optional* — if not importable, the bridge uses a no-op context manager.

---

## Part 2 — Implementation brief

### 2.1 Project setup

**Python**: 3.11+. **Package manager**: `uv` exclusively. **License**: AGPL-3.0.

#### `pyproject.toml` (verbatim)

```toml
[project]
name = "pa"
version = "0.1.0"
description = "Self-evolving Pydantic-AI agent harness. Agents extend themselves by writing Monty registrations."
readme = "README.md"
license = { text = "AGPL-3.0-or-later" }
requires-python = ">=3.11"
authors = [{ name = "Bill Easton", email = "bill@pydantic.dev" }]
dependencies = [
    "pydantic-ai-slim[anthropic,openai,logfire]>=1.95.1",
    "pydantic-ai-harness[code-mode]>=0.2.0",
    "pydantic-monty>=0.0.17",
    "pydantic>=2.10",
    "pyyaml>=6.0.2",
    "typer>=0.15.0",
    "rich>=13.9",
    "httpx>=0.27",
]

[project.optional-dependencies]
logfire = ["logfire>=3.0"]
dev = ["pytest>=8.3", "pytest-asyncio>=0.24", "ruff>=0.7"]

[project.scripts]
pa = "pa.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pa"]

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "-ra --strict-markers"
```

A `LICENSE` file with the full AGPL-3.0 text MUST be committed (canonical SPDX text).

### 2.2 Directory tree

```
pa/
  __init__.py
  manifest.py
  monty_bridge.py
  slots.py
  registrations.py
  capability.py
  primitives.py
  registration_tools.py
  runtime.py
  cli.py
  __main__.py
  agent_template.yaml
tests/
  conftest.py
  test_manifest.py
  test_monty_bridge.py
  test_registration_instruction.py
  test_registration_compaction.py
  test_registration_guard.py
  test_registration_tool_filter.py
  test_self_improvement.py
pyproject.toml
LICENSE
README.md
```

**User project layout** (created by `pa init`):

```
my-agent/
  agent.yaml
  pa/
    registrations.yaml
  .pa/
    runs/<id>/                  # v0.2, not written in v0.1
```

### 2.3 `pa/__init__.py`

**Purpose**: version + public exports.

```python
"""pa — Self-evolving Pydantic-AI agent harness."""
from pa.manifest import Manifest, Registration, ManifestError, CardinalityError
from pa.runtime import build_agent
from pa.capability import PaRegistrations

__version__ = "0.1.0"
__all__ = [
    "Manifest", "Registration", "ManifestError", "CardinalityError",
    "PaRegistrations", "build_agent", "__version__",
]
```

**Acceptance**: `python -c "import pa; print(pa.__version__)"` prints `0.1.0`.

### 2.4 `pa/slots.py`

**Purpose**: source-of-truth schema for the four slots.

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal

SlotName = Literal["instruction", "compaction", "guard", "tool_filter"]

class Cardinality(str, Enum):
    ONE = "one"
    MANY = "many"

@dataclass(frozen=True)
class SlotDef:
    name: SlotName
    cardinality: Cardinality
    return_shape: str               # human-readable; enforced in monty_bridge
    description: str
    inputs: tuple[str, ...]         # variable names injected into the snippet

SLOTS: dict[SlotName, SlotDef] = {
    "instruction": SlotDef(
        name="instruction", cardinality=Cardinality.MANY,
        return_shape="str", inputs=("ctx_summary",),
        description="Returns a string appended to dynamic instructions before each model request.",
    ),
    "compaction": SlotDef(
        name="compaction", cardinality=Cardinality.ONE,
        return_shape="list[int]", inputs=("messages",),
        description="Receives messages: list[dict] (jsonable ModelMessage). Returns list of indices to keep.",
    ),
    "guard": SlotDef(
        name="guard", cardinality=Cardinality.MANY,
        return_shape="dict[str, Any]", inputs=("tool_name", "args"),
        description=("Receives the about-to-execute tool call. Returns "
                     "{'action': 'allow'} | {'action': 'deny', 'reason': str} | "
                     "{'action': 'modify', 'args': dict}. First deny wins."),
    ),
    "tool_filter": SlotDef(
        name="tool_filter", cardinality=Cardinality.MANY,
        return_shape="list[str]", inputs=("tool_names",),
        description="Receives tool_names: list[str]. Returns the filtered list of names to keep. Pipelines.",
    ),
}

SLOT_NAMES: tuple[SlotName, ...] = ("instruction", "compaction", "guard", "tool_filter")
```

**Acceptance**: `SLOTS["compaction"].cardinality is Cardinality.ONE`; iteration over `SLOT_NAMES` matches the four slot types.

**Dependencies**: none.

### 2.5 `pa/manifest.py`

**Purpose**: load, validate, save the YAML manifest; enforce cardinality.

```python
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError
from pa.slots import SLOTS, SLOT_NAMES, SlotName, Cardinality

MANIFEST_PATH_DEFAULT = Path("pa") / "registrations.yaml"

class ManifestError(Exception): ...
class CardinalityError(ManifestError): ...

class Registration(BaseModel):
    slot: SlotName
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    code: str = Field(min_length=1, max_length=8000)

class Manifest(BaseModel):
    registrations: list[Registration] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str = MANIFEST_PATH_DEFAULT) -> "Manifest":
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ManifestError(f"{p}: top-level YAML must be a mapping")
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise ManifestError(f"{p}: invalid schema:\n{e}") from e

    def save(self, path: Path | str = MANIFEST_PATH_DEFAULT) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="python")
        p.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))

    def by_slot(self, slot: SlotName) -> list[Registration]:
        return [r for r in self.registrations if r.slot == slot]

    def find(self, name: str) -> Registration | None:
        return next((r for r in self.registrations if r.name == name), None)

    def add(self, reg: Registration) -> None:
        if self.find(reg.name) is not None:
            raise ManifestError(
                f"a registration named {reg.name!r} already exists; "
                f"call remove_registration({reg.name!r}) first."
            )
        slot_def = SLOTS[reg.slot]
        if slot_def.cardinality is Cardinality.ONE and self.by_slot(reg.slot):
            existing = self.by_slot(reg.slot)[0]
            raise CardinalityError(
                f"slot {reg.slot!r} is single-cardinality and already has "
                f"registration {existing.name!r}; "
                f"call remove_registration({existing.name!r}) first."
            )
        self.registrations.append(reg)

    def remove(self, name: str) -> Registration:
        for i, r in enumerate(self.registrations):
            if r.name == name:
                return self.registrations.pop(i)
        raise ManifestError(f"no registration named {name!r}")
```

**Implementation notes**:

- Name pattern `^[a-z][a-z0-9_]*$` forces snake_case, lowercase, no leading digit — safe for filenames, logs, and Logfire span attributes.
- `save()` writes block style YAML so code blocks render as readable multi-line literal strings.
- No file locking in v0.1. Interactive REPL is single-threaded; concurrent runs are v0.2.

**Acceptance** (`tests/test_manifest.py`): (1) round-trip three registrations; (2) two `compaction` adds raise `CardinalityError`; (3) same name across slots raises `ManifestError`; (4) `"My-Reg"` raises `ValidationError`; (5) loading nonexistent path returns empty `Manifest`.

**Dependencies**: `pa/slots.py`.

### 2.6 `pa/monty_bridge.py` — the load-bearing piece

**Purpose**: a small, opinionated wrapper around `pydantic-monty` that takes Python inputs, runs a registration's code with a strict external-functions surface, validates the return shape, and maps Monty exceptions to Python exceptions our framework can react to.

```python
from __future__ import annotations
import asyncio
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

class MontyBridgeError(Exception): ...
class MontySyntaxBridgeError(MontyBridgeError): ...
class MontyRuntimeBridgeError(MontyBridgeError): ...
class MontyReturnShapeError(MontyBridgeError): ...

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
        raise MontyBridgeError(
            f"slot {slot!r} expects inputs {slot_def.inputs}; got {tuple(inputs)}"
        )
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
    def print_callback(s: str) -> None: stdout_buf.append(s)

    t0 = asyncio.get_event_loop().time()
    try:
        result = await pm.run_monty_async(
            m,
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
    raise MontyBridgeError(f"unknown slot {slot!r}")
```

**Implementation notes**:

- We **never reuse** a `Monty` instance — per Monty docs, the heap is undefined after a resource-limit hit. A fresh instance per execution is cheap (single-digit microseconds).
- The serializer is `pydantic_core.to_jsonable_python`. Unlike CodeMode's `TypeAdapter(ToolReturnContent)`, we restrict ourselves to JSON-shaped values. Sufficient for all four slots and keeps the bridge tiny.
- We make NO promises about thread safety. The bridge runs on the agent's event loop.

**Acceptance** (`tests/test_monty_bridge.py`):

1. **Happy path**: `slot="instruction"`, `code='"hi from monty"'`, `inputs={"ctx_summary": {}}` → `value == "hi from monty"`.
2. **Syntax error**: `code='def foo)'` → `MontySyntaxBridgeError`.
3. **Runtime error**: `code='1/0'` → `MontyRuntimeBridgeError`.
4. **Wrong return shape**: `code='42'` for `instruction` → `MontyReturnShapeError`.
5. **Sandbox isolation**: `code='import os\nos.listdir(".")'` → `MontySyntaxBridgeError` (Monty rejects the import).
6. **External function**: `code='double(x)'`, `inputs={"x":5}`, `external_functions={'double': lambda x: x*2}` → `value == 10`. (Use a custom slot fixture for this test, or hand-roll a stripped wrapper that bypasses slot validation; the cleanest version is a separate `_execute` helper called by tests.)
7. **Timeout**: `code='while True: pass'`, `limits=ResourceLimits(max_duration_secs=0.05)` → `MontyRuntimeBridgeError` within ~100ms.

**Dependencies**: `pa/slots.py`, `pydantic-monty`, `pydantic-core`.

### 2.7 `pa/registrations.py`

**Purpose**: for each slot, the factory that turns a `Registration` into the framework artifact.

```python
from __future__ import annotations
from typing import Any, Awaitable, Callable
from pydantic_ai import RunContext, ModelMessage
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.exceptions import ModelRetry
from pydantic_core import to_jsonable_python

from pa.manifest import Registration
from pa.monty_bridge import execute_registration, MontyBridgeError

def make_instruction_fn(reg: Registration) -> Callable[[RunContext[Any]], Awaitable[str | None]]:
    async def _instruction(ctx: RunContext[Any]) -> str | None:
        ctx_summary = {
            "agent_name": getattr(ctx.agent, "name", None) if ctx.agent else None,
            "run_step": ctx.run_step,
        }
        try:
            res = await execute_registration(
                slot="instruction", name=reg.name, code=reg.code,
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
                slot="compaction", name=reg.name, code=reg.code,
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
        ctx: RunContext[Any], *,
        call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            res = await execute_registration(
                slot="guard", name=reg.name, code=reg.code,
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

def make_tool_filter_fn(reg: Registration):
    async def _filter(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        names = [td.name for td in tool_defs]
        try:
            res = await execute_registration(
                slot="tool_filter", name=reg.name, code=reg.code,
                inputs={"tool_names": names},
            )
        except MontyBridgeError:
            return tool_defs  # fail-safe: do not hide tools on bridge error
        keep = set(res.value)
        return [td for td in tool_defs if td.name in keep]
    _filter.__name__ = f"pa_tool_filter_{reg.name}"
    return _filter
```

**Implementation notes**:

- **Fail-safe defaults**: compaction failure returns original history; tool-filter failure keeps all tools; guard failure raises `ModelRetry` (loud-fail — the agent sees and can self-correct).
- **First-deny-wins**: because each guard is its own `@hooks.on.before_tool_execute`, hooks fire in registration order, and `ModelRetry` short-circuits.
- **Tool-filter pipeline**: each filter is its own `PrepareTools` capability; pydantic-ai pipelines them.
- **Compaction cardinality (ONE)**: the single registered compaction wraps into one `ProcessHistory(callable)` capability.

**Dependencies**: `pa/manifest.py`, `pa/monty_bridge.py`, `pydantic_ai`, `pydantic_core`.

### 2.8 `pa/capability.py` — `PaRegistrations`

**Purpose**: the custom capability that wires every registration into Pydantic-AI on agent construction.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory, PrepareTools, Hooks

from pa.manifest import Manifest, MANIFEST_PATH_DEFAULT
from pa.registrations import (
    make_instruction_fn, make_compaction_fn, make_guard_hook, make_tool_filter_fn,
)

try:
    from pydantic_ai.capabilities import CombinedCapability  # type: ignore
    _HAS_COMBINED = True
except ImportError:
    CombinedCapability = None  # type: ignore
    _HAS_COMBINED = False


@dataclass
class _InstructionBundle(AbstractCapability[Any]):
    fns: list[Callable[..., Any]]
    def get_instructions(self):
        # get_instructions accepts a sequence; pydantic-ai concatenates them.
        return list(self.fns)


@dataclass
class PaRegistrations(AbstractCapability[Any]):
    """Loads ./pa/registrations.yaml and wires every entry into the agent."""
    manifest_path: str = str(MANIFEST_PATH_DEFAULT)
    _sub: list[AbstractCapability[Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        manifest = Manifest.load(self.manifest_path)
        self._sub = self._build(manifest)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return "PaRegistrations"

    def _build(self, manifest: Manifest) -> list[AbstractCapability[Any]]:
        caps: list[AbstractCapability[Any]] = []

        inst_fns = [make_instruction_fn(r) for r in manifest.by_slot("instruction")]
        if inst_fns:
            caps.append(_InstructionBundle(fns=inst_fns))

        comp = manifest.by_slot("compaction")
        if comp:
            assert len(comp) == 1
            caps.append(ProcessHistory(make_compaction_fn(comp[0])))

        guards = manifest.by_slot("guard")
        if guards:
            hooks = Hooks()
            for r in guards:
                hooks.on.before_tool_execute(make_guard_hook(r))
            caps.append(hooks)

        for r in manifest.by_slot("tool_filter"):
            caps.append(PrepareTools(make_tool_filter_fn(r)))

        return caps

    def apply(self, visitor):
        for cap in self._sub:
            visitor(cap)
        visitor(self)
```

**Composition decision**: If `CombinedCapability` is importable on the installed pydantic-ai (it is per docs as of v1.71+), wrap `self._sub` in `CombinedCapability(self._sub)` from a single returned capability. The `apply()`-visitor pattern above is the conservative fallback that works either way.

**Acceptance**:

- Empty manifest: `PaRegistrations()` constructs, contributes zero hooks.
- One instruction registration: `Agent(..., capabilities=[PaRegistrations()]).run_sync("hi")` shows the instruction appended to `result.all_messages()[0].instructions`.

**Dependencies**: `pa/manifest.py`, `pa/registrations.py`, `pa/slots.py`, `pydantic_ai.capabilities`.

### 2.9 `pa/primitives.py`

**Purpose**: the four user-provided primitive tools. Registered as `@agent.tool_plain` in `runtime.py`; kept here as free functions for testability.

```python
from __future__ import annotations
import asyncio
from pathlib import Path
import httpx

_DEFAULT_TIMEOUT_S = 30.0

async def read_file(path: str) -> str:
    """Read a UTF-8 text file. Returns its contents."""
    return Path(path).read_text(encoding="utf-8")

async def write_file(path: str, content: str) -> str:
    """Write `content` to `path` (UTF-8, overwriting). Returns a status string."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content.encode('utf-8'))} bytes to {path}"

async def bash(command: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    """Run a bash command; return {'stdout', 'stderr', 'returncode'}."""
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        return {"stdout": "", "stderr": f"timeout after {timeout_s}s", "returncode": 124}
    return {
        "stdout": out.decode("utf-8", errors="replace"),
        "stderr": err.decode("utf-8", errors="replace"),
        "returncode": proc.returncode if proc.returncode is not None else -1,
    }

async def http_get(url: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    """HTTP GET. Returns {'status', 'body', 'content_type'}; body truncated to 256 KB."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(url)
    body = resp.text
    if len(body) > 256 * 1024:
        body = body[: 256 * 1024] + "\n...[truncated]"
    return {"status": resp.status_code, "body": body, "content_type": resp.headers.get("content-type", "")}
```

### 2.10 `pa/registration_tools.py`

**Purpose**: the six registration-management tools the agent uses to extend itself. Each writes to manifest then returns a deterministic success/error string.

```python
from __future__ import annotations
import json
from pa.manifest import Manifest, Registration, ManifestError, CardinalityError, MANIFEST_PATH_DEFAULT
from pa.slots import SlotName

def _load() -> Manifest:
    return Manifest.load(MANIFEST_PATH_DEFAULT)

def _save(m: Manifest) -> None:
    m.save(MANIFEST_PATH_DEFAULT)

def _register(slot: SlotName, name: str, code: str) -> str:
    try:
        reg = Registration(slot=slot, name=name, code=code)
    except Exception as e:
        return f"ERROR: invalid registration: {e}"
    m = _load()
    try:
        m.add(reg)
    except CardinalityError as e:
        return f"ERROR: cardinality: {e}"
    except ManifestError as e:
        return f"ERROR: {e}"
    _save(m)
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

def list_registrations() -> str:
    """Return a JSON-encoded list of {slot, name, lines, preview} entries."""
    m = _load()
    out = []
    for r in m.registrations:
        out.append({
            "slot": r.slot, "name": r.name,
            "lines": r.code.count("\n") + 1,
            "preview": r.code[:120] + ("..." if len(r.code) > 120 else ""),
        })
    return json.dumps(out, indent=2)

def remove_registration(name: str) -> str:
    """Remove the registration with the given name."""
    m = _load()
    try:
        removed = m.remove(name)
    except ManifestError as e:
        return f"ERROR: {e}"
    _save(m)
    return f"OK: removed {removed.slot}/{removed.name}."
```

**Implementation notes**:

- Return strings, not exceptions, so the agent reads them inside `run_code` and reacts. Exceptions would surface as run failures.
- "It will be active on next agent run" wording is deliberate and correct for v0.1: registrations load at agent-construction time. Mid-session reload is v0.2.

**Dependencies**: `pa/manifest.py`, `pa/slots.py`.

### 2.11 `pa/runtime.py`

**Purpose**: `build_agent()` — the one-call entry point.

```python
from __future__ import annotations
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

from pa.capability import PaRegistrations
from pa import primitives, registration_tools

DEFAULT_AGENT_SPEC = Path("agent.yaml")

def build_agent(agent_spec_path: str | Path = DEFAULT_AGENT_SPEC) -> Agent:
    agent: Agent = Agent.from_file(
        str(agent_spec_path),
        custom_capability_types=[PaRegistrations, CodeMode],
    )
    # Primitives
    for fn in (primitives.read_file, primitives.write_file,
               primitives.bash, primitives.http_get):
        agent.tool_plain(fn)
    # Registration-management tools
    for fn in (registration_tools.register_instruction,
               registration_tools.register_compaction,
               registration_tools.register_guard,
               registration_tools.register_tool_filter,
               registration_tools.list_registrations,
               registration_tools.remove_registration):
        agent.tool_plain(fn)
    return agent
```

**Why this order works**: `Agent.from_file` constructs the agent with `CodeMode()` and `PaRegistrations()` already in its capability list (from YAML); then we register the ten tools as `@agent.tool_plain`. CodeMode wraps every tool registered before its toolset materializes — and `tool_plain` decoration mutates the agent's default toolset, so CodeMode sees all ten and bundles them into `run_code`. The model only ever sees `run_code`.

**Dependencies**: `pa/capability.py`, `pa/primitives.py`, `pa/registration_tools.py`, `pydantic_ai`, `pydantic_ai_harness`.

### 2.12 `pa/agent_template.yaml`

```yaml
# agent.yaml — written by `pa init`. Edit freely.
model: anthropic:claude-sonnet-4-6
name: pa-agent
instructions: |
  You are a pa agent — a self-evolving Pydantic-AI agent.

  You have four primitive tools (read_file, write_file, bash, http_get) and six
  registration-management tools (register_instruction, register_compaction,
  register_guard, register_tool_filter, list_registrations, remove_registration).

  Because CodeMode is active, you never call tools directly. Instead, you write
  Monty Python (a sandboxed subset of Python) inside `run_code(code=...)` that
  calls the tools as functions. The last expression in your snippet is the
  return value.

  You can extend your own behavior at runtime by calling one of the register_*
  tools. Each registration is a single Monty snippet bound to a hook:
    - instruction: snippet returns a string appended to the system prompt.
    - compaction: snippet receives `messages` (list of dicts) and returns a
        list of indices to keep. Cardinality: ONE.
    - guard: snippet receives `tool_name, args` and returns
        {'action': 'allow' | 'deny' | 'modify', ...}. First deny wins.
    - tool_filter: snippet receives `tool_names` and returns the filtered list.

  Registrations are written to ./pa/registrations.yaml and take effect on the
  next agent run. Registrations CANNOT call each other; they are standalone.
  If a single-cardinality slot is occupied, remove the existing registration
  first via remove_registration(name).

capabilities:
  - CodeMode: {}
  - PaRegistrations: {}
```

### 2.13 `pa/cli.py`

**Purpose**: Typer app exposing `pa init`, `pa run "..."`, `pa repl`.

```python
from __future__ import annotations
from pathlib import Path
import shutil
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from pa.runtime import build_agent

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="pa — self-evolving Pydantic-AI agent harness.")
console = Console()

_TEMPLATE = Path(__file__).parent / "agent_template.yaml"

@app.command()
def init() -> None:
    """Create agent.yaml and pa/registrations.yaml in the current directory."""
    target = Path("agent.yaml")
    if target.exists():
        console.print(f"[yellow]agent.yaml already exists — leaving untouched.[/yellow]")
    else:
        shutil.copyfile(_TEMPLATE, target)
        console.print(f"[green]wrote {target}[/green]")
    reg_dir = Path("pa")
    reg_dir.mkdir(exist_ok=True)
    reg_path = reg_dir / "registrations.yaml"
    if not reg_path.exists():
        reg_path.write_text("registrations: []\n")
        console.print(f"[green]wrote {reg_path}[/green]")

@app.command()
def run(prompt: str) -> None:
    """Run the agent once with the given prompt."""
    _try_logfire()
    agent = build_agent()
    result = agent.run_sync(prompt)
    console.print(Panel(str(result.output), title="agent"))

@app.command()
def repl() -> None:
    """Interactive single-session REPL with persistent message history."""
    _try_logfire()
    agent = build_agent()
    console.print("[bold]pa repl[/bold] — type /exit to quit, /list to list registrations.")
    history = []
    while True:
        try:
            line = Prompt.ask("[cyan]>[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye"); return
        if line.strip() == "/exit":
            return
        if line.strip() == "/list":
            from pa.registration_tools import list_registrations
            console.print(list_registrations()); continue
        result = agent.run_sync(line, message_history=history)
        history = result.all_messages()
        console.print(Panel(str(result.output), title="agent"))

def _try_logfire() -> None:
    try:
        import logfire  # type: ignore
        logfire.configure(send_to_logfire="if-token-present")
        logfire.instrument_pydantic_ai()
    except ImportError:
        pass

if __name__ == "__main__":
    app()
```

### 2.14 `pa/__main__.py`

```python
from pa.cli import app
app()
```

### 2.15 `tests/conftest.py`

```python
from pathlib import Path
import pytest

@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Isolate each test in a fresh CWD so pa/registrations.yaml does not leak."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pa").mkdir()
    return tmp_path
```

### 2.16 Tests (file-by-file)

**`tests/test_manifest.py`** — five tests as listed under 2.5.

**`tests/test_monty_bridge.py`** — seven tests as listed under 2.6.

**`tests/test_registration_instruction.py`**:

```python
import pytest
from pa.manifest import Manifest, Registration
from pa.capability import PaRegistrations
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

@pytest.mark.asyncio
async def test_instruction_registration_in_request(tmp_cwd):
    m = Manifest()
    m.add(Registration(slot="instruction", name="cheerio",
                       code='"Always end your responses with Cheerio!"'))
    m.save()
    agent = Agent(TestModel(), capabilities=[PaRegistrations()])
    result = await agent.run("hi")
    instr = result.all_messages()[0].instructions or ""
    assert "Cheerio" in instr
```

**`tests/test_registration_compaction.py`** — register `code="[len(messages) - 1]"`, run two prompts, assert the second model request was preceded by a one-message history.

**`tests/test_registration_guard.py`** — register a guard with `code='{"action": "deny", "reason": "no bash"} if tool_name == "bash" else {"action": "allow"}'`. Use a FunctionModel scripted to call `bash` and assert a `ModelRetry` appears in the message history.

**`tests/test_registration_tool_filter.py`** — register a filter with `code="[n for n in tool_names if n != \"bash\"]"`. Assert via `TestModel.last_model_request_parameters` that `bash` was not in the tool list.

**`tests/test_self_improvement.py`** — the critical end-to-end test. Use a FunctionModel scripted to: (1) call `run_code(code='register_instruction("cheerio", "\\"end with Cheerio!\\"")')`; (2) return a final text result. After the run, assert `pa/registrations.yaml` contains the registration. Then build a *second* agent (simulating restart), run it, assert the new request's instructions contain `"Cheerio"`.

### 2.17 Acceptance demo (live proof, 5 steps)

```bash
# 1. Initialize a fresh agent in an empty dir.
mkdir my-agent && cd my-agent
pa init
# expect: agent.yaml + pa/registrations.yaml written

# 2. Start the REPL.
pa repl

# 3. At the prompt:
> Register an instruction named "cheerio" that makes you always end
  responses with the word "Cheerio!". Then confirm it's saved.

#    expected:
#    - model writes run_code(code='register_instruction("cheerio",
#      \'"Always end your reply with the word Cheerio!"\')')
#    - tool returns "OK: registered instruction/cheerio. It will be
#      active on next agent run."
#    - model summarizes: "Done — registered as 'cheerio'. Restart to activate."

# 4. Verify the manifest.
cat pa/registrations.yaml
# expect:
# registrations:
#   - slot: instruction
#     name: cheerio
#     code: |
#       "Always end your reply with the word Cheerio!"

# 5. Restart and observe self-improvement.
> /exit
pa repl
> What is 2+2?
# expect: "2+2 equals 4. Cheerio!"
```

If step 5 produces a response ending in "Cheerio!", the harness is end-to-end working. This is the minimum-viable self-improvement loop.

### 2.18 Effort estimate (junior dev, 2 days = 16 hours)

| Hour | Task | Checkpoint |
|------|------|-----------|
| 0–1 | `uv init`, `pyproject.toml`, `__init__.py`, `slots.py`, LICENSE, README skeleton | `python -c "from pa.slots import SLOTS"` works |
| 1–3 | `manifest.py` + all five tests | `pytest tests/test_manifest.py` green |
| 3–6 | `monty_bridge.py` + seven tests | `pytest tests/test_monty_bridge.py` green (the hard one) |
| 6–8 | `primitives.py`, `registration_tools.py`, `agent_template.yaml` | quick sanity import & call each |
| 8–11 | `registrations.py` + `capability.py` | unit-test against `TestModel` |
| 11–13 | `runtime.py`, `cli.py`, `__main__.py` | `pa init && pa repl` enters the loop |
| 13–15 | Four end-to-end registration tests | each green |
| 15–16 | `test_self_improvement.py`, polish, README screenshot | full `pytest -x` green; demo runs |

**Hour-6 checkpoint is the go/no-go gate.** If the Monty bridge is not solid by then, no further file makes sense.

### 2.19 Risks the junior must escalate IMMEDIATELY

1. **`pydantic_monty.Monty(...)` signature differs.** If `inputs=`, `type_check_stubs=`, or `script_name=` raise `TypeError` on construction, *stop* and ask. Monty is pre-1.0. Verify with `python -c "import pydantic_monty; help(pydantic_monty.Monty)"`.
2. **`run_monty_async(m, inputs=..., external_functions=..., limits=..., print_callback=...)` differs.** Same drill. The entry point may be `Monty.run_async(...)` on the instance.
3. **`AbstractCapability` does not expose `apply()` or `CombinedCapability` is not importable.** Fall back to emitting each sub-capability as a top-level YAML entry rather than wrapping in `PaRegistrations`. Escalate before changing the design.
4. **`Hooks().on.before_tool_execute(fn)` does not accept a bare callable.** Wrap each guard hook in a closure that produces a decorated function. The docs show both decorator and constructor-kwarg forms; if only one works on the installed version, use that.
5. **`Agent.from_file` lacks `custom_capability_types=` kw.** Older API used a separate registry. Check the installed version's docstring before assuming.
6. **`ProcessHistory` rejects async callables.** If so, wrap our async `make_compaction_fn` in a sync shim via `asyncio.run`. It should accept async per the docs.

Each is a documented likely-stable surface but, given pre-1.0 versioning, must be verified against the actual installed package, not against this document.

### 2.20 README skeleton

```markdown
# pa

Self-evolving Pydantic-AI agent harness. An agent extends its own behavior at
runtime by writing Monty (sandboxed Python) snippets bound to Pydantic-AI hook
points.

## Quick start

    uv add pa
    pa init
    pa repl

Then ask the agent to extend itself:

    > Register an instruction that makes you always sign off with "Cheerio!"

## Concepts

- **Registration**: a named Monty code snippet bound to a slot.
- **Slot**: one of `instruction`, `compaction`, `guard`, `tool_filter`.
- **Cardinality**: `compaction` is single; the other three stack.
- All registrations are standalone; composition happens at the framework layer.

## License

AGPL-3.0.
```

### 2.21 Logfire instrumentation strategy

`_try_logfire()` in `pa/cli.py` is the only entry point. We do not require Logfire as a dependency (runtime-optional). For per-registration spans, the bridge wraps `pm.run_monty_async` in `logfire.span("pa.registration.execute", slot=slot, name=name)` *iff* logfire is importable:

```python
try:
    import logfire as _logfire  # type: ignore
    _span_ctx = lambda **kw: _logfire.span("pa.registration.execute", **kw)
except ImportError:
    from contextlib import nullcontext
    _span_ctx = lambda **kw: nullcontext()
```

Each registration execution becomes a child span of the agent's `run_code` span, which nests inside the model-request span. This mirrors the hierarchy pydantic-ai-harness already emits for CodeMode tool calls.

### 2.22 Out of scope for v0.1 (explicit)

- Dispatcher / worktrees / multi-session — **v0.2.**
- Snapshot persistence — **v0.2.**
- Hot-reload — registrations apply on next `build_agent()`. **No mid-session reload.**
- Cross-registration calls — **forbidden by design.**
- Registration introspection during execution — each Monty execution sees only its declared inputs.

---

## Closing note

Every load-bearing decision in this spec is grounded in source: Monty's Python API (PyPI README and the monty README, version 0.0.17 released April 22, 2026), pydantic-ai-harness's CodeMode bridge (verbatim from `_toolset.py` in pydantic-ai-harness 0.2.0 released April 25, 2026), Pydantic-AI's `AbstractCapability` and `Hooks` surface (the v1.71+ docs), and the `Agent.from_file` / custom-capability-types contract. The four slot semantics map one-to-one onto existing hooks: instruction → `get_instructions` callable; compaction → `ProcessHistory`; guard → `Hooks().on.before_tool_execute`; tool_filter → `PrepareTools`. Nothing new is being invented at the framework layer — `pa` is a twelve-file, roughly 1,000-line shim that turns "I want my agent to learn this" into a YAML row.