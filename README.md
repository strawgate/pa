# pa

Self-evolving Pydantic-AI agent harness. The agent extends its own toolset at
runtime by writing Monty (sandboxed Python) snippets that persist across runs.

**Requirements:** Python 3.11+

## Quick start

```bash
pip install pa
pa init          # creates ~/.pa/agent.yaml, forks ./agent.yaml, and creates local state
pa run "your prompt"
pa repl          # interactive REPL with history
```

## How it works

**Two-tier tool architecture:**
- **Sandboxed primitives** (`run_code`): `read_file`, `write_file`, `bash`,
  `list_dir`, `http_get`, `complete` — available as async functions inside a Monty
  sandbox. State persists between calls (REPL-style).
- **Native tools**: registration functions and user-defined tools registered
  from prior runs — callable directly, visible in the agent's tool list.

**Self-evolution:** the agent can call
`register_tool(name, description, code, parameters_json_schema, example_args, timeout_s=10)`
to create a persistent tool. Tools without an example are saved as drafts.
Only validated active tools appear as native tools on the next run.
Use `timeout_s` for tools that legitimately need more sandbox time; it is
capped at 60 seconds.

Agent-facing registration tools:
- `register_tool` — save a proven repeatable operation as a native tool
- `validate_tool` — promote a draft tool after a concrete example works
- `register_instruction` — remember durable preferences, project conventions,
  or workflow guidance
- `register_before_run_hook` — run once at the start of each run and inject
  run-local guidance
- `register_after_run_hook` — run once at the end of each run and optionally
  replace final output
- `register_before_tool_hook` — allow, deny, or modify a tool call before it runs
- `register_after_tool_hook` — allow, retry, or modify a tool result after it runs
- `register_compaction` — choose history message indices to keep
- `register_tool_filter` — filter available primitive tools
- `list_registrations` / `check_registrations` — inspect registration health
- `disable_registration` / `remove_registration` — quarantine or remove bad
  registrations

**`complete()`** allows the agent to call its own model for sub-tasks
(summarization, code generation, structured extraction).

Registrations persist in pa's local state directory for the current project.
Run `pa state path` to see the exact directory.

For detailed registration patterns and gotchas, see
[`docs/registrations.md`](docs/registrations.md). `pa init` also writes this
guide into new projects so the agent can read it with
`read_file(path="docs/registrations.md")`.

**Retry budgets:** Pydantic AI raises `UnexpectedModelBehavior` and aborts the
current agent run when a tool exhausts its retry budget. pa configures
`run_code`, registration-management tools, and active registered tools with a
budget of 15 retries. Schema validation errors, hook denials, bad hook outputs,
and after-tool `retry` responses all count toward the relevant native tool's
budget, so broken self-evolution should be inspected or disabled rather than
retried blindly.

**Observability**

**Progress events:** `pa run --jsonl` emits structured JSON lines for every
significant event in an agent run, useful for logging and integration:

```json
{"type": "run_started", "prompt": "hello", "history_messages": 5}
{"type": "tool_call_started", "tool_name": "run_code", "args_summary": "code=..."}
{"type": "tool_call_finished", "tool_name": "run_code", "outcome": "success", "result_summary": "..."}
{"type": "history_saved", "phase": "step", "messages": 7}
{"type": "run_completed", "output": "Hello!", "history_messages": 7}
```

When the `logfire` package is installed, pa automatically instruments Pydantic AI
calls for debugging and tracing.

## Concepts

- **Progress events**: structured events emitted during agent runs. Types include
  `run_started`, `tool_call_started`, `tool_call_finished`, `retry_requested`,
  `history_saved`, `run_completed`, `run_failed`. Progress callbacks receive
  typed `ProgressEvent` objects rather than plain strings.
- **Registration**: a named Monty snippet bound to a slot. Created via
  agent-facing tools such as `register_tool`, `validate_tool`,
  `register_instruction`, lifecycle hook registration tools, `register_compaction`,
  or `register_tool_filter`.
- **Slot**: one of `tool`, `instruction`, `compaction`, `before_run_hook`,
  `after_run_hook`, `before_tool_hook`, `after_tool_hook`, or `tool_filter`.
  `compaction` is single-cardinality; the rest stack. Tool registrations have
  `draft`, `active`, or `disabled` status. Disabled registrations remain in the
  manifest but are not wired into Pydantic AI hooks or toolsets.
- **Primitives**: `read_file`, `write_file`, `list_dir`, `bash`, `http_get`,
  `complete`. Sandboxed inside `run_code` via Monty. The `tools` list in
  `agent.yaml` controls which primitives are sandboxed. Registered Monty tools
  can also call these primitives directly; lifecycle hooks see the outer
  registered tool call rather than each inner primitive call.

## Configuration

`pa init` creates `~/.pa/agent.yaml` as the user's default profile if needed,
then forks that file into `./agent.yaml` for the current working directory.
The project fork is user-owned and does not silently sync from the default.
Set `PA_HOME` to override `~/.pa`, which is useful for tests and isolated
workspaces.

`agent.yaml`:

```yaml
model: anthropic:claude-sonnet-4-20250514    # or gateway/<route>:<model>
sdk: anthropic                                 # openai, groq, google-cloud
base_url: https://api.minimax.io/anthropic     # optional direct provider
instructions: |
  You are pa — a self-evolving agent …
capabilities:
  - CodeMode: {max_retries: 15, tools: [read_file, write_file, list_dir, bash, http_get, complete]}
  - PaRegistrations: {}
```

Supported SDKs: `anthropic`, `openai`/`openai-chat`, `groq`, `google-cloud`.
Model string supports `gateway/<route>:<model-name>` for Pydantic-AI Gateway.
Set `base_url` to point directly at a provider API (bypasses the gateway).

Conversation history is saved in the resolved state directory (last 40
messages), giving the agent memory across runs without writing mutable state
into the project by default.

## CLI

```
pa init                    Create the home default, project agent.yaml, docs, and state
pa run <prompt>            Run once, resume from saved history
pa run --no-history <p>    Run ignoring saved history
pa run --no-progress <p>   Suppress compact progress events
pa run --verbose <p>       Include run lifecycle and history-save events
pa run --jsonl <p>         Emit structured progress events as JSON lines
pa repl                    Interactive REPL (/exit, /list /health /clear)
pa repl --no-history       Start with a blank history
pa repl --no-progress      Suppress compact progress events
pa repl --verbose          Include run lifecycle and history-save events
pa doctor                  Smoke-check registration health
pa clear-history           Delete saved history
pa state path              Print this project's state directory
pa state ls                Show resolved config and state files
pa state wipe --history    Delete saved conversation history
pa state wipe --registrations --yes
                           Reset learned registrations
pa state wipe --all --yes  Delete this project's whole state directory
```

## License

AGPL-3.0.
