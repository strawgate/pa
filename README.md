# pa

Self-evolving Pydantic-AI agent harness. The agent extends its own toolset at
runtime by writing Monty (sandboxed Python) snippets that persist across runs.

## Quick start

```bash
uv add pa
pa init          # creates agent.yaml and pa/registrations.yaml
pa run "your prompt"
pa repl          # interactive REPL with history
```

## How it works

**Two-tier tool architecture:**
- **Sandboxed primitives** (`run_code`): `read_file`, `write_file`, `bash`,
  `http_get`, `complete` — available as async functions inside a Monty
  sandbox. State persists between calls (REPL-style).
- **Native tools**: registration functions and user-defined tools registered
  from prior runs — callable directly, visible in the agent's tool list.

**Self-evolution:** the agent can call
`register_tool(name, description, code, parameters_json_schema, example_args)`
to create a persistent tool. Tools without an example are saved as drafts.
Only validated active tools appear as native tools on the next run.

Other hook slots:
- `register_instruction` — dynamic system prompt additions
- `register_compaction` — history compaction (single slot)
- `register_guard` — pre-execution guard on tool calls (allow/deny/modify)
- `register_tool_filter` — filter available primitives with native tool preparation
- `validate_tool` / `disable_tool` — promote or quarantine registered tools
- `list_registrations` / `check_registrations` — inspect registration health
- `disable_registration` / `remove_registration` — manage registrations

**`complete()`** allows the agent to call its own model for sub-tasks
(summarization, code generation, structured extraction).

All registrations persist in `pa/registrations.yaml`.

## Concepts

- **Registration**: a named Monty snippet bound to a slot. Created via
  `register_tool`, `validate_tool`, `register_instruction`,
  `register_compaction`, `register_guard`, or `register_tool_filter`.
- **Slot**: one of `tool`, `instruction`, `compaction`, `guard`, `tool_filter`.
  `compaction` is single-cardinality; the rest stack. Tool registrations have
  `draft`, `active`, or `disabled` status. Disabled registrations remain in the
  manifest but are not wired into Pydantic AI hooks or toolsets.
- **Primitives**: `read_file`, `write_file`, `bash`, `http_get`, `complete`.
  Sandboxed inside `run_code` via Monty. The `tools` list in `agent.yaml`
  controls which primitives are sandboxed.

## Configuration

`agent.yaml` (created by `pa init`):

```yaml
model: anthropic:claude-sonnet-4-20250514    # or gateway/<route>:<model>
sdk: anthropic                                 # openai, groq, google-cloud
base_url: https://api.minimax.io/anthropic     # optional direct provider
instructions: |
  You are pa — a self-evolving agent …
capabilities:
  - CodeMode: {max_retries: 15, tools: [read_file, write_file, bash, http_get, complete]}
  - PaRegistrations: {}
```

Supported SDKs: `anthropic`, `openai`/`openai-chat`, `groq`, `google-cloud`.
Model string supports `gateway/<route>:<model-name>` for Pydantic-AI Gateway.
Set `base_url` to point directly at a provider API (bypasses the gateway).

Conversation history is saved to `pa/history.json` (last 40 messages),
giving the agent memory across runs.

## CLI

```
pa init                    Create agent.yaml and pa/registrations.yaml
pa run <prompt>            Run once, resume from saved history
pa run --no-history <p>    Run ignoring saved history
pa repl                    Interactive REPL (/exit, /list, /health, /clear)
pa doctor                  Smoke-check registration health
pa clear-history           Delete saved history
```

## License

AGPL-3.0.
