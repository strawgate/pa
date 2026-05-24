# Registration Guide

This guide is for pa agents and humans working on self-evolution. If you are an
agent and the task involves registrations, read this file before writing or
repairing them.

## Mental Model

pa has two execution surfaces:

- `run_code`: a native Pydantic AI tool that runs Monty code with sandboxed
  primitives such as `read_file`, `write_file`, `list_dir`, `bash`, `http_get`,
  and `complete`.
- Registrations: persistent Monty snippets in `pa/registrations.yaml`. Some
  become native tools, and others become lifecycle hooks or prompt guidance.

Registration management tools are native tools. Call `register_tool`,
`register_before_tool_hook`, `check_registrations`, and friends directly. Do not
wrap those calls inside `run_code`.

## Monty Snippet Rules

Registration code is a Monty snippet, not a Python module with an entrypoint.
Inputs are injected as variables and the final expression is the return value.

Do:

```python
{"action": "deny", "reason": ".env blocked"} if ".env" in str(args).lower() else {"action": "allow"}
```

Do not:

```python
def hook(tool_name, args):
    return {"action": "allow"}
```

Avoid top-level `return`. Avoid defining classes. Prefer small, direct snippets.
If a snippet needs helpers, use simple top-level assignments and loops, then end
with the value to return.

Primitive calls are async and keyword-only:

```python
content = await read_file(path="README.md")
entries = await list_dir(path=".")
result = await bash(command="pytest -q", timeout_s=30)
```

## Slots

Use the smallest surface that fits the job:

- `instruction`: durable guidance injected into future requests.
- `before_run_hook`: run-local note at the start of a run.
- `after_run_hook`: final output policy, usually `allow`.
- `before_tool_hook`: allow, deny, or modify a native tool call.
- `after_tool_hook`: allow, modify a result, or ask for a retry.
- `compaction`: choose history message indices to keep.
- `tool_filter`: hide CodeMode primitives from `run_code`.
- `tool`: reusable native tool backed by Monty.

Legacy `guard` registrations still work, but prefer `before_tool_hook`.

## Registered Tools

Only register a tool for a proven repeatable operation. Always provide:

- a clear `description`
- a JSON object schema
- real `example_args`

Example schema:

```json
{
  "type": "object",
  "properties": {
    "path": {"type": "string", "default": "."},
    "pattern": {"type": "string"},
    "case_sensitive": {"type": "boolean", "default": true},
    "max_results": {"type": "integer", "default": 50}
  },
  "required": ["pattern"],
  "additionalProperties": false
}
```

Registered tools may call `read_file`, `write_file`, `list_dir`, `bash`,
`http_get`, and `complete`. Hooks see the outer registered tool call, not every
primitive call made inside the tool. If a registered tool is policy-sensitive,
put the policy in the tool itself too.

For filesystem tools, use `list_dir` plus `read_file` instead of shelling out.
Skip sensitive and noisy paths explicitly:

```python
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}
SKIP_NAMES = {".env", ".env.local", ".env.production"}
```

## Hooks

Before-tool hooks receive `tool_name` and `args`. They must return one of:

```python
{"action": "allow"}
{"action": "deny", "reason": "short reason"}
{"action": "modify", "args": new_args}
```

A broad `.env` blocker can inspect serialized arguments:

```python
{"action": "deny", "reason": ".env access blocked"} if ".env" in str(args).lower() else {"action": "allow"}
```

After-tool hooks receive `tool_name`, `args`, and `result`. They must return one
of:

```python
{"action": "allow"}
{"action": "modify", "result": new_result}
{"action": "retry", "reason": "short reason"}
```

Use `retry` sparingly. Every retry consumes the native tool retry budget, and
exhausting that budget aborts the whole agent run.

## Tool Filters

`tool_filter` controls which primitives CodeMode exposes inside `run_code`.
It receives `tool_names` and returns the allowed subset:

```python
[name for name in tool_names if name != "bash"]
```

Important: tool filters hide primitives from `run_code`. They do not rewrite
already-registered tool code. Policy-sensitive registered tools should enforce
their own constraints.

## Compaction

Compaction receives `messages` and returns message indices to keep. Keep the
current request. A conservative policy:

```python
list(range(max(0, len(messages) - 8), len(messages)))
```

## Health And Repair Loop

After adding or changing registrations, call:

```python
check_registrations()
```

If something fails:

1. Read the `last_error`.
2. Decide whether to repair, disable, or remove.
3. Prefer `disable_registration(name, reason)` when behavior is risky or
   unclear.
4. Re-register repaired code under a clear name, or remove the broken entry
   only when you are sure it is not useful.

Broken before/after tool hooks fail open and record health errors. That keeps
the agent alive, but it also means a broken safety hook is not enforcing policy.

## Retry Budget

Pydantic AI aborts the whole agent run when a tool exhausts its retry budget.
pa gives `run_code`, registration-management tools, and registered tools 15
retries. Schema validation errors, hook denials, bad hook return shapes, runtime
errors, and after-tool `retry` responses all count.

If a call fails three times in a row, stop repeating it. Change strategy, inspect
registrations, or disable the broken registration.

## Working Examples

Durable instruction:

```python
"Call check_registrations after changing registrations."
```

Before-run checklist:

```python
"Checklist: inspect registration health before edits."
```

Before-tool `.env` blocker:

```python
{"action": "deny", "reason": ".env access blocked"} if ".env" in str(args).lower() else {"action": "allow"}
```

After-tool nonzero annotation:

```python
{"action": "modify", "result": {**result, "pa_note": "nonzero return"}} if isinstance(result, dict) and result.get("returncode", 0) != 0 else {"action": "allow"}
```

Simple registered tool:

```python
args["text"].strip().lower()
```

One-level grep core:

```python
pattern = args["pattern"]
matches = []
for entry in await list_dir(path=args.get("path", ".")):
    if entry["is_file"] and entry["name"] not in {".env"}:
        text = await read_file(path=entry["path"])
        for line_no, line in enumerate(text.splitlines(), 1):
            if pattern in line:
                matches.append(f"{entry['path']}:{line_no}:{line}")
matches
```
