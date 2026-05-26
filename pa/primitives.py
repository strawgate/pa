from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

_DEFAULT_TIMEOUT_S = 30.0

# Injected at build time by runtime.py — holds a callable that performs a completion.
_complete_fn: Any = None
_ASYNC_PLACEHOLDER_PREFIXES = ("<coroutine ",)
_ASYNC_PLACEHOLDER_MARKERS = ("external_future(",)


async def read_file(*, path: str) -> str:
    """Read a UTF-8 text file at `path`. Relative paths resolve from the current working directory."""
    return Path(path).read_text(encoding="utf-8")


async def write_file(*, path: str, content: str) -> str:
    """Write `content` to `path` as UTF-8, overwriting existing content. Relative paths resolve from cwd."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content.encode('utf-8'))} bytes to {path}"


async def list_dir(*, path: str = ".") -> list[dict[str, object]]:
    """List one directory level at `path`. Relative paths resolve from the current working directory."""
    p = Path(path)
    out: list[dict[str, object]] = []
    for child in sorted(p.iterdir(), key=lambda entry: entry.name):
        try:
            stat = child.stat()
            size: int | None = stat.st_size if child.is_file() else None
        except OSError:
            size = None
        out.append(
            {
                "name": child.name,
                "path": str(child),
                "is_dir": child.is_dir(),
                "is_file": child.is_file(),
                "size": size,
            }
        )
    return out


async def bash(*, command: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    """Run a bash command in the current working directory; return {'stdout', 'stderr', 'returncode'}."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
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


async def http_get(*, url: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    """HTTP GET. Returns {'status', 'body', 'content_type'}; body truncated to 256 KB."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(url)
    body = resp.text
    if len(body) > 256 * 1024:
        body = body[: 256 * 1024] + "\n...[truncated]"
    return {"status": resp.status_code, "body": body, "content_type": resp.headers.get("content-type", "")}


async def complete(
    *,
    prompt: str,
    system: str = "",
    data: str = "",
    output_schema: dict[str, Any] | None = None,
    output_mode: str = "native",
) -> Any:
    """Perform an LLM completion. Returns text, or a validated dict when `output_schema` is provided.

    Args:
        prompt: The user message / instruction to send.
        system: Optional system message to set context for the completion.
        data: Optional data payload (stringified objects, file contents, etc.)
              appended to the prompt so the model can process it.
        output_schema: Optional JSON object schema for structured output.
        output_mode: Structured output mode. `native` tries provider-native JSON
              schema first and falls back to prompted JSON when unsupported;
              `prompted` uses prompt-based structured output directly.
    """
    if _complete_fn is None:
        return "ERROR: completion function not configured"
    return await _complete_fn(prompt, system, data, output_schema, output_mode)


def find_unresolved_async_placeholder(value: Any, *, path: str = "$") -> str | None:
    """Find Monty/Python coroutine placeholders caused by missing `await`."""
    if isinstance(value, str) and (
        value.startswith(_ASYNC_PLACEHOLDER_PREFIXES) or any(marker in value for marker in _ASYNC_PLACEHOLDER_MARKERS)
    ):
        return f"{path}: unresolved async value {value!r}; did you forget to await an async primitive?"
    if isinstance(value, dict):
        for key, item in value.items():
            found = find_unresolved_async_placeholder(item, path=f"{path}.{key}")
            if found:
                return found
    if isinstance(value, list):
        for i, item in enumerate(value):
            found = find_unresolved_async_placeholder(item, path=f"{path}[{i}]")
            if found:
                return found
    return None


def reject_unresolved_async_placeholders(value: Any) -> None:
    if found := find_unresolved_async_placeholder(value):
        raise ValueError(found)


def validate_json_schema_subset(schema: dict[str, Any], value: Any, *, path: str = "$") -> None:
    """Validate common JSON-schema constraints used by pa self-authored tools."""
    reject_unresolved_async_placeholders(value)

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

    typ = schema.get("type")
    if isinstance(typ, list):
        errors = []
        for option in typ:
            try:
                validate_json_schema_subset({**schema, "type": option}, value, path=path)
            except ValueError as e:
                errors.append(str(e))
            else:
                return
        raise ValueError(f"{path}: did not match any allowed type: {'; '.join(errors)}")

    if typ == "null":
        if value is not None:
            raise ValueError(f"{path}: expected null, got {type(value).__name__}")
        return
    if typ == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean, got {type(value).__name__}")
        return
    if typ == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path}: expected integer, got {type(value).__name__}")
        return
    if typ == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}: expected number, got {type(value).__name__}")
        return
    if typ == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string, got {type(value).__name__}")
        return
    if typ == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array, got {type(value).__name__}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                validate_json_schema_subset(item_schema, item, path=f"{path}[{i}]")
        return
    if typ == "object" or "properties" in schema:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object, got {type(value).__name__}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"{path}: schema properties must be an object")
        for required in schema.get("required", []):
            if required not in value:
                raise ValueError(f"{path}: missing required field {required!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path}: unexpected field(s): {', '.join(extra)}")
        for key, item_schema in properties.items():
            if key in value and isinstance(item_schema, dict):
                validate_json_schema_subset(item_schema, value[key], path=f"{path}.{key}")
