from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

_DEFAULT_TIMEOUT_S = 30.0

# Injected at build time by runtime.py — holds a callable that performs a completion.
_complete_fn: Any = None


async def read_file(*, path: str) -> str:
    """Read a UTF-8 text file from the current local working tree. Returns its contents."""
    return Path(path).read_text(encoding="utf-8")


async def write_file(*, path: str, content: str) -> str:
    """Write `content` to a local working-tree file (UTF-8, overwriting). Returns a status string."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content.encode('utf-8'))} bytes to {path}"


async def list_dir(*, path: str = ".") -> list[dict[str, object]]:
    """List one local directory level. Returns entry metadata sorted by name."""
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


async def complete(*, prompt: str, system: str = "", data: str = "") -> str:
    """Perform an LLM completion. Returns the model's text response.

    Args:
        prompt: The user message / instruction to send.
        system: Optional system message to set context for the completion.
        data: Optional data payload (stringified objects, file contents, etc.)
              appended to the prompt so the model can process it.
    """
    if _complete_fn is None:
        return "ERROR: completion function not configured"
    return await _complete_fn(prompt, system, data)
