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


def register_tool(name: str, description: str, code: str) -> str:
    """Register a reusable Monty tool. The code receives `args: dict` with
    whatever arguments callers pass. Returns any value as the tool result.
    The tool becomes callable by name on the next agent run."""
    try:
        reg = Registration(slot="tool", name=name, code=code, description=description)
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
    return f"OK: registered tool/{name}. It will be callable on next agent run."


def list_registrations() -> str:
    """Return a JSON-encoded list of {slot, name, lines, preview} entries."""
    m = _load()
    out = []
    for r in m.registrations:
        out.append(
            {
                "slot": r.slot,
                "name": r.name,
                "lines": r.code.count("\n") + 1,
                "preview": r.code[:120] + ("..." if len(r.code) > 120 else ""),
            }
        )
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
