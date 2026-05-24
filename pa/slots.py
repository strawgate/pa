from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

SlotName = Literal["instruction", "compaction", "guard", "tool_filter", "tool"]


class Cardinality(str, Enum):
    ONE = "one"
    MANY = "many"


@dataclass(frozen=True)
class SlotDef:
    name: SlotName
    cardinality: Cardinality
    return_shape: str
    description: str
    inputs: tuple[str, ...]


SLOTS: dict[SlotName, SlotDef] = {
    "instruction": SlotDef(
        name="instruction",
        cardinality=Cardinality.MANY,
        return_shape="str",
        inputs=("ctx_summary",),
        description="Returns a string appended to dynamic instructions before each model request.",
    ),
    "compaction": SlotDef(
        name="compaction",
        cardinality=Cardinality.ONE,
        return_shape="list[int]",
        inputs=("messages",),
        description="Receives messages: list[dict] (jsonable ModelMessage). Returns list of indices to keep.",
    ),
    "guard": SlotDef(
        name="guard",
        cardinality=Cardinality.MANY,
        return_shape="dict[str, Any]",
        inputs=("tool_name", "args"),
        description=(
            "Receives the about-to-execute tool call. Returns "
            "{'action': 'allow'} | {'action': 'deny', 'reason': str} | "
            "{'action': 'modify', 'args': dict}. First deny wins."
        ),
    ),
    "tool_filter": SlotDef(
        name="tool_filter",
        cardinality=Cardinality.MANY,
        return_shape="list[str]",
        inputs=("tool_names",),
        description="Receives tool_names: list[str]. Returns the filtered list of names to keep. Pipelines.",
    ),
    "tool": SlotDef(
        name="tool",
        cardinality=Cardinality.MANY,
        return_shape="Any",
        inputs=("args",),
        description=(
            "A user-defined tool. Receives `args: dict` with the call arguments. "
            "Returns any JSON-serializable value as the tool result."
        ),
    ),
}
