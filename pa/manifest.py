from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from pa.slots import SLOTS, SlotName, Cardinality

MANIFEST_PATH_DEFAULT = Path("pa") / "registrations.yaml"

RegistrationStatus = Literal["draft", "active", "disabled"]
RegistrationRunStatus = Literal["unknown", "ok", "error"]


def default_tool_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": True}


class ManifestError(Exception): ...


class CardinalityError(ManifestError): ...


class Registration(BaseModel):
    slot: SlotName
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    code: str = Field(min_length=1, max_length=8000)
    description: str = Field(default="", max_length=256)
    parameters_json_schema: dict[str, Any] = Field(default_factory=default_tool_schema)
    status: RegistrationStatus = "active"
    validated_example_args: dict[str, Any] | None = None
    last_error: str = ""
    last_run_status: RegistrationRunStatus = "unknown"
    last_run_at: str = ""
    last_ok_at: str = ""
    last_duration_ms: float | None = None


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
                f"a registration named {reg.name!r} already exists; call remove_registration({reg.name!r}) first."
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
