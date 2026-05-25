from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import re
import shutil
import subprocess

import yaml

PACKAGE_AGENT_TEMPLATE = Path(__file__).parent / "agent_template.yaml"
PA_HOME_ENV = "PA_HOME"
DEFAULT_AGENT_NAME = "pa-agent"
LEGACY_REGISTRATIONS = Path("pa") / "registrations.yaml"
LEGACY_HISTORY = Path("pa") / "history.json"


@dataclass(frozen=True)
class PaState:
    home: Path
    default_agent_path: Path
    agent_spec_path: Path
    agent_name: str
    working_dir: Path
    project_root: Path
    project_key: str
    state_dir: Path
    registrations_path: Path
    history_path: Path
    sessions_dir: Path

    @property
    def legacy_registrations_path(self) -> Path:
        return self.agent_spec_path.parent / LEGACY_REGISTRATIONS

    @property
    def legacy_history_path(self) -> Path:
        return self.agent_spec_path.parent / LEGACY_HISTORY


def pa_home() -> Path:
    raw = os.environ.get(PA_HOME_ENV)
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".pa").resolve()


def ensure_default_agent(
    *, template_path: Path = PACKAGE_AGENT_TEMPLATE, home: Path | None = None
) -> tuple[Path, bool]:
    """Create the user's default agent profile if it does not exist."""
    default_path = (home or pa_home()) / "agent.yaml"
    if default_path.exists():
        return default_path, False
    default_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, default_path)
    return default_path, True


def ensure_project_agent(
    *,
    target_path: Path = Path("agent.yaml"),
    template_path: Path = PACKAGE_AGENT_TEMPLATE,
    home: Path | None = None,
) -> tuple[Path, bool, Path, bool]:
    """Ensure both the home default and this working directory's fork exist."""
    default_path, created_default = ensure_default_agent(template_path=template_path, home=home)
    target = target_path.expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    if target.exists():
        return target, False, default_path, created_default
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(default_path, target)
    return target, True, default_path, created_default


def resolve_state(agent_spec_path: Path | str = Path("agent.yaml")) -> PaState:
    spec_path = Path(agent_spec_path).expanduser()
    if not spec_path.is_absolute():
        spec_path = Path.cwd() / spec_path
    spec_path = spec_path.resolve()

    home = pa_home()
    agent_name = _agent_name(spec_path)
    working_dir = spec_path.parent
    project_root = _project_root(working_dir)
    project_hash = _project_hash(project_root, working_dir)
    project_key = f"{_slug(agent_name)}--{_slug(working_dir.name)}--{project_hash}"
    state_dir = home / "agents" / project_key
    return PaState(
        home=home,
        default_agent_path=home / "agent.yaml",
        agent_spec_path=spec_path,
        agent_name=agent_name,
        working_dir=working_dir,
        project_root=project_root,
        project_key=project_key,
        state_dir=state_dir,
        registrations_path=state_dir / "registrations.yaml",
        history_path=state_dir / "history.json",
        sessions_dir=state_dir / "sessions",
    )


def ensure_state(state: PaState) -> list[str]:
    """Create state directories and migrate old project-local state once."""
    notes: list[str] = []
    state.state_dir.mkdir(parents=True, exist_ok=True)
    state.sessions_dir.mkdir(parents=True, exist_ok=True)

    if not state.registrations_path.exists():
        if state.legacy_registrations_path.exists():
            shutil.copyfile(state.legacy_registrations_path, state.registrations_path)
            notes.append(f"migrated {state.legacy_registrations_path} -> {state.registrations_path}")
        else:
            state.registrations_path.write_text("registrations: []\n")
            notes.append(f"wrote {state.registrations_path}")

    if not state.history_path.exists() and state.legacy_history_path.exists():
        shutil.copyfile(state.legacy_history_path, state.history_path)
        notes.append(f"migrated {state.legacy_history_path} -> {state.history_path}")

    return notes


def _agent_name(spec_path: Path) -> str:
    if not spec_path.exists():
        return DEFAULT_AGENT_NAME
    try:
        raw = yaml.safe_load(spec_path.read_text()) or {}
    except Exception:
        return DEFAULT_AGENT_NAME
    if not isinstance(raw, dict):
        return DEFAULT_AGENT_NAME
    name = raw.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return DEFAULT_AGENT_NAME


def _project_root(cwd: Path) -> Path:
    root = _git_output(cwd, "rev-parse", "--show-toplevel")
    if root:
        return Path(root).expanduser().resolve()
    return cwd.resolve()


def _project_hash(project_root: Path, working_dir: Path) -> str:
    parts = [f"working-dir={working_dir}", f"root={project_root}"]
    remote = _git_output(project_root, "config", "--get", "remote.origin.url")
    if remote:
        parts.append(f"remote={remote}")
    common_dir = _git_output(project_root, "rev-parse", "--git-common-dir")
    if common_dir:
        common_path = Path(common_dir)
        if not common_path.is_absolute():
            common_path = project_root / common_path
        parts.append(f"git-common-dir={common_path.resolve()}")
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return digest[:12]


def _git_output(cwd: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "agent"
