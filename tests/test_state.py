from __future__ import annotations

from pa.state import ensure_default_agent, ensure_project_agent, ensure_state, resolve_state


def test_project_agent_is_forked_from_home_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / ".pa"
    template = tmp_path / "template.yaml"
    template.write_text("name: base-agent\ninstructions: base\ncapabilities: []\n")

    default_path, created_default = ensure_default_agent(template_path=template, home=home)
    assert created_default is True
    default_path.write_text("name: custom-agent\ninstructions: custom\ncapabilities: []\n")

    agent_path, created_agent, same_default_path, created_default_again = ensure_project_agent(
        template_path=template,
        home=home,
    )

    assert created_agent is True
    assert created_default_again is False
    assert same_default_path == default_path
    assert agent_path == tmp_path / "agent.yaml"
    assert agent_path.read_text() == default_path.read_text()


def test_resolved_state_uses_agent_and_project_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / ".pa"
    monkeypatch.setenv("PA_HOME", str(home))
    (tmp_path / "agent.yaml").write_text("name: project Agent!\ncapabilities: []\n")

    state = resolve_state(tmp_path / "agent.yaml")

    assert state.home == home
    assert state.agent_name == "project Agent!"
    assert state.working_dir == tmp_path
    assert state.project_root == tmp_path
    assert state.project_key.startswith("project-agent--")
    assert state.registrations_path == state.state_dir / "registrations.yaml"
    assert state.history_path == state.state_dir / "history.json"


def test_ensure_state_migrates_legacy_files_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PA_HOME", str(tmp_path / ".pa"))
    (tmp_path / "agent.yaml").write_text("name: migrator\ncapabilities: []\n")
    legacy_dir = tmp_path / "pa"
    legacy_dir.mkdir()
    (legacy_dir / "registrations.yaml").write_text("registrations:\n- name: old\n")
    (legacy_dir / "history.json").write_text("[]")

    state = resolve_state(tmp_path / "agent.yaml")
    notes = ensure_state(state)

    assert state.registrations_path.read_text() == "registrations:\n- name: old\n"
    assert not state.history_path.exists()
    assert state.legacy_history_archive_path.read_text() == "[]"
    assert any("migrated" in note for note in notes)
    assert any("archived legacy history" in note for note in notes)

    state.registrations_path.write_text("registrations: []\n")
    ensure_state(state)

    assert state.registrations_path.read_text() == "registrations: []\n"
