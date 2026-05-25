from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pa.conversation import run_coro_sync, run_with_incremental_history
from pa import history
from pa.runtime import build_agent
from pa.state import ensure_state, resolve_state


def test_history_does_not_replay_persisted_instructions(tmp_path):
    path = tmp_path / "history.json"
    messages = [
        ModelRequest(
            parts=[UserPromptPart(content="hello")],
            instructions="old cwd and stale tool instructions",
        )
    ]

    history.save(messages, path)
    loaded = history.load(path)

    assert len(loaded) == 1
    assert loaded[0].instructions is None


def test_append_user_prompt_preserves_prompt_when_run_never_starts(tmp_path):
    path = tmp_path / "history.json"

    history.save(history.append_user_prompt([], "please remember this"), path)

    loaded = history.load(path)
    assert len(loaded) == 1
    assert loaded[0].parts[0].content == "please remember this"


def test_default_history_path_uses_pa_home(tmp_path):
    expected_path = tmp_path / ".pa-home" / "history.json"

    history.clear()
    history.save(history.append_user_prompt([], "stored in pa home"))

    assert expected_path.exists()
    loaded = history.load()
    assert loaded[0].parts[0].content == "stored in pa home"

    history.clear()


def test_run_with_incremental_history_saves_partial_tool_progress(tmp_cwd):
    template = Path(__file__).parent.parent / "pa" / "agent_template.yaml"
    shutil.copyfile(template, tmp_cwd / "agent.yaml")
    state = resolve_state(tmp_cwd / "agent.yaml")
    ensure_state(state)
    history_path = tmp_cwd / "history.json"
    call_count = 0

    def scripted(messages, info: AgentInfo):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "1 + 1"},
                        tool_call_id="tc1",
                    )
                ]
            )
        raise RuntimeError("model stopped mid-run")

    agent = build_agent(tmp_cwd / "agent.yaml", model=FunctionModel(scripted))

    with pytest.raises(RuntimeError, match="model stopped mid-run"):
        run_coro_sync(lambda: run_with_incremental_history(agent, "calculate", [], history_path))

    saved = history.load(history_path)
    parts = [part for message in saved for part in message.parts]
    assert any(isinstance(part, UserPromptPart) and part.content == "calculate" for part in parts)
    assert any(getattr(part, "tool_name", None) == "run_code" for part in parts)
    assert any(getattr(part, "tool_call_id", None) == "tc1" and hasattr(part, "content") for part in parts)


def test_run_with_incremental_history_normalizes_in_memory_replay(tmp_cwd):
    template = Path(__file__).parent.parent / "pa" / "agent_template.yaml"
    shutil.copyfile(template, tmp_cwd / "agent.yaml")
    state = resolve_state(tmp_cwd / "agent.yaml")
    ensure_state(state)
    history_path = tmp_cwd / "history.json"
    seen_instructions = []
    prior = [
        ModelRequest(
            parts=[UserPromptPart(content="old prompt")],
            instructions="stale dynamic instructions",
        )
    ]

    def scripted(messages, info: AgentInfo):
        nonlocal seen_instructions
        seen_instructions = [message.instructions for message in messages if isinstance(message, ModelRequest)]
        return ModelResponse(parts=[TextPart(content="done")])

    agent = build_agent(tmp_cwd / "agent.yaml", model=FunctionModel(scripted))

    run_coro_sync(lambda: run_with_incremental_history(agent, "new prompt", prior, history_path))

    assert "stale dynamic instructions" not in seen_instructions
