from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pa.conversation import run_coro_sync, run_with_incremental_history, summarize_progress
from pa.cli import _render_tool_event
from pa import history
from pa.progress import (
    HistorySavedEvent,
    ProgressEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    event_to_dict,
    event_to_json,
    summarize_args,
)
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


def test_summarize_progress_describes_tool_calls_returns_and_retries():
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="run_code",
                    args={"code": 'result = await bash(command="pwd", timeout_s=5)\nresult'},
                    tool_call_id="tc1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="run_code",
                    content={"stdout": "/tmp/project\n", "stderr": "", "returncode": 0},
                    tool_call_id="tc1",
                ),
                RetryPromptPart(tool_name="run_code", content="change approach", tool_call_id="tc2"),
            ]
        ),
    ]

    lines = [line for message in messages for line in summarize_progress(message)]

    assert lines == [
        '-> run_code code=result = await bash(command="pwd", timeout_s=5) result',
        "<- run_code success: returncode=0 stdout=/tmp/project",
        "retry run_code: change approach",
    ]


def test_run_with_incremental_history_emits_progress(tmp_cwd):
    template = Path(__file__).parent.parent / "pa" / "agent_template.yaml"
    shutil.copyfile(template, tmp_cwd / "agent.yaml")
    state = resolve_state(tmp_cwd / "agent.yaml")
    ensure_state(state)
    events: list[ProgressEvent] = []
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
        return ModelResponse(parts=[TextPart(content="done")])

    agent = build_agent(tmp_cwd / "agent.yaml", model=FunctionModel(scripted))

    run_coro_sync(
        lambda: run_with_incremental_history(
            agent,
            "calculate",
            [],
            tmp_cwd / "history.json",
            progress=events.append,
        )
    )

    assert any(isinstance(event, RunStartedEvent) for event in events)
    assert any(isinstance(event, HistorySavedEvent) for event in events)
    assert any(isinstance(event, RunCompletedEvent) for event in events)
    assert any(
        isinstance(event, ToolCallStartedEvent) and event.message == "-> run_code code=1 + 1" for event in events
    )
    assert any(
        isinstance(event, ToolCallFinishedEvent) and event.message.startswith("<- run_code success:")
        for event in events
    )


def test_tool_result_progress_emits_before_next_model_request(tmp_cwd):
    template = Path(__file__).parent.parent / "pa" / "agent_template.yaml"
    shutil.copyfile(template, tmp_cwd / "agent.yaml")
    state = resolve_state(tmp_cwd / "agent.yaml")
    ensure_state(state)
    timeline: list[object] = []
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
        timeline.append("second_model_request")
        return ModelResponse(parts=[TextPart(content="done")])

    agent = build_agent(tmp_cwd / "agent.yaml", model=FunctionModel(scripted))

    run_coro_sync(
        lambda: run_with_incremental_history(
            agent,
            "calculate",
            [],
            tmp_cwd / "history.json",
            progress=timeline.append,
        )
    )

    result_index = next(i for i, item in enumerate(timeline) if isinstance(item, ToolCallFinishedEvent))
    second_model_index = timeline.index("second_model_request")
    assert result_index < second_model_index


def test_progress_events_are_jsonl_serializable():
    event = ToolCallStartedEvent(
        message="-> bash command=pwd",
        tool_name="bash",
        tool_call_id="tc1",
        args_summary="command=pwd",
    )

    payload = event_to_dict(event)

    assert payload == {
        "type": "tool_call_started",
        "message": "-> bash command=pwd",
        "tool_name": "bash",
        "tool_call_id": "tc1",
        "args_summary": "command=pwd",
    }
    assert event_to_json(event).startswith('{"args_summary": "command=pwd",')


def test_progress_arg_summary_includes_multiple_native_tool_args():
    assert summarize_args({"path": ".", "query": "needle"}) == "path=. query=needle"


def test_progress_arg_summary_keeps_registration_name_with_code():
    assert summarize_args({"name": "literal_search", "code": "args"}) == "name=literal_search code=args"


def test_default_cli_renderer_hides_large_success_results():
    event = ToolCallFinishedEvent(
        message="<- run_code success: # pa Self-evolving Pydantic-AI agent harness...",
        tool_name="run_code",
        tool_call_id="tc1",
        outcome="success",
        result_summary="# pa Self-evolving Pydantic-AI agent harness...",
    )

    assert _render_tool_event(event, verbose=False) == "<- run_code success"
    assert _render_tool_event(event, verbose=True).startswith("<- run_code success: # pa")


def test_default_cli_renderer_shows_native_tool_success_summary():
    event = ToolCallFinishedEvent(
        message="<- review_changes success: files=6 findings=1 parse_error=invalid JSON",
        tool_name="review_changes",
        tool_call_id="tc1",
        outcome="success",
        result_summary="files=6 findings=1 parse_error=invalid JSON",
    )

    assert _render_tool_event(event, verbose=False) == (
        "<- review_changes success: files=6 findings=1 parse_error=invalid JSON"
    )


def test_progress_value_summary_has_hard_cap():
    event = ToolCallFinishedEvent(
        message="",
        tool_name="bash",
        tool_call_id="tc1",
        outcome="success",
        result_summary="",
    )
    payload = {
        "returncode": 1,
        "stdout": "o" * 240,
        "stderr": "e" * 240,
    }

    from pa.progress import summarize_value

    summary = summarize_value(payload)

    assert len(summary) <= 240
    assert summary.endswith("...")
    assert _render_tool_event(event, verbose=False) == "<- bash success"


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
