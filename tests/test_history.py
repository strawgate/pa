from __future__ import annotations

from pydantic_ai.messages import ModelRequest, UserPromptPart

from pa import history


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
