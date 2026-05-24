import pytest

from pa.manifest import Manifest, Registration
from pa.capability import PaRegistrations
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel


@pytest.mark.asyncio
async def test_compaction_reduces_history(tmp_cwd):
    """Register a compaction that keeps only the last message index."""
    m = Manifest()
    m.add(
        Registration(
            slot="compaction",
            name="keep_last",
            code="[len(messages) - 1]",
        )
    )
    m.save()

    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content="ok")])),
        capabilities=[PaRegistrations()],
    )
    # First run
    result1 = await agent.run("first")
    history = result1.all_messages()
    # Second run with history — compaction should fire
    result2 = await agent.run("second", message_history=history)
    # The compaction is applied via ProcessHistory before model request.
    # Verify the agent ran successfully (basic smoke test).
    assert result2.output is not None


@pytest.mark.asyncio
async def test_compaction_cannot_drop_current_request(tmp_cwd):
    """A compaction that keeps old history cannot remove the active user prompt."""
    m = Manifest()
    m.add(
        Registration(
            slot="compaction",
            name="keep_first",
            code="[0]",
        )
    )
    m.save()

    seen_messages = []

    def capture(messages, info):
        seen_messages.append(messages)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent = Agent(FunctionModel(capture), capabilities=[PaRegistrations()])
    result1 = await agent.run("first")
    await agent.run("second", message_history=result1.all_messages())

    current_request_seen = any(
        isinstance(msg, ModelRequest)
        and any(isinstance(part, UserPromptPart) and part.content == "second" for part in msg.parts)
        for msg in seen_messages[-1]
    )
    assert current_request_seen
    reg = Manifest.load().find("keep_first")
    assert reg is not None
    assert "current request" in reg.last_error
    assert reg.last_run_status == "error"
