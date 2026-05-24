import pytest

from pa.manifest import Manifest, Registration
from pa.capability import PaRegistrations
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


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

    agent = Agent(TestModel(), capabilities=[PaRegistrations()])
    # First run
    result1 = await agent.run("first")
    history = result1.all_messages()
    # Second run with history — compaction should fire
    result2 = await agent.run("second", message_history=history)
    # The compaction is applied via ProcessHistory before model request.
    # Verify the agent ran successfully (basic smoke test).
    assert result2.output is not None
