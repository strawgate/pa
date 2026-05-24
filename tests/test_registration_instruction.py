import pytest

from pa.manifest import Manifest, Registration
from pa.capability import PaRegistrations
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@pytest.mark.asyncio
async def test_instruction_registration_in_request(tmp_cwd):
    m = Manifest()
    m.add(Registration(slot="instruction", name="cheerio", code='"Always end your responses with Cheerio!"'))
    m.save()

    agent = Agent(TestModel(), capabilities=[PaRegistrations()])
    result = await agent.run("hi")
    # Check that the instruction was injected
    messages = result.all_messages()
    # The first message should contain the instruction text
    first_msg = messages[0]
    # Instructions may be in the system prompt parts or as part of ModelRequest
    found = False
    for part in first_msg.parts:
        if hasattr(part, "content") and "Cheerio" in str(part.content):
            found = True
            break
    if not found:
        # Check if instructions are attached differently
        if hasattr(first_msg, "instructions") and first_msg.instructions:
            found = "Cheerio" in first_msg.instructions
    assert found, f"Expected 'Cheerio' in instructions, got messages: {messages}"
