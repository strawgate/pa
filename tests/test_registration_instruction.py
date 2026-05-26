import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pa.capability import PaRegistrations
from pa.manifest import Manifest, Registration


@pytest.mark.asyncio
async def test_instruction_registration_in_request(tmp_cwd):
    m = Manifest()
    m.add(Registration(slot="instruction", name="cheerio", code='"Always end your responses with Cheerio!"'))
    m.save()

    instruction_text = ""

    def capture_instructions(messages, info: AgentInfo):
        nonlocal instruction_text
        parts = info.model_request_parameters.instruction_parts or []
        instruction_text = "\n".join(part.content for part in parts)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent = Agent(FunctionModel(capture_instructions), capabilities=[PaRegistrations()])
    result = await agent.run("hi")

    assert result.output == "ok"
    assert "Always end your responses with Cheerio!" in instruction_text
