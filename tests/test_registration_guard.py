import pytest

from pa.manifest import Registration
from pa.registrations import make_guard_hook
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_guard_denies_bash():
    """A guard that denies bash raises ModelRetry."""
    reg = Registration(
        slot="guard",
        name="no_bash",
        code='{"action": "deny", "reason": "no bash"} if tool_name == "bash" else {"action": "allow"}',
    )
    hook = make_guard_hook(reg)

    ctx = MagicMock()
    call = ToolCallPart(tool_name="bash", args={"command": "rm -rf /"}, tool_call_id="test-id")
    tool_def = ToolDefinition(name="bash", description="run bash", parameters_json_schema={})

    with pytest.raises(ModelRetry, match="denied.*bash.*no bash"):
        await hook(ctx, call=call, tool_def=tool_def, args={"command": "rm -rf /"})


@pytest.mark.asyncio
async def test_guard_allows_read_file():
    """A guard that only denies bash allows read_file."""
    reg = Registration(
        slot="guard",
        name="no_bash",
        code='{"action": "deny", "reason": "no bash"} if tool_name == "bash" else {"action": "allow"}',
    )
    hook = make_guard_hook(reg)

    ctx = MagicMock()
    call = ToolCallPart(tool_name="read_file", args={"path": "/tmp/x"}, tool_call_id="test-id")
    tool_def = ToolDefinition(name="read_file", description="read a file", parameters_json_schema={})

    result = await hook(ctx, call=call, tool_def=tool_def, args={"path": "/tmp/x"})
    assert result == {"path": "/tmp/x"}


@pytest.mark.asyncio
async def test_guard_modifies_args():
    """A guard that modifies args returns the modified dict."""
    reg = Registration(
        slot="guard",
        name="force_timeout",
        code='{"action": "modify", "args": {**args, "timeout_s": 5.0}}',
    )
    hook = make_guard_hook(reg)

    ctx = MagicMock()
    call = ToolCallPart(tool_name="bash", args={"command": "ls"}, tool_call_id="test-id")
    tool_def = ToolDefinition(name="bash", description="run bash", parameters_json_schema={})

    result = await hook(ctx, call=call, tool_def=tool_def, args={"command": "ls"})
    assert result == {"command": "ls", "timeout_s": 5.0}
