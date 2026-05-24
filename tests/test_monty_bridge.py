import pytest

from pa.monty_bridge import (
    execute_registration,
    MontySyntaxBridgeError,
    MontyRuntimeBridgeError,
    MontyReturnShapeError,
)
import pydantic_monty as pm


@pytest.mark.asyncio
async def test_happy_path_instruction():
    res = await execute_registration(
        slot="instruction",
        name="test_happy",
        code='"hi from monty"',
        inputs={"ctx_summary": {}},
    )
    assert res.value == "hi from monty"
    assert res.duration_ms >= 0


@pytest.mark.asyncio
async def test_syntax_error():
    with pytest.raises(MontySyntaxBridgeError, match="syntax error"):
        await execute_registration(
            slot="instruction",
            name="test_syntax",
            code="def foo)",
            inputs={"ctx_summary": {}},
        )


@pytest.mark.asyncio
async def test_runtime_error():
    with pytest.raises(MontyRuntimeBridgeError, match="runtime error"):
        await execute_registration(
            slot="instruction",
            name="test_runtime",
            code="1/0",
            inputs={"ctx_summary": {}},
        )


@pytest.mark.asyncio
async def test_wrong_return_shape():
    with pytest.raises(MontyReturnShapeError, match="instruction must return str"):
        await execute_registration(
            slot="instruction",
            name="test_shape",
            code="42",
            inputs={"ctx_summary": {}},
        )


@pytest.mark.asyncio
async def test_sandbox_isolation():
    with pytest.raises(MontySyntaxBridgeError):
        await execute_registration(
            slot="instruction",
            name="test_sandbox",
            code='import os\nos.listdir(".")',
            inputs={"ctx_summary": {}},
        )


@pytest.mark.asyncio
async def test_guard_return_shape():
    res = await execute_registration(
        slot="guard",
        name="test_guard",
        code='{"action": "allow"}',
        inputs={"tool_name": "bash", "args": {"command": "ls"}},
    )
    assert res.value == {"action": "allow"}


@pytest.mark.asyncio
async def test_timeout():
    with pytest.raises(MontyRuntimeBridgeError, match="runtime error"):
        await execute_registration(
            slot="instruction",
            name="test_timeout",
            code="x = 0\nwhile True:\n    x = x + 1\n'done'",
            inputs={"ctx_summary": {}},
            limits=pm.ResourceLimits(max_duration_secs=0.05),
        )
