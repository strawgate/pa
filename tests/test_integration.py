"""Integration tests for the full pa agent loop.

These tests exercise the complete pipeline:
- build_agent() with model override
- CodeMode wrapping primitives into run_code
- Registration tools called as native tools
- Registrations persisted and loaded on next agent build
"""

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

from pa.runtime import build_agent


@pytest.fixture
def agent_dir(tmp_path, monkeypatch):
    """Set up a fresh agent directory with agent.yaml and pa/registrations.yaml."""
    monkeypatch.chdir(tmp_path)
    template = Path(__file__).parent.parent / "pa" / "agent_template.yaml"
    shutil.copyfile(template, tmp_path / "agent.yaml")
    (tmp_path / "pa").mkdir()
    (tmp_path / "pa" / "registrations.yaml").write_text("registrations: []\n")
    return tmp_path


class TestBuildAgent:
    def test_constructs_with_function_model(self, agent_dir):
        """build_agent() with model override constructs without API keys."""

        def noop_model(messages, info: AgentInfo):
            return ModelResponse(parts=[TextPart(content="hi")])

        agent = build_agent(model=FunctionModel(noop_model))
        assert agent.name == "pa-agent"

    def test_only_run_code_tool_exposed(self, agent_dir):
        """CodeMode sandboxes primitives only; registration/registered tools are native."""
        seen_tools = []

        def capture_tools(messages, info: AgentInfo):
            seen_tools.extend(t.name for t in info.function_tools)
            return ModelResponse(parts=[TextPart(content="hi")])

        agent = build_agent(model=FunctionModel(capture_tools))
        agent.run_sync("test")
        assert "run_code" in seen_tools
        assert "register_tool" in seen_tools
        assert "register_instruction" in seen_tools
        assert "list_registrations" in seen_tools
        assert "remove_registration" in seen_tools

    def test_run_code_description_lists_all_tools(self, agent_dir):
        """The run_code tool description includes the 5 sandboxed primitives."""
        description = ""

        def capture_desc(messages, info: AgentInfo):
            nonlocal description
            for t in info.function_tools:
                if t.name == "run_code":
                    description = t.description
            return ModelResponse(parts=[TextPart(content="hi")])

        agent = build_agent(model=FunctionModel(capture_desc))
        agent.run_sync("test")

        for tool_name in [
            "read_file",
            "write_file",
            "bash",
            "http_get",
        ]:
            assert tool_name in description, f"{tool_name} not found in run_code description"


class TestSelfImprovementLoop:
    def test_register_instruction_via_run_code(self, agent_dir):
        """Model calls register_instruction as a native tool; manifest is written."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="register_instruction",
                        args={"name": "cheerio", "code": '"Always end with Cheerio!"'},
                        tool_call_id="tc1",
                    )]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        result = agent.run_sync("register cheerio")
        assert result.output == "done"

        # Verify manifest
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert len(manifest["registrations"]) == 1
        assert manifest["registrations"][0]["name"] == "cheerio"
        assert manifest["registrations"][0]["slot"] == "instruction"

    def test_registered_instruction_active_on_next_build(self, agent_dir):
        """After registering an instruction, next agent build includes it in model requests."""
        # Step 1: Register
        call_count = 0

        def register_model(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="register_instruction",
                        args={"name": "cheerio", "code": '"Always end with Cheerio!"'},
                        tool_call_id="tc1",
                    )]
                )
            return ModelResponse(parts=[TextPart(content="registered")])

        agent1 = build_agent(model=FunctionModel(register_model))
        agent1.run_sync("register")

        # Step 2: Build new agent, check dynamic instruction appears
        instruction_parts = []

        def check_model(messages, info: AgentInfo):
            instruction_parts.extend(info.model_request_parameters.instruction_parts or [])
            return ModelResponse(parts=[TextPart(content="4. Cheerio!")])

        agent2 = build_agent(model=FunctionModel(check_model))
        agent2.run_sync("What is 2+2?")

        dynamic_contents = [p.content for p in instruction_parts if getattr(p, "dynamic", False)]
        assert any("Cheerio" in c for c in dynamic_contents), f"Expected 'Cheerio' in dynamic instructions, got: {dynamic_contents}"

    def test_register_guard_via_run_code(self, agent_dir):
        """Model calls register_guard as a native tool; manifest persists it."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                guard_code = '{"action": "deny", "reason": "no bash"} if tool_name == "bash" else {"action": "allow"}'
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="register_guard",
                        args={"name": "no_bash", "code": guard_code},
                        tool_call_id="tc1",
                    )]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register guard")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert len(manifest["registrations"]) == 1
        assert manifest["registrations"][0]["slot"] == "guard"

    def test_register_compaction_via_run_code(self, agent_dir):
        """Model calls register_compaction as a native tool."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="register_compaction",
                        args={"name": "keep_last", "code": "[len(messages) - 1]"},
                        tool_call_id="tc1",
                    )]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register compaction")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert len(manifest["registrations"]) == 1
        assert manifest["registrations"][0]["slot"] == "compaction"
        assert manifest["registrations"][0]["name"] == "keep_last"

    def test_list_registrations_via_run_code(self, agent_dir):
        """After registering natively, list_registrations returns the entry."""
        call_count = 0
        list_result = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, list_result
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="register_instruction",
                        args={"name": "cheerio", "code": '"Cheerio!"'},
                        tool_call_id="tc1",
                    )]
                )
            elif call_count == 2:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_registrations", args={}, tool_call_id="tc2")]
                )
            else:
                # Capture the tool return from list_registrations
                for msg in messages:
                    for p in msg.parts:
                        if hasattr(p, "content") and "cheerio" in str(p.content):
                            list_result = str(p.content)
                return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register and list")
        assert "cheerio" in list_result
        assert "instruction" in list_result

    def test_remove_registration_via_run_code(self, agent_dir):
        """Register natively then remove natively; manifest ends up empty."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="register_instruction",
                        args={"name": "cheerio", "code": '"hi"'},
                        tool_call_id="tc1",
                    )]
                )
            elif call_count == 2:
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="remove_registration",
                        args={"name": "cheerio"},
                        tool_call_id="tc2",
                    )]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register then remove")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert manifest["registrations"] == []


class TestCLIInit:
    def test_pa_init_creates_files(self, tmp_path, monkeypatch):
        """pa init creates agent.yaml and pa/registrations.yaml."""
        monkeypatch.chdir(tmp_path)
        from pa.cli import init

        init()

        assert (tmp_path / "agent.yaml").exists()
        assert (tmp_path / "pa" / "registrations.yaml").exists()

        # Validate agent.yaml is parseable
        agent_yaml = yaml.safe_load((tmp_path / "agent.yaml").read_text())
        assert agent_yaml["name"] == "pa-agent"
        assert "CodeMode" in str(agent_yaml["capabilities"])
        assert "PaRegistrations" in str(agent_yaml["capabilities"])

    def test_pa_init_idempotent(self, tmp_path, monkeypatch):
        """Running pa init twice doesn't overwrite existing files."""
        monkeypatch.chdir(tmp_path)
        from pa.cli import init

        init()
        # Modify agent.yaml
        (tmp_path / "agent.yaml").write_text("model: test\n")
        init()
        # Should not have been overwritten
        assert (tmp_path / "agent.yaml").read_text() == "model: test\n"
