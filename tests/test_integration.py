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

from pa.builtin_instructions import PA_BUILTIN_INSTRUCTIONS
from pa.manifest import Manifest
from pa.registration_tools import SELF_EVOLUTION_TOOL_MAX_RETRIES, check_registrations
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


def enable_advanced_registration_tools(agent_dir: Path) -> None:
    agent_yaml = yaml.safe_load((agent_dir / "agent.yaml").read_text())
    for cap in agent_yaml["capabilities"]:
        if "PaRegistrations" in cap:
            cap["PaRegistrations"] = {"expose_advanced_registration_tools": True}
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump(agent_yaml))


def hide_advanced_registration_tools(agent_dir: Path) -> None:
    agent_yaml = yaml.safe_load((agent_dir / "agent.yaml").read_text())
    for cap in agent_yaml["capabilities"]:
        if "PaRegistrations" in cap:
            cap["PaRegistrations"] = {"expose_advanced_registration_tools": False}
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump(agent_yaml))


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
        assert "register_before_tool_hook" in seen_tools
        assert "register_after_tool_hook" in seen_tools
        assert "register_before_run_hook" in seen_tools
        assert "register_after_run_hook" in seen_tools
        assert "register_compaction" in seen_tools
        assert "register_tool_filter" in seen_tools
        assert "validate_tool" in seen_tools
        assert "list_registrations" in seen_tools
        assert "check_registrations" in seen_tools
        assert "disable_registration" in seen_tools
        assert "remove_registration" in seen_tools
        assert "disable_tool" not in seen_tools
        assert "register_guard" not in seen_tools

    def test_advanced_registration_tools_can_be_hidden(self, agent_dir):
        """Hosts can still hide high-risk policy surfaces explicitly."""
        hide_advanced_registration_tools(agent_dir)
        seen_tools = []

        def capture_tools(messages, info: AgentInfo):
            seen_tools.extend(t.name for t in info.function_tools)
            return ModelResponse(parts=[TextPart(content="hi")])

        build_agent(model=FunctionModel(capture_tools)).run_sync("test")

        assert "register_before_tool_hook" in seen_tools
        assert "register_after_tool_hook" in seen_tools
        assert "register_before_run_hook" in seen_tools
        assert "register_after_run_hook" in seen_tools
        assert "register_compaction" not in seen_tools
        assert "register_tool_filter" not in seen_tools
        assert "disable_tool" not in seen_tools

    def test_run_code_description_lists_all_tools(self, agent_dir):
        """The run_code tool description includes the configured sandboxed primitives."""
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
            "list_dir",
            "bash",
            "http_get",
        ]:
            assert tool_name in description, f"{tool_name} not found in run_code description"

    def test_run_code_can_list_directories(self, agent_dir):
        """CodeMode exposes list_dir as a sandbox primitive."""
        call_count = 0
        observed_return = []

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, observed_return
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="run_code",
                            args={"code": 'entries = await list_dir(path=".")\n[e["name"] for e in entries]'},
                            tool_call_id="tc1",
                        )
                    ]
                )
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "run_code" and hasattr(part, "content"):
                        observed_return = part.content
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("list files")

        assert result.output == "done"
        assert "agent.yaml" in observed_return

    def test_agent_template_instructions_are_user_owned(self, agent_dir):
        """pa's built-in instructions are injected by code, not copied into agent.yaml."""
        agent_yaml = yaml.safe_load((agent_dir / "agent.yaml").read_text())
        assert "Tool organization" not in agent_yaml["instructions"]
        assert "Registration code is Monty" not in agent_yaml["instructions"]
        assert "self-evolving agent working in this project" in agent_yaml["instructions"]

    def test_builtin_and_user_instructions_are_both_present(self, agent_dir):
        """build_agent layers code-owned pa guidance with user-owned template instructions."""
        instruction_text = ""

        def capture_instructions(messages, info: AgentInfo):
            nonlocal instruction_text
            parts = info.model_request_parameters.instruction_parts or []
            instruction_text = "\n".join(part.content for part in parts)
            return ModelResponse(parts=[TextPart(content="done")])

        build_agent(model=FunctionModel(capture_instructions)).run_sync("test")

        assert "self-evolving agent working in this project" in instruction_text
        assert PA_BUILTIN_INSTRUCTIONS.splitlines()[0] in instruction_text

    def test_tool_filter_uses_native_prepare_tools(self, agent_dir):
        """tool_filter registrations filter primitives before CodeMode builds run_code."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "tool_filter",
                "name": "read_only",
                "code": '["read_file"]',
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        description = ""

        def capture_desc(messages, info: AgentInfo):
            nonlocal description
            for t in info.function_tools:
                if t.name == "run_code":
                    description = t.description
            return ModelResponse(parts=[TextPart(content="hi")])

        agent = build_agent(model=FunctionModel(capture_desc))
        agent.run_sync("test")

        assert "read_file" in description
        assert "list_dir" not in description
        assert "bash" not in description

    def test_tool_filter_failures_are_recorded(self, agent_dir):
        """Broken tool_filter registrations fail open but persist their error."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "tool_filter",
                "name": "broken_filter",
                "code": "missing_name",
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))

        def noop(messages, info: AgentInfo):
            return ModelResponse(parts=[TextPart(content="hi")])

        build_agent(model=FunctionModel(noop)).run_sync("test")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        reg = manifest["registrations"][0]
        assert reg["name"] == "broken_filter"
        assert reg["last_error"]
        assert reg["last_run_status"] == "error"

    def test_disabled_tool_filter_is_not_applied(self, agent_dir):
        """Disabled tool_filter registrations are ignored by native prepare_tools."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "tool_filter",
                "name": "disabled_read_only",
                "code": '["read_file"]',
                "status": "disabled",
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        description = ""

        def capture_desc(messages, info: AgentInfo):
            nonlocal description
            for t in info.function_tools:
                if t.name == "run_code":
                    description = t.description
            return ModelResponse(parts=[TextPart(content="hi")])

        build_agent(model=FunctionModel(capture_desc)).run_sync("test")

        assert "read_file" in description
        assert "bash" in description

    def test_guard_failures_are_recorded_without_crashing_agent(self, agent_dir):
        """Broken legacy guards fail open and persist health."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "guard",
                "name": "broken_guard",
                "code": "missing_name",
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="list_registrations", args={}, tool_call_id="tc1")])
            return ModelResponse(parts=[TextPart(content="recovered")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("trigger guard")

        assert result.output == "recovered"
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        reg = manifest["registrations"][0]
        assert reg["last_error"]
        assert reg["last_run_status"] == "error"

    def test_disabled_guard_does_not_execute(self, agent_dir):
        """Disabled guards do not participate in before_tool_execute."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "guard",
                "name": "disabled_guard",
                "code": "missing_name",
                "status": "disabled",
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="list_registrations", args={}, tool_call_id="tc1")])
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("trigger disabled guard")

        assert result.output == "done"
        reg = Manifest.load(agent_dir / "pa" / "registrations.yaml").find("disabled_guard")
        assert reg is not None
        assert reg.last_run_status == "unknown"
        assert reg.last_error == ""


class TestSelfImprovementLoop:
    def test_register_instruction_via_run_code(self, agent_dir):
        """Model calls register_instruction as a native tool; manifest is written."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_instruction",
                            args={"name": "cheerio", "code": '"Always end with Cheerio!"'},
                            tool_call_id="tc1",
                        )
                    ]
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
                    parts=[
                        ToolCallPart(
                            tool_name="register_instruction",
                            args={"name": "cheerio", "code": '"Always end with Cheerio!"'},
                            tool_call_id="tc1",
                        )
                    ]
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
        assert any("Cheerio" in c for c in dynamic_contents), (
            f"Expected 'Cheerio' in dynamic instructions, got: {dynamic_contents}"
        )

    def test_disabled_instruction_is_not_injected(self, agent_dir):
        """Disabled instruction registrations are ignored by get_instructions."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "instruction",
                "name": "disabled_note",
                "code": '"Never include this."',
                "status": "disabled",
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        instruction_parts = []

        def check_model(messages, info: AgentInfo):
            instruction_parts.extend(info.model_request_parameters.instruction_parts or [])
            return ModelResponse(parts=[TextPart(content="done")])

        build_agent(model=FunctionModel(check_model)).run_sync("test")

        dynamic_contents = [p.content for p in instruction_parts if getattr(p, "dynamic", False)]
        assert all("Never include this" not in c for c in dynamic_contents)

    def test_before_run_hook_injects_run_local_guidance(self, agent_dir):
        """before_run_hook return text is exposed as dynamic run-local guidance."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "before_run_hook",
                "name": "start_note",
                "code": '"Use the project checklist before answering."',
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        instruction_parts = []

        def check_model(messages, info: AgentInfo):
            instruction_parts.extend(info.model_request_parameters.instruction_parts or [])
            return ModelResponse(parts=[TextPart(content="done")])

        build_agent(model=FunctionModel(check_model)).run_sync("test")

        dynamic_contents = [p.content for p in instruction_parts if getattr(p, "dynamic", False)]
        assert any("Use the project checklist" in c for c in dynamic_contents)

    def test_after_run_hook_can_replace_output(self, agent_dir):
        """after_run_hook can modify final output through the native after_run hook."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "after_run_hook",
                "name": "signoff",
                "code": '{"action": "replace_output", "output": output + " signed"}',
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))

        def model(messages, info: AgentInfo):
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(model)).run_sync("test")

        assert result.output == "done signed"

    def test_after_tool_hook_can_modify_result(self, agent_dir):
        """after_tool_hook can transform native tool results."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "after_tool_hook",
                "name": "rewrite_registration_listing",
                "code": '{"action": "modify", "result": "hooked result"} if tool_name == "list_registrations" else {"action": "allow"}',
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        observed_return = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, observed_return
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="list_registrations", args={}, tool_call_id="tc1")])
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "list_registrations" and hasattr(part, "content"):
                        observed_return = str(part.content)
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("test")

        assert result.output == "done"
        assert observed_return == "hooked result"

    def test_broken_before_tool_hook_fails_open_and_records_error(self, agent_dir):
        """Broken before-tool hooks should not brick management tools."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "before_tool_hook",
                "name": "broken_before",
                "code": "missing_name",
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        observed_return = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, observed_return
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="list_registrations", args={}, tool_call_id="tc1")])
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "list_registrations" and hasattr(part, "content"):
                        observed_return = str(part.content)
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("test")

        assert result.output == "done"
        assert "broken_before" in observed_return
        assert "missing_name" in observed_return

    def test_broken_after_tool_hook_fails_open_and_records_error(self, agent_dir):
        """Broken after-tool hooks should return the original tool result."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "after_tool_hook",
                "name": "broken_after",
                "code": "missing_name",
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        observed_return = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, observed_return
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="list_registrations", args={}, tool_call_id="tc1")])
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "list_registrations" and hasattr(part, "content"):
                        observed_return = str(part.content)
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("test")

        assert result.output == "done"
        assert "broken_after" in observed_return
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert manifest["registrations"][0]["last_run_status"] == "error"
        assert "missing_name" in manifest["registrations"][0]["last_error"]

    def test_before_tool_hook_can_block_run_code_primitive_usage(self, agent_dir):
        """Tool hooks can govern sandbox primitives by inspecting run_code."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "before_tool_hook",
                "name": "block_writes",
                "code": '{"action": "deny", "reason": "no writes"} if tool_name == "run_code" and "write_file(" in args.get("code", "") else {"action": "allow"}',
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        retry_prompt = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, retry_prompt
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="run_code",
                            args={"code": 'await write_file(path="blocked.txt", content="nope")'},
                            tool_call_id="tc1",
                        )
                    ]
                )
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "run_code" and hasattr(part, "content"):
                        retry_prompt = str(part.content)
            return ModelResponse(parts=[TextPart(content="blocked")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("try a blocked write")

        assert result.output == "blocked"
        assert "no writes" in retry_prompt
        assert not (agent_dir / "blocked.txt").exists()

    def test_after_tool_hook_can_modify_run_code_result(self, agent_dir):
        """Tool hooks can transform the native run_code result."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "after_tool_hook",
                "name": "tag_nonzero",
                "code": '{"action": "modify", "result": {**result, "pa_note": "nonzero"}} if isinstance(result, dict) and result.get("returncode", 0) != 0 else {"action": "allow"}',
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        observed_return = {}

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, observed_return
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="run_code",
                            args={"code": 'await bash(command="false", timeout_s=5)'},
                            tool_call_id="tc1",
                        )
                    ]
                )
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "run_code" and hasattr(part, "content"):
                        observed_return = part.content
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("run failing command")

        assert result.output == "done"
        assert observed_return["returncode"] == 1
        assert observed_return["pa_note"] == "nonzero"

    def test_register_before_tool_hook_via_run_code(self, agent_dir):
        """Model calls register_before_tool_hook as a native tool; manifest persists it."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                guard_code = '{"action": "deny", "reason": "no bash"} if tool_name == "bash" else {"action": "allow"}'
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_before_tool_hook",
                            args={"name": "no_bash", "code": guard_code},
                            tool_call_id="tc1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register before-tool hook")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert len(manifest["registrations"]) == 1
        assert manifest["registrations"][0]["slot"] == "before_tool_hook"

    def test_register_compaction_via_run_code(self, agent_dir):
        """Model calls register_compaction as a native tool."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_compaction",
                            args={"name": "keep_last", "code": "[len(messages) - 1]"},
                            tool_call_id="tc1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register compaction")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert len(manifest["registrations"]) == 1
        assert manifest["registrations"][0]["slot"] == "compaction"
        assert manifest["registrations"][0]["name"] == "keep_last"

    def test_registration_management_tools_serialize_manifest_writes(self, agent_dir):
        """Parallel registration tool calls should not corrupt the manifest."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_instruction",
                            args={"name": "one", "code": '"one"'},
                            tool_call_id="tc1",
                        ),
                        ToolCallPart(
                            tool_name="register_instruction",
                            args={"name": "two", "code": '"two"'},
                            tool_call_id="tc2",
                        ),
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        build_agent(model=FunctionModel(scripted)).run_sync("register two instructions")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert [reg["name"] for reg in manifest["registrations"]] == ["one", "two"]

    def test_register_tool_without_example_is_draft(self, agent_dir):
        """register_tool without example args saves a non-callable draft."""
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_tool",
                            args={
                                "name": "double",
                                "description": "Double an integer.",
                                "code": 'args["x"] * 2',
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {"x": {"type": "integer"}},
                                    "required": ["x"],
                                    "additionalProperties": False,
                                },
                            },
                            tool_call_id="tc1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register draft")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert manifest["registrations"][0]["name"] == "double"
        assert manifest["registrations"][0]["status"] == "draft"

        seen_tools = []

        def capture_tools(messages, info: AgentInfo):
            seen_tools.extend(t.name for t in info.function_tools)
            return ModelResponse(parts=[TextPart(content="hi")])

        build_agent(model=FunctionModel(capture_tools)).run_sync("next")
        assert "double" not in seen_tools

    def test_registration_toolset_allows_multiple_tool_arg_repairs(self, agent_dir):
        """Registration management tools do not use a tiny retry budget."""
        assert SELF_EVOLUTION_TOOL_MAX_RETRIES == 15
        call_count = 0

        def scripted(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count in (1, 2, 3):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_tool",
                            args={
                                "name": "double",
                                "code": 'args["x"] * 2',
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {"x": {"type": "integer"}},
                                    "required": ["x"],
                                    "additionalProperties": False,
                                },
                                "example_args": {"x": 2},
                            },
                            tool_call_id=f"tc{call_count}",
                        )
                    ]
                )
            if call_count == 4:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_tool",
                            args={
                                "name": "double",
                                "description": "Double an integer.",
                                "code": 'args["x"] * 2',
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {"x": {"type": "integer"}},
                                    "required": ["x"],
                                    "additionalProperties": False,
                                },
                                "example_args": {"x": 2},
                            },
                            tool_call_id="tc4",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        build_agent(model=FunctionModel(scripted)).run_sync("register active")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert call_count == 5
        assert manifest["registrations"][0]["name"] == "double"
        assert manifest["registrations"][0]["status"] == "active"

    def test_validated_registered_tool_is_native_with_schema(self, agent_dir):
        """Validated tools are exposed as native tools with their declared schema."""
        call_count = 0

        def register_model(messages, info: AgentInfo):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_tool",
                            args={
                                "name": "double",
                                "description": "Double an integer.",
                                "code": 'args["x"] * 2',
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {"x": {"type": "integer"}},
                                    "required": ["x"],
                                    "additionalProperties": False,
                                },
                                "example_args": {"x": 2},
                            },
                            tool_call_id="tc1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="registered")])

        build_agent(model=FunctionModel(register_model)).run_sync("register active")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        reg = manifest["registrations"][0]
        assert reg["status"] == "active"
        assert reg["validated_example_args"] == {"x": 2}

        call_count = 0
        tool_schema = {}
        observed_return = ""

        def use_model(messages, info: AgentInfo):
            nonlocal call_count, tool_schema, observed_return
            call_count += 1
            if call_count == 1:
                for tool in info.function_tools:
                    if tool.name == "double":
                        tool_schema = tool.parameters_json_schema
                return ModelResponse(parts=[ToolCallPart(tool_name="double", args={"x": 3}, tool_call_id="tc2")])
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "double" and hasattr(part, "content"):
                        observed_return = str(part.content)
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(use_model)).run_sync("use double")
        assert result.output == "done"
        assert tool_schema["properties"]["x"]["type"] == "integer"
        assert observed_return == "6"

    def test_registered_tool_can_use_filesystem_primitives(self, agent_dir):
        """Registered Monty tools can call list_dir and read_file directly."""
        (agent_dir / "notes.txt").write_text("alpha\nneedle\n", encoding="utf-8")
        (agent_dir / ".env").write_text("needle=secret\n", encoding="utf-8")
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "tool",
                "name": "one_level_grep",
                "description": "Search text files in one directory level, skipping .env.",
                "code": (
                    'pattern = args["pattern"]\n'
                    "matches = []\n"
                    'for entry in await list_dir(path=args.get("path", ".")):\n'
                    '    if entry["is_file"] and entry["name"] != ".env":\n'
                    '        text = await read_file(path=entry["path"])\n'
                    "        lines = text.splitlines()\n"
                    "        for line_no, line in enumerate(lines, 1):\n"
                    "            if pattern in line:\n"
                    "                matches.append(f\"{entry['name']}:{line_no}:{line}\")\n"
                    "matches"
                ),
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}},
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                "status": "active",
                "validated_example_args": {"path": ".", "pattern": "needle"},
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        observed_return = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, observed_return
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="one_level_grep",
                            args={"path": ".", "pattern": "needle"},
                            tool_call_id="tc1",
                        )
                    ]
                )
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "one_level_grep" and hasattr(part, "content"):
                        observed_return = str(part.content)
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("search files")

        assert result.output == "done"
        assert "notes.txt:2:needle" in observed_return
        assert ".env" not in observed_return
        health = check_registrations()
        assert '"name": "one_level_grep"' in health
        assert '"check": "ok"' in health

    def test_registered_tools_allow_multiple_arg_repairs(self, agent_dir):
        """Active registered tools share the self-evolution retry budget."""
        assert SELF_EVOLUTION_TOOL_MAX_RETRIES == 15
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "tool",
                "name": "double",
                "description": "Double an integer.",
                "code": 'args["x"] * 2',
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
                "status": "active",
                "validated_example_args": {"x": 2},
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        observed_return = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, observed_return
            call_count += 1
            if call_count in (1, 2, 3):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="double",
                            args={"x": "not-an-integer"},
                            tool_call_id=f"tc{call_count}",
                        )
                    ]
                )
            if call_count == 4:
                return ModelResponse(parts=[ToolCallPart(tool_name="double", args={"x": 4}, tool_call_id="tc4")])
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "tool_name", None) == "double" and hasattr(part, "content"):
                        observed_return = str(part.content)
            return ModelResponse(parts=[TextPart(content="done")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("use double")

        assert result.output == "done"
        assert call_count == 5
        assert observed_return == "8"

    def test_broken_legacy_registered_tool_retries_instead_of_crashing(self, agent_dir):
        """Active legacy tools with bad Monty return retry feedback instead of crashing."""
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        manifest["registrations"].append(
            {
                "slot": "tool",
                "name": "bad_tool",
                "description": "A broken legacy tool.",
                "code": 'import os\nos.listdir(".")',
                "status": "active",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
        )
        (agent_dir / "pa" / "registrations.yaml").write_text(yaml.safe_dump(manifest))
        call_count = 0
        saw_retry = False

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, saw_retry
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="bad_tool", args={}, tool_call_id="tc1")])
            saw_retry = any("bad_tool" in str(part) for msg in messages for part in msg.parts)
            return ModelResponse(parts=[TextPart(content="recovered")])

        result = build_agent(model=FunctionModel(scripted)).run_sync("call bad")
        assert result.output == "recovered"
        assert saw_retry
        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        reg = manifest["registrations"][0]
        assert reg["last_error"]
        assert reg["last_run_status"] == "error"

    def test_list_registrations_via_run_code(self, agent_dir):
        """After registering natively, list_registrations returns the entry."""
        call_count = 0
        list_result = ""

        def scripted(messages, info: AgentInfo):
            nonlocal call_count, list_result
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="register_instruction",
                            args={"name": "cheerio", "code": '"Cheerio!"'},
                            tool_call_id="tc1",
                        )
                    ]
                )
            elif call_count == 2:
                return ModelResponse(parts=[ToolCallPart(tool_name="list_registrations", args={}, tool_call_id="tc2")])
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
                    parts=[
                        ToolCallPart(
                            tool_name="register_instruction",
                            args={"name": "cheerio", "code": '"hi"'},
                            tool_call_id="tc1",
                        )
                    ]
                )
            elif call_count == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="remove_registration",
                            args={"name": "cheerio"},
                            tool_call_id="tc2",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(model=FunctionModel(scripted))
        agent.run_sync("register then remove")

        manifest = yaml.safe_load((agent_dir / "pa" / "registrations.yaml").read_text())
        assert manifest["registrations"] == []


class TestCLIInit:
    def test_pa_init_creates_files(self, tmp_path, monkeypatch):
        """pa init creates agent.yaml, pa/registrations.yaml, and agent-readable docs."""
        monkeypatch.chdir(tmp_path)
        from pa.cli import init

        init()

        assert (tmp_path / "agent.yaml").exists()
        assert (tmp_path / "pa" / "registrations.yaml").exists()
        guide = tmp_path / "docs" / "registrations.md"
        assert guide.exists()
        assert "Registration Guide" in guide.read_text()

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
        (tmp_path / "docs" / "registrations.md").write_text("custom guide\n")
        init()
        # Should not have been overwritten
        assert (tmp_path / "agent.yaml").read_text() == "model: test\n"
        assert (tmp_path / "docs" / "registrations.md").read_text() == "custom guide\n"
