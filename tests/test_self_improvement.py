import json

from pa import primitives
from pa.manifest import Manifest, Registration
from pa.registration_tools import (
    check_registrations,
    disable_registration,
    list_registrations,
    register_instruction,
    register_tool,
    validate_tool,
)


class TestSelfImprovement:
    def test_register_instruction_writes_manifest(self, tmp_cwd):
        """Calling register_instruction creates the registration in the manifest file."""
        result = register_instruction("cheerio", '"end with Cheerio!"')
        assert result.startswith("OK:")
        assert "instruction/cheerio" in result

        # Verify it's persisted
        m = Manifest.load()
        reg = m.find("cheerio")
        assert reg is not None
        assert reg.slot == "instruction"
        assert reg.code == '"end with Cheerio!"'

    def test_list_registrations_shows_registered(self, tmp_cwd):
        """After registering, list_registrations includes the entry."""
        register_instruction("cheerio", '"end with Cheerio!"')
        listing = list_registrations()
        assert "cheerio" in listing
        assert "instruction" in listing

    def test_duplicate_name_errors(self, tmp_cwd):
        """Registering the same name twice returns an error string."""
        register_instruction("cheerio", '"end with Cheerio!"')
        result = register_instruction("cheerio", '"something else"')
        assert result.startswith("ERROR:")
        assert "already exists" in result

    def test_end_to_end_self_improvement_cycle(self, tmp_cwd):
        """
        Simulates the self-improvement cycle:
        1. Agent registers an instruction
        2. On next build, the instruction is loaded into the capability
        """
        from pa.capability import PaRegistrations

        # Step 1: "Agent" registers
        result = register_instruction("cheerio", '"Always end your reply with Cheerio!"')
        assert result.startswith("OK:")

        # Step 2: Build the capability (simulating next agent construction)
        cap = PaRegistrations()
        # Verify that instruction functions are exposed through the native capability hook
        assert cap.get_instructions()


async def test_register_tool_sync_wrapper_works_inside_event_loop(tmp_cwd):
    """The public sync helper should not depend on the caller lacking an event loop."""
    result = register_tool(
        "double",
        "Double an integer.",
        'args["x"] * 2',
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        {"x": 2},
    )

    assert result.startswith("OK:")
    reg = Manifest.load().find("double")
    assert reg is not None
    assert reg.status == "active"


def test_register_tool_persists_timeout(tmp_cwd):
    result = register_tool(
        "complete_backed",
        "Use complete for a bounded sub-task.",
        'args["text"].upper()',
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        {"text": "hello"},
        timeout_s=12.0,
    )

    assert result.startswith("OK:")
    reg = Manifest.load().find("complete_backed")
    assert reg is not None
    assert reg.status == "active"
    assert reg.timeout_s == 12.0
    listing = json.loads(list_registrations())
    assert listing[0]["timeout_s"] == 12.0


def test_register_tool_success_message_explains_native_next_run(tmp_cwd):
    result = register_tool(
        "lower",
        "Lowercase text.",
        'args["text"].lower()',
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        {"text": "HELLO"},
    )

    assert result.startswith("OK:")
    assert "next agent run" in result
    assert "run_code" in result
    assert "later in this run" in result


def test_validate_tool_success_message_explains_native_next_run(tmp_cwd):
    result = register_tool(
        "lower",
        "Lowercase text.",
        'args["text"].lower()',
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )
    assert result.startswith("OK:")

    result = validate_tool("lower", {"text": "HELLO"})

    assert result.startswith("OK:")
    assert "next agent run" in result
    assert "run_code" in result
    assert "later in this run" in result


def test_register_tool_rejects_unawaited_async_primitive(tmp_cwd):
    result = register_tool(
        "bad_complete",
        "Incorrectly uses complete without await.",
        (
            "schema = {\n"
            '    "type": "object",\n'
            '    "properties": {"category": {"type": "string"}},\n'
            '    "required": ["category"],\n'
            '    "additionalProperties": False,\n'
            "}\n"
            '{"category": complete(prompt="Classify the request", output_schema=schema)}'
        ),
        {"type": "object", "properties": {}, "additionalProperties": False},
        {},
    )

    assert result.startswith("OK:")
    assert "disabled" in result
    assert "did you forget to await" in result
    reg = Manifest.load().find("bad_complete")
    assert reg is not None
    assert reg.status == "disabled"
    assert "did you forget to await" in reg.last_error


def test_register_tool_rejects_complete_without_output_schema(tmp_cwd):
    result = register_tool(
        "prompt_json",
        "Incorrectly asks complete for prompt-only JSON.",
        'await complete(prompt="Return JSON for " + args["text"])',
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        {"text": "hello"},
    )

    assert result.startswith("ERROR: invalid registration:")
    assert "output_schema" in result
    assert Manifest.load().registrations == []


def test_register_tool_validates_output_schema(tmp_cwd, monkeypatch):
    schema = {
        "type": "object",
        "properties": {"category": {"type": "string"}, "urgency": {"type": "string"}},
        "required": ["category", "urgency"],
        "additionalProperties": False,
    }

    async def fake_complete(prompt, system, data, output_schema, output_mode):
        assert output_schema == schema
        return {"category": "bug", "urgency": "high"}

    monkeypatch.setattr(primitives, "_complete_fn", fake_complete)

    result = register_tool(
        "triage",
        "Classify text.",
        (
            "schema = {\n"
            '    "type": "object",\n'
            '    "properties": {"category": {"type": "string"}, "urgency": {"type": "string"}},\n'
            '    "required": ["category", "urgency"],\n'
            '    "additionalProperties": False,\n'
            "}\n"
            'await complete(prompt="Classify", data=args["text"], output_schema=schema)'
        ),
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        {"text": "checkout failed"},
        output_json_schema=schema,
        expected_example_output={"category": "bug", "urgency": "high"},
    )

    assert result.startswith("OK:")
    reg = Manifest.load().find("triage")
    assert reg is not None
    assert reg.status == "active"
    assert reg.output_json_schema == schema
    assert reg.expected_example_output == {"category": "bug", "urgency": "high"}
    listing = json.loads(list_registrations())
    assert listing[0]["has_output_schema"] is True
    assert listing[0]["has_expected_example_output"] is True


def test_register_tool_allows_schema_without_expected_output(tmp_cwd):
    result = register_tool(
        "line_count",
        "Count lines.",
        '{"total": 3}',
        {"type": "object", "properties": {}, "additionalProperties": False},
        {},
        output_json_schema={
            "type": "object",
            "properties": {"total": {"type": "integer"}},
            "required": ["total"],
        },
    )

    assert result.startswith("OK:")
    assert "exact example semantics were not checked" in result
    reg = Manifest.load().find("line_count")
    assert reg is not None
    assert reg.status == "active"
    assert reg.output_json_schema is not None
    assert reg.output_json_schema["additionalProperties"] is False
    listing = json.loads(list_registrations())
    assert listing[0]["has_output_schema"] is True
    assert listing[0]["has_expected_example_output"] is False


def test_register_tool_rejects_expected_output_mismatch(tmp_cwd):
    result = register_tool(
        "line_count",
        "Count lines.",
        '{"total": 999}',
        {"type": "object", "properties": {}, "additionalProperties": False},
        {},
        output_json_schema={
            "type": "object",
            "properties": {"total": {"type": "integer"}},
            "required": ["total"],
            "additionalProperties": False,
        },
        expected_example_output={"total": 3},
    )

    assert result.startswith("OK:")
    assert "disabled" in result
    assert "example output mismatch" in result


def test_register_tool_disables_output_schema_mismatch(tmp_cwd):
    result = register_tool(
        "bad_output",
        "Returns the wrong output shape.",
        '{"category": "bug"}',
        {"type": "object", "properties": {}, "additionalProperties": False},
        {},
        output_json_schema={
            "type": "object",
            "properties": {"category": {"type": "string"}, "urgency": {"type": "string"}},
            "required": ["category", "urgency"],
            "additionalProperties": False,
        },
        expected_example_output={"category": "bug"},
    )

    assert result.startswith("OK:")
    assert "disabled" in result
    assert "missing required field" in result


def test_register_tool_rejects_timeout_above_limit(tmp_cwd):
    result = register_tool(
        "too_slow",
        "Timeout exceeds the registration cap.",
        '"ok"',
        {"type": "object", "properties": {}, "additionalProperties": False},
        {},
        timeout_s=61.0,
    )

    assert result.startswith("ERROR: invalid registration:")
    assert "less than or equal to 60" in result
    assert Manifest.load().registrations == []


def test_validate_tool_uses_registered_timeout(tmp_cwd):
    result = register_tool(
        "never_finishes",
        "Loop forever.",
        "x = 0\nwhile True:\n    x = x + 1\nx",
        {"type": "object", "properties": {}, "additionalProperties": False},
        timeout_s=0.05,
    )
    assert result.startswith("OK:")

    result = validate_tool("never_finishes", {})

    assert result.startswith("ERROR: validation failed for tool/never_finishes:")
    reg = Manifest.load().find("never_finishes")
    assert reg is not None
    assert reg.status == "disabled"
    assert reg.timeout_s == 0.05
    assert reg.last_error


def test_check_registrations_records_health(tmp_cwd):
    m = Manifest()
    m.add(Registration(slot="instruction", name="broken", code="missing_name"))
    m.save()

    report = json.loads(check_registrations())

    assert report[0]["name"] == "broken"
    assert report[0]["check"] == "error"
    reg = Manifest.load().find("broken")
    assert reg is not None
    assert reg.last_error
    assert reg.last_run_status == "error"


def test_check_registrations_reports_compaction_policy_errors(tmp_cwd):
    m = Manifest()
    m.add(Registration(slot="compaction", name="drops_current", code="[0]"))
    m.save()

    report = json.loads(check_registrations())

    assert report[0]["check"] == "error"
    assert "current request" in report[0]["last_error"]


def test_disable_registration_quarantines_any_slot(tmp_cwd):
    m = Manifest()
    m.add(Registration(slot="before_tool_hook", name="broken_hook", code="missing_name"))
    m.save()

    result = disable_registration("broken_hook", "quarantined")

    assert result == "OK: disabled before_tool_hook/broken_hook."
    reg = Manifest.load().find("broken_hook")
    assert reg is not None
    assert reg.status == "disabled"
    assert reg.last_error == "quarantined"
    report = json.loads(check_registrations())
    assert report[0]["check"] == "skipped"
    assert report[0]["reason"] == "disabled"
