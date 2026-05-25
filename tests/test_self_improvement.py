import json

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
