import json

from pa.manifest import Manifest, Registration
from pa.registration_tools import check_registrations, list_registrations, register_instruction, register_tool


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
