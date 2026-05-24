from pa.manifest import Manifest
from pa.registration_tools import register_instruction, list_registrations


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
        # Verify that instruction functions were created
        assert cap._sub  # At least one sub-capability was built
