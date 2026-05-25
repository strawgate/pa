import pytest
from pydantic import ValidationError

from pa.manifest import Manifest, Registration, ManifestError, CardinalityError


class TestManifestRoundTrip:
    def test_round_trip_three_registrations(self, tmp_cwd):
        m = Manifest()
        m.add(Registration(slot="instruction", name="greet", code='"hello"'))
        m.add(
            Registration(
                slot="before_tool_hook",
                name="no_bash",
                code='{"action": "deny", "reason": "no bash"}',
            )
        )
        m.add(Registration(slot="tool_filter", name="keep_read", code='["read_file"]'))
        m.save()

        loaded = Manifest.load()
        assert len(loaded.registrations) == 3
        assert loaded.find("greet") is not None
        assert loaded.find("no_bash") is not None
        assert loaded.find("keep_read") is not None

    def test_active_by_slot_filters_disabled_registrations(self, tmp_cwd):
        m = Manifest()
        m.add(Registration(slot="instruction", name="active_note", code='"active"'))
        m.add(Registration(slot="instruction", name="disabled_note", code='"disabled"', status="disabled"))

        active = m.active_by_slot("instruction")

        assert [r.name for r in active] == ["active_note"]

    def test_two_compaction_raises_cardinality_error(self, tmp_cwd):
        m = Manifest()
        m.add(Registration(slot="compaction", name="compact_one", code="[0]"))
        with pytest.raises(CardinalityError, match="single-cardinality"):
            m.add(Registration(slot="compaction", name="compact_two", code="[1]"))

    def test_same_name_raises_manifest_error(self, tmp_cwd):
        m = Manifest()
        m.add(Registration(slot="instruction", name="greet", code='"hello"'))
        with pytest.raises(ManifestError, match="already exists"):
            m.add(Registration(slot="before_tool_hook", name="greet", code='{"action": "allow"}'))

    def test_invalid_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Registration(slot="instruction", name="My-Reg", code='"hi"')

    def test_nonexistent_path_returns_empty_manifest(self, tmp_cwd):
        loaded = Manifest.load("nonexistent/path.yaml")
        assert loaded.registrations == []


class TestManifestRemove:
    def test_remove_existing(self, tmp_cwd):
        m = Manifest()
        m.add(Registration(slot="instruction", name="greet", code='"hello"'))
        removed = m.remove("greet")
        assert removed.name == "greet"
        assert m.registrations == []

    def test_remove_nonexistent_raises(self, tmp_cwd):
        m = Manifest()
        with pytest.raises(ManifestError, match="no registration named"):
            m.remove("nope")
