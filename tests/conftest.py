import pytest


@pytest.fixture(autouse=True)
def isolated_pa_home(tmp_path, monkeypatch):
    """Keep ~/.pa state out of developer machines during tests."""
    monkeypatch.setenv("PA_HOME", str(tmp_path / ".pa-home"))


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Isolate each test in a fresh CWD so registrations do not leak."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pa").mkdir()
    return tmp_path
