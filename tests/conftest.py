import pytest


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Isolate each test in a fresh CWD so pa/registrations.yaml does not leak."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pa").mkdir()
    return tmp_path
