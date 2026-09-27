import pytest


@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    """Send audit records to a per-test file instead of logs/audit.jsonl."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FORMULATECH_AUDIT_LOG", str(path))
    return path
