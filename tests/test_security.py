from pathlib import Path

import pytest

from agent_runner.security import validate_auth_file
from agent_runner.session import _process_record, process_alive


def test_auth_file_must_be_private(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    auth.chmod(0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        validate_auth_file(auth)
    auth.chmod(0o600)
    assert validate_auth_file(auth) == auth.resolve()


def test_process_identity_prevents_stale_pid_reuse() -> None:
    import os

    record = _process_record(os.getpid())
    assert process_alive(record["pid"], record["start_ticks"])
    assert not process_alive(record["pid"], record["start_ticks"] + 1)
