from pathlib import Path
import re
from types import SimpleNamespace

from agent_runner.environment import AllowlistDockerEnvironment
from agent_runner.naming import unique_trial_name


def test_allowlist_probe_does_not_use_default_bridge(monkeypatch) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("agent_runner.environment.subprocess.run", fake_run)
    assert AllowlistDockerEnvironment._egress_control_kernel_support() is True
    assert observed["command"][0:4] == [
        "docker",
        "container",
        "run",
        "--network",
    ]
    assert observed["command"][4] == "none"


def test_trial_names_isolate_parallel_run_directories(tmp_path: Path) -> None:
    first = unique_trial_name("development", tmp_path / "owner-a" / "run")
    second = unique_trial_name("development", tmp_path / "owner-b" / "run")
    assert first != second
    assert first == unique_trial_name("development", tmp_path / "owner-a" / "run")
    assert re.fullmatch(r"[a-z0-9_-]+", first)
