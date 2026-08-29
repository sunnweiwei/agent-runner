from types import SimpleNamespace

from agent_runner.environment import AllowlistDockerEnvironment


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
