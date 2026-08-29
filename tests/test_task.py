from pathlib import Path

import toml
import pytest
import yaml

from agent_runner.session import start_run
from agent_runner.task import stage_task


def fixture_task(root: Path) -> Path:
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.0"\nartifacts = ["/workspace/submission"]\n'
        '[task]\nname = "test/task"\n'
        "[environment]\ngpus = 8\n"
        '[verifier]\nenvironment_mode = "separate"\n'
        "[verifier.environment]\ngpus = 8\n"
    )
    (task / "instruction.md").write_text("develop")
    (task / "environment/Dockerfile").write_text("FROM scratch\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\n")
    for path in (
        task / "environment/docker-compose.yaml",
        task / "tests/docker-compose.yaml",
    ):
        path.write_text("services:\n  main:\n    shm_size: 1gb\n")
    return task


def test_stage_task_adds_offline_policy_images_and_gpu_devices(tmp_path: Path) -> None:
    source = fixture_task(tmp_path)
    staged = stage_task(
        source,
        tmp_path / "runtime",
        network="allowlist",
        allowed_hosts=["api.example.com"],
        gpus=8,
        development_image="dev:test",
        verifier_image="verifier:test",
    )
    config = toml.load(staged / "task.toml")
    assert config["environment"]["network_mode"] == "allowlist"
    assert config["agent"]["allowed_hosts"] == ["api.example.com"]
    assert config["verifier"]["network_mode"] == "no-network"
    assert config["verifier"]["environment"]["docker_image"] == "verifier:test"
    assert config["environment"]["docker_image"] == "dev:test"
    for path in (
        staged / "environment/docker-compose.yaml",
        staged / "tests/docker-compose.yaml",
    ):
        compose = yaml.safe_load(path.read_text())
        device = compose["services"]["main"]["deploy"]["resources"]["reservations"][
            "devices"
        ][0]
        assert device == {"driver": "nvidia", "count": 8, "capabilities": ["gpu"]}
    assert "docker_image" not in toml.load(source / "task.toml")["environment"]


def test_no_network_uses_compose_none_for_both_sandboxes(tmp_path: Path) -> None:
    staged = stage_task(
        fixture_task(tmp_path),
        tmp_path / "runtime",
        network="no-network",
        allowed_hosts=[],
        gpus=0,
    )
    for path in (
        staged / "environment/docker-compose.yaml",
        staged / "tests/docker-compose.yaml",
    ):
        assert (
            yaml.safe_load(path.read_text())["services"]["main"]["network_mode"]
            == "none"
        )


def test_stage_creates_compose_overlays_when_task_has_none(tmp_path: Path) -> None:
    source = fixture_task(tmp_path)
    (source / "environment/docker-compose.yaml").unlink()
    (source / "tests/docker-compose.yaml").unlink()
    staged = stage_task(
        source,
        tmp_path / "runtime",
        network="no-network",
        allowed_hosts=[],
        gpus=2,
    )
    for path in (
        staged / "environment/docker-compose.yaml",
        staged / "tests/docker-compose.yaml",
    ):
        main = yaml.safe_load(path.read_text())["services"]["main"]
        assert main["network_mode"] == "none"
        assert main["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 2


def test_start_rejects_missing_agent_binary_before_launch(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Agent executable does not exist"):
        start_run(
            task_dir=fixture_task(tmp_path),
            run_dir=tmp_path / "run",
            data_dir=None,
            agent="codex",
            model="openai/test",
            agent_kwargs={},
            network="allowlist",
            allowed_hosts=["api.example.com"],
            gpus="0",
            snapshot_interval=1,
            development_image=None,
            verifier_image=None,
            auth_file=None,
            agent_bins={"codex": str(tmp_path / "missing")},
        )
