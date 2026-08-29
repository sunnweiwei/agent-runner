from __future__ import annotations

from pathlib import Path
import shutil
from typing import Literal

import toml
import yaml


NetworkMode = Literal["task", "public", "no-network", "allowlist"]


def validate_task(task_dir: Path) -> Path:
    task_dir = task_dir.expanduser().resolve()
    required = (
        "task.toml",
        "instruction.md",
        "tests/test.sh",
    )
    missing = [name for name in required if not (task_dir / name).is_file()]
    if missing:
        raise ValueError(f"Not a supported Harbor task; missing {', '.join(missing)}")
    config = toml.load(task_dir / "task.toml")
    artifact_sources = {
        artifact if isinstance(artifact, str) else artifact.get("source")
        for artifact in config.get("artifacts", [])
    }
    if "/workspace/submission" not in artifact_sources:
        raise ValueError(
            "Model-development tasks must declare /workspace/submission as a Harbor artifact"
        )
    return task_dir


def require_separate_verifier(task_dir: Path) -> None:
    config = toml.load(task_dir / "task.toml")
    if config.get("verifier", {}).get("environment_mode") != "separate":
        raise ValueError(
            "Independent evaluation requires verifier.environment_mode = 'separate'"
        )


def task_gpu_count(task_dir: Path) -> int:
    config = toml.load(task_dir / "task.toml")
    return int(config.get("environment", {}).get("gpus") or 0)


def _set_policy(table: dict, mode: str, allowed_hosts: list[str] | None = None) -> None:
    table["network_mode"] = mode
    if mode == "allowlist":
        table["allowed_hosts"] = list(allowed_hosts or [])
    else:
        table.pop("allowed_hosts", None)


def _patch_network(config: dict, mode: NetworkMode, allowed_hosts: list[str]) -> None:
    if mode == "task":
        return
    environment = config.setdefault("environment", {})
    agent = config.setdefault("agent", {})
    verifier = config.setdefault("verifier", {})
    verifier_environment = verifier.setdefault("environment", {})
    if mode == "allowlist":
        if not allowed_hosts:
            raise ValueError("allowlist network mode needs at least one --allow-host")
        _set_policy(environment, mode, allowed_hosts)
        _set_policy(agent, mode, allowed_hosts)
    else:
        _set_policy(environment, mode)
        _set_policy(agent, mode)
    # Evaluation never needs model-provider or package-index egress.
    _set_policy(verifier_environment, "no-network")
    _set_policy(verifier, "no-network")


def _patch_gpu_compose(path: Path, gpus: int) -> None:
    if gpus <= 0:
        return
    compose = yaml.safe_load(path.read_text()) if path.is_file() else {}
    compose = compose or {}
    main = compose.setdefault("services", {}).setdefault("main", {})
    deploy = main.setdefault("deploy", {})
    resources = deploy.setdefault("resources", {})
    reservations = resources.setdefault("reservations", {})
    devices = reservations.setdefault("devices", [])
    devices[:] = [
        device
        for device in devices
        if not isinstance(device, dict) or device.get("driver") != "nvidia"
    ]
    devices.append({"driver": "nvidia", "count": gpus, "capabilities": ["gpu"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compose, sort_keys=False))


def _patch_no_network_compose(path: Path) -> None:
    compose = yaml.safe_load(path.read_text()) if path.is_file() else {}
    compose = compose or {}
    compose.setdefault("services", {}).setdefault("main", {})["network_mode"] = "none"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compose, sort_keys=False))


def stage_task(
    source: Path,
    destination: Path,
    *,
    network: NetworkMode,
    allowed_hosts: list[str],
    gpus: int,
    development_image: str | None = None,
    verifier_image: str | None = None,
) -> Path:
    source = validate_task(source)
    if destination.exists():
        raise FileExistsError(f"Runtime task already exists: {destination}")
    shutil.copytree(source, destination, symlinks=True)
    config_path = destination / "task.toml"
    config = toml.load(config_path)
    _patch_network(config, network, allowed_hosts)
    if development_image:
        config.setdefault("environment", {})["docker_image"] = development_image
    if verifier_image:
        config.setdefault("verifier", {}).setdefault("environment", {})[
            "docker_image"
        ] = verifier_image
    config_path.write_text(toml.dumps(config))
    _patch_gpu_compose(destination / "environment/docker-compose.yaml", gpus)
    _patch_gpu_compose(destination / "tests/docker-compose.yaml", gpus)
    if network == "no-network":
        _patch_no_network_compose(destination / "environment/docker-compose.yaml")
        _patch_no_network_compose(destination / "tests/docker-compose.yaml")
    return destination
