from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Iterator

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from .io import read_json, write_json_atomic
from .snapshot import verify_snapshot
from .task import require_separate_verifier, stage_task


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_snapshot(run_dir: Path, snapshot: str) -> Path:
    snapshots = (run_dir / "snapshots").resolve()
    if snapshot == "latest":
        pointer = read_json(snapshots / "latest.json")
        candidate = Path(pointer["path"]).resolve()
    else:
        candidate = (snapshots / snapshot).resolve()
    if not candidate.is_relative_to(snapshots) or not candidate.is_dir():
        raise ValueError(f"Unknown snapshot: {snapshot}")
    verify_snapshot(candidate)
    return candidate


async def _evaluate(
    run_dir: Path,
    snapshot_dir: Path,
    output_dir: Path,
    data_dir: Path | None,
    gpus: int,
    runtime_env: dict[str, str],
) -> dict:
    run = read_json(run_dir / "run.json")
    runtime_task = stage_task(
        Path(run["task_dir"]),
        output_dir / "runtime-task",
        network="no-network",
        allowed_hosts=[],
        gpus=gpus,
        development_image=run.get("development_image"),
        verifier_image=run.get("verifier_image"),
    )
    require_separate_verifier(runtime_task)
    snapshot_id = snapshot_dir.name
    config = TrialConfig(
        task=TaskConfig(path=runtime_task),
        trial_name=f"evaluate-{snapshot_id[-20:]}",
        trials_dir=output_dir / "harbor",
        agent=AgentConfig(name="agent_runner.agents:ArtifactReplayAgent"),
        environment=EnvironmentConfig(
            import_path="agent_runner.environment:NoNetworkDockerEnvironment",
            override_gpus=0,
            mounts=[
                {
                    "type": "bind",
                    "source": str((snapshot_dir / "submission").resolve()),
                    "target": "/runner-input/submission",
                    "read_only": True,
                    "bind": {"create_host_path": False},
                }
            ],
            delete=True,
        ),
    )
    data_env_var = run.get("data_env_var", "FOLDBENCH_DATA_DIR")
    environment = dict(runtime_env)
    if data_dir is not None:
        environment[data_env_var] = str(data_dir)
    with _temporary_environment(environment):
        trial = await Trial.create(config)
        result = await trial.run()
    verify_snapshot(snapshot_dir)
    record = {
        "schema_version": "agent-runner-evaluation/v1",
        "snapshot_id": snapshot_id,
        "snapshot_tree_sha256": read_json(snapshot_dir / "manifest.json")[
            "tree_sha256"
        ],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "trial_dir": str((output_dir / "harbor" / config.trial_name).resolve()),
        "rewards": result.verifier_result.rewards if result.verifier_result else None,
        "status": "failed" if result.exception_info else "completed",
        "error": result.exception_info.exception_message
        if result.exception_info
        else None,
        "exception_type": result.exception_info.exception_type
        if result.exception_info
        else None,
    }
    write_json_atomic(output_dir / "evaluation.json", record)
    if result.exception_info:
        raise RuntimeError(
            f"External evaluation failed: {result.exception_info.exception_type}: "
            f"{result.exception_info.exception_message}"
        )
    return record


def evaluate_snapshot(
    run_dir: Path,
    *,
    snapshot: str = "latest",
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    gpus: int | None = None,
    runtime_env: dict[str, str] | None = None,
) -> dict:
    run_dir = run_dir.expanduser().resolve()
    run = read_json(run_dir / "run.json")
    snapshot_dir = resolve_snapshot(run_dir, snapshot)
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir
        else run_dir / "evaluations" / snapshot_dir.name
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_data = data_dir.expanduser().resolve() if data_dir else None
    if resolved_data is None and run.get("data_dir"):
        resolved_data = Path(run["data_dir"]).resolve()
    if resolved_data is not None and not resolved_data.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {resolved_data}")
    return asyncio.run(
        _evaluate(
            run_dir,
            snapshot_dir,
            output_dir,
            resolved_data,
            int(run["gpus"] if gpus is None else gpus),
            runtime_env or {},
        )
    )
