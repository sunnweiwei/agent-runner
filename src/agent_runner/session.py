from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from .io import read_json, write_json_atomic
from .security import validate_auth_file
from .snapshot import create_snapshot, latest_snapshot
from .task import NetworkMode, stage_task, task_gpu_count, validate_task


ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _new_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_gpu_count(task_dir: Path, value: str) -> int:
    if value == "task":
        return task_gpu_count(task_dir)
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("--gpus must be 'task' or a non-negative integer") from error
    if count < 0:
        raise ValueError("--gpus must be non-negative")
    return count


def start_run(
    *,
    task_dir: Path,
    run_dir: Path,
    data_dir: Path | None,
    agent: str,
    model: str | None,
    agent_kwargs: dict[str, Any],
    network: NetworkMode,
    allowed_hosts: list[str],
    gpus: str,
    snapshot_interval: float,
    development_image: str | None,
    verifier_image: str | None,
    auth_file: Path | None,
    runtime_env: dict[str, str] | None = None,
    data_env_var: str = "FOLDBENCH_DATA_DIR",
) -> dict:
    task_dir = validate_task(task_dir)
    run_dir = _new_directory(run_dir)
    if snapshot_interval <= 0:
        raise ValueError("Snapshot interval must be positive")
    gpu_count = resolve_gpu_count(task_dir, gpus)
    data_path = data_dir.expanduser().resolve() if data_dir else None
    if not ENV_NAME.fullmatch(data_env_var):
        raise ValueError(f"Invalid --data-env-var name: {data_env_var}")
    if data_path is not None and not data_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    if auth_file is not None:
        auth_file = validate_auth_file(auth_file)

    runtime_task = stage_task(
        task_dir,
        run_dir / "runtime-task",
        network=network,
        allowed_hosts=allowed_hosts,
        gpus=gpu_count,
        development_image=development_image,
        verifier_image=verifier_image,
    )
    live_submission = run_dir / "live/submission"
    live_submission.mkdir(parents=True)
    starter = runtime_task / "environment/starter"
    if starter.is_dir():
        shutil.copytree(starter, live_submission, dirs_exist_ok=True, symlinks=True)

    config = {
        "schema_version": "agent-runner-run/v1",
        "task_dir": str(task_dir),
        "runtime_task_dir": str(runtime_task),
        "run_dir": str(run_dir),
        "data_dir": str(data_path) if data_path else None,
        "data_env_var": data_env_var,
        "agent": agent,
        "model": model,
        "agent_kwargs": agent_kwargs,
        "network": network,
        "allowed_hosts": allowed_hosts,
        "gpus": gpu_count,
        "snapshot_interval": snapshot_interval,
        "development_image": development_image,
        "verifier_image": verifier_image,
        "runtime_env_names": sorted((runtime_env or {}).keys()),
    }
    # Authentication paths and environment values are intentionally not persisted.
    write_json_atomic(run_dir / "run.json", config)
    write_json_atomic(
        run_dir / "state.json",
        {"schema_version": "agent-runner-state/v1", "status": "starting"},
    )

    child_env = os.environ.copy()
    child_env.update(runtime_env or {})
    if data_path is not None:
        child_env[data_env_var] = str(data_path)
    if auth_file is not None:
        child_env["CODEX_AUTH_JSON_PATH"] = str(auth_file)

    log_path = run_dir / "runner.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "agent_runner.worker", str(run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )
    write_json_atomic(run_dir / "pid.json", _process_record(process.pid))
    return {**config, "pid": process.pid, "state_path": str(run_dir / "state.json")}


def _process_details(pid: int) -> tuple[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        fields = raw.rsplit(")", 1)[1].strip().split()
        return fields[0], int(fields[19])
    except (IndexError, ValueError):
        return None


def _process_record(pid: int) -> dict[str, int]:
    details = _process_details(pid)
    return {"pid": pid, "start_ticks": details[1] if details else -1}


def process_alive(pid: int, start_ticks: int | None = None) -> bool:
    # kill(pid, 0) also succeeds for an exited child that has not been reaped.
    # Treat Linux zombies as stopped so detached-session status and stop do not
    # wait for their timeout merely because the short-lived launcher still lives.
    details = _process_details(pid)
    if details is None or details[0] == "Z":
        return False
    if start_ticks is not None and start_ticks >= 0 and details[1] != start_ticks:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def run_status(run_dir: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    state = read_json(run_dir / "state.json")
    pid_record = (
        read_json(run_dir / "pid.json") if (run_dir / "pid.json").is_file() else {}
    )
    pid = pid_record.get("pid")
    state["pid"] = pid
    state["process_alive"] = bool(
        pid and process_alive(int(pid), pid_record.get("start_ticks"))
    )
    latest = latest_snapshot(run_dir / "snapshots")
    state["latest_snapshot"] = latest
    state["live_submission"] = str(run_dir / "live/submission")
    return state


def snapshot_now(run_dir: Path) -> dict | None:
    run_dir = run_dir.expanduser().resolve()
    return create_snapshot(run_dir / "live/submission", run_dir / "snapshots")


def stop_run(run_dir: Path, *, timeout: float = 45.0, force: bool = False) -> dict:
    run_dir = run_dir.expanduser().resolve()
    pid_record = read_json(run_dir / "pid.json")
    pid = int(pid_record["pid"])
    start_ticks = pid_record.get("start_ticks")
    if not process_alive(pid, start_ticks):
        return run_status(run_dir)
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while process_alive(pid, start_ticks) and time.monotonic() < deadline:
        time.sleep(0.2)
    if process_alive(pid, start_ticks):
        if not force:
            raise TimeoutError(
                f"Runner {pid} did not stop within {timeout}s; retry with --force"
            )
        os.killpg(pid, signal.SIGKILL)
    return run_status(run_dir)


def materialize(task_dir: Path, data_dir: Path, *, runtime_env: dict[str, str]) -> None:
    task_dir = validate_task(task_dir)
    script = task_dir / "environment/data/materialize.sh"
    if not script.is_file():
        raise FileNotFoundError(f"Task has no materializer: {script}")
    environment = os.environ.copy()
    environment.update(runtime_env)
    result = subprocess.run(
        ["bash", str(script), str(data_dir.expanduser().resolve())], env=environment
    )
    if result.returncode:
        raise RuntimeError(
            f"Task data materializer exited with status {result.returncode}"
        )
