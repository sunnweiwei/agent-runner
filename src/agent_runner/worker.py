from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import signal
import sys

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial

from .io import read_json, update_json
from .snapshot import create_snapshot


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _snapshot_loop(run_dir: Path, interval: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            create_snapshot(
                run_dir / "live/submission",
                run_dir / "snapshots",
                settle_seconds=min(0.2, interval / 4),
            )
        except Exception as error:
            update_json(run_dir / "state.json", snapshot_error=str(error))
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_worker(run_dir: Path) -> int:
    config = read_json(run_dir / "run.json")
    stop = asyncio.Event()
    stop_reason: dict[str, str | None] = {"value": None}
    loop = asyncio.get_running_loop()

    def request_stop(reason: str) -> None:
        if not stop.is_set():
            stop_reason["value"] = reason
            stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            signum,
            request_stop,
            f"signal:{signal.Signals(signum).name}",
        )

    duration_seconds = config.get("duration_seconds")
    deadline_handle = None
    if duration_seconds is not None:
        duration_seconds = float(duration_seconds)
        deadline = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
        update_json(run_dir / "state.json", deadline_at=deadline.isoformat())
        deadline_handle = loop.call_later(
            duration_seconds, request_stop, "duration_elapsed"
        )

    mounts = [
        {
            "type": "bind",
            "source": str((run_dir / "live/submission").resolve()),
            "target": "/workspace/submission",
            "bind": {"create_host_path": False},
        }
    ]
    for name, source in (config.get("agent_bins") or {}).items():
        mounts.append(
            {
                "type": "bind",
                "source": source,
                "target": f"/usr/local/bin/{name}",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        )

    environment_import = None
    if config["network"] == "no-network":
        environment_import = "agent_runner.environment:NoNetworkDockerEnvironment"
    elif config["network"] == "allowlist":
        environment_import = "agent_runner.environment:AllowlistDockerEnvironment"

    trial_config = TrialConfig(
        task=TaskConfig(path=Path(config["runtime_task_dir"])),
        trial_name="development",
        trials_dir=run_dir / "harbor",
        agent=AgentConfig(
            name=config["agent"],
            model_name=config.get("model"),
            kwargs=config.get("agent_kwargs") or {},
        ),
        environment=EnvironmentConfig(
            import_path=environment_import,
            override_gpus=0,
            mounts=mounts,
            delete=True,
        ),
        verifier=VerifierConfig(disable=True),
    )
    try:
        trial = await Trial.create(trial_config)
    except BaseException as error:
        update_json(
            run_dir / "state.json",
            status="failed",
            finished_at=_now(),
            error=str(error),
            exception_type=type(error).__name__,
        )
        return 1

    async def on_agent_start(_event) -> None:
        update_json(
            run_dir / "state.json",
            status="running",
            started_at=_now(),
            pid=os.getpid(),
        )

    trial.add_hook(TrialEvent.AGENT_START, on_agent_start)
    snapshot_task = asyncio.create_task(
        _snapshot_loop(run_dir, float(config["snapshot_interval"]), stop)
    )
    trial_task = asyncio.create_task(trial.run())
    stop_task = asyncio.create_task(stop.wait())
    status = "failed"
    exit_code = 1
    try:
        done, _ = await asyncio.wait(
            {trial_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done and stop.is_set() and not trial_task.done():
            trial_task.cancel()
            try:
                await trial_task
            except asyncio.CancelledError:
                pass
            status = "stopped"
            exit_code = 0
        else:
            result = await trial_task
            if result.exception_info is None:
                status = "completed"
                exit_code = 0
            else:
                status = "failed"
                update_json(
                    run_dir / "state.json",
                    error=result.exception_info.exception_message,
                    exception_type=result.exception_info.exception_type,
                )
    except BaseException as error:
        if (
            isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            or stop.is_set()
        ):
            status = "stopped"
            exit_code = 0
        else:
            update_json(
                run_dir / "state.json",
                error=str(error),
                exception_type=type(error).__name__,
            )
    finally:
        if deadline_handle is not None:
            deadline_handle.cancel()
        stop.set()
        stop_task.cancel()
        await snapshot_task
        try:
            create_snapshot(run_dir / "live/submission", run_dir / "snapshots")
        except Exception:
            pass
        update_json(
            run_dir / "state.json",
            status=status,
            finished_at=_now(),
            stop_reason=stop_reason["value"],
        )
    return exit_code


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m agent_runner.worker RUN_DIR")
    raise SystemExit(asyncio.run(run_worker(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
