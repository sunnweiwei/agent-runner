from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .evaluate import evaluate_snapshot
from .session import materialize, run_status, snapshot_now, start_run, stop_run


def _key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE")
    key, item = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("Environment/config key cannot be empty")
    return key, item


def _kwargs(values: list[tuple[str, str]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in values:
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"--agent-kwarg {key} must contain a JSON value"
            ) from error
    return parsed


def _mapping(values: list[tuple[str, str]]) -> dict[str, str]:
    return dict(values)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="agent-runner")
    result.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = result.add_subparsers(dest="command", required=True)

    start = commands.add_parser(
        "start", help="Start one detached Harbor development session"
    )
    start.add_argument("--task", type=Path, required=True)
    start.add_argument("--run-dir", type=Path, required=True)
    start.add_argument("--data-dir", type=Path)
    start.add_argument("--data-env-var", default="FOLDBENCH_DATA_DIR")
    start.add_argument("--agent", required=True)
    start.add_argument("--model")
    start.add_argument("--agent-kwarg", type=_key_value, action="append", default=[])
    start.add_argument("--auth-file", type=Path)
    start.add_argument(
        "--agent-bin",
        type=_key_value,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Mount an operator-supplied executable at /usr/local/bin/NAME",
    )
    start.add_argument("--env", type=_key_value, action="append", default=[])
    start.add_argument(
        "--network",
        choices=("task", "public", "no-network", "allowlist"),
        default="no-network",
    )
    start.add_argument("--allow-host", action="append", default=[])
    start.add_argument("--gpus", default="task")
    start.add_argument("--snapshot-interval", type=float, default=5.0)
    start.add_argument(
        "--duration-hours",
        type=float,
        help="Stop the complete session after this many wall-clock hours",
    )
    start.add_argument("--development-image")
    start.add_argument("--verifier-image")

    status = commands.add_parser("status", help="Show durable session state")
    status.add_argument("run_dir", type=Path)

    snapshot = commands.add_parser(
        "snapshot", help="Publish a stable immutable snapshot now"
    )
    snapshot.add_argument("run_dir", type=Path)

    stop = commands.add_parser("stop", help="Gracefully stop the Harbor session")
    stop.add_argument("run_dir", type=Path)
    stop.add_argument("--timeout", type=float, default=45.0)
    stop.add_argument("--force", action="store_true")

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate an immutable snapshot independently"
    )
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--snapshot", default="latest")
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--data-dir", type=Path)
    evaluate.add_argument("--gpus", type=int)
    evaluate.add_argument("--env", type=_key_value, action="append", default=[])

    prepare = commands.add_parser(
        "materialize", help="Run the task-owned data materializer"
    )
    prepare.add_argument("--task", type=Path, required=True)
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--env", type=_key_value, action="append", default=[])
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "start":
            if args.network == "public":
                print(
                    "warning: agent runtime has unrestricted network access",
                    file=sys.stderr,
                )
            record = start_run(
                task_dir=args.task,
                run_dir=args.run_dir,
                data_dir=args.data_dir,
                agent=args.agent,
                model=args.model,
                agent_kwargs=_kwargs(args.agent_kwarg),
                network=args.network,
                allowed_hosts=args.allow_host,
                gpus=args.gpus,
                snapshot_interval=args.snapshot_interval,
                development_image=args.development_image,
                verifier_image=args.verifier_image,
                auth_file=args.auth_file,
                agent_bins=_mapping(args.agent_bin),
                runtime_env=_mapping(args.env),
                data_env_var=args.data_env_var,
                duration_seconds=(
                    args.duration_hours * 3600
                    if args.duration_hours is not None
                    else None
                ),
            )
            _print(record)
        elif args.command == "status":
            _print(run_status(args.run_dir))
        elif args.command == "snapshot":
            _print(snapshot_now(args.run_dir))
        elif args.command == "stop":
            _print(stop_run(args.run_dir, timeout=args.timeout, force=args.force))
        elif args.command == "evaluate":
            _print(
                evaluate_snapshot(
                    args.run_dir,
                    snapshot=args.snapshot,
                    output_dir=args.output_dir,
                    data_dir=args.data_dir,
                    gpus=args.gpus,
                    runtime_env=_mapping(args.env),
                )
            )
        elif args.command == "materialize":
            materialize(args.task, args.data_dir, runtime_env=_mapping(args.env))
        else:
            raise AssertionError(args.command)
    except (
        FileNotFoundError,
        FileExistsError,
        PermissionError,
        TimeoutError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"agent-runner: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
