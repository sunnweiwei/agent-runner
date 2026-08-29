from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import textwrap
import time

import pytest

from agent_runner.evaluate import evaluate_snapshot
from agent_runner.session import run_status, start_run, stop_run


pytestmark = [
    pytest.mark.foldbench_smoke,
    pytest.mark.skipif(
        os.environ.get("RUN_FOLDBENCH_SMOKE") != "1",
        reason="set RUN_FOLDBENCH_SMOKE=1 for the Docker integration test",
    ),
]


REQUEST_ID = "7pv5-assembly1"


def _write_smoke_data(root: Path, suite_source: Path, truth_source: Path) -> Path:
    suite = root / "foldbench/suite"
    request_source = suite_source / f"requests/{REQUEST_ID}.json"
    request = json.loads(request_source.read_text())
    request_path = suite / f"requests/{REQUEST_ID}.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    (suite / "targets").mkdir()
    with (suite / "targets/monomer_protein.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pdb_id", "chain_id"])
        writer.writeheader()
        writer.writerow({"pdb_id": REQUEST_ID, "chain_id": "A"})
    (suite / "suite_manifest.json").write_text(
        json.dumps(
            {
                "contains_ground_truth_coordinates": False,
                "requests": [
                    {
                        "request_id": REQUEST_ID,
                        "path": f"requests/{REQUEST_ID}.json",
                        "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    truths = root / "foldbench/ground_truths"
    truths.mkdir()
    shutil.copyfile(truth_source / f"{REQUEST_ID}.cif", truths / f"{REQUEST_ID}.cif")
    manifest = root / "manifests/foldingtrain-100k-v1/foldingtrain_100k.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"smoke_fixture":true}\n')
    return root


def _write_payload(root: Path, truth_source: Path) -> Path:
    root.mkdir()
    shutil.copyfile(truth_source / f"{REQUEST_ID}.cif", root / "fixture.cif")
    infer = root / "infer.sh"
    infer.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nexec python "$(dirname "$0")/infer.py" "$@"\n'
    )
    infer.chmod(infer.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (root / "infer.py").write_text(
        textwrap.dedent(
            """
            import csv, json, shutil, sys
            from pathlib import Path

            inputs, output = Path(sys.argv[1]), Path(sys.argv[2])
            output.mkdir(parents=True, exist_ok=True)
            fields = ["pdb_id", "seed", "sample", "ranking_score", "prediction_path"]
            rows = []
            for line in (inputs / "requests.jsonl").read_text().splitlines():
                request = json.loads(line)
                for seed in request["run"]["seeds"]:
                    for sample in range(request["run"]["num_samples"]):
                        name = f"{request['request_id']}-{seed}-{sample}.cif"
                        shutil.copyfile(Path(__file__).parent / "fixture.cif", output / name)
                        rows.append(dict(pdb_id=request["request_id"], seed=seed, sample=sample,
                                         ranking_score=0, prediction_path=name))
            with (output / "prediction_reference.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            """
        ).lstrip()
    )
    return root


def _wait_for_snapshot(run_dir: Path, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = run_status(run_dir)
        latest = last.get("latest_snapshot")
        if last.get("status") == "running" and latest:
            submission = Path(latest["submission_path"])
            if (submission / "runner_smoke.txt").is_file():
                return last
        if last.get("status") == "failed":
            break
        time.sleep(1)
    raise AssertionError(f"No live snapshot became ready: {last}")


def test_continuous_foldbench_session_and_independent_verifier(tmp_path: Path) -> None:
    task = Path(os.environ["FOLDBENCH_TASK_DIR"]).resolve()
    suite_source = Path(os.environ["FOLDBENCH_SMOKE_SUITE_DIR"]).resolve()
    truth_source = Path(os.environ["FOLDBENCH_SMOKE_GROUND_TRUTH_DIR"]).resolve()
    gpus = int(os.environ.get("FOLDBENCH_SMOKE_GPUS", "0"))
    data = _write_smoke_data(tmp_path / "data", suite_source, truth_source)
    payload = _write_payload(tmp_path / "payload", truth_source)
    run_dir = tmp_path / "run"

    start_run(
        task_dir=task,
        run_dir=run_dir,
        data_dir=data,
        agent="agent_runner.agents:ContinuousPublisherAgent",
        model=None,
        agent_kwargs={"payload_dir": str(payload), "expected_gpus": gpus},
        network="no-network",
        allowed_hosts=[],
        gpus=str(gpus),
        snapshot_interval=1,
        development_image=os.environ.get(
            "FOLDBENCH_DEV_IMAGE", "ai-tasks-foldbench:dev-v0.1"
        ),
        verifier_image=os.environ.get(
            "FOLDBENCH_VERIFIER_IMAGE", "ai-tasks-foldbench:verifier-v0.1"
        ),
        auth_file=None,
    )
    try:
        state = _wait_for_snapshot(run_dir)
        assert state["process_alive"] is True
        latest = state["latest_snapshot"]
        assert (
            Path(latest["submission_path"]) / "gpu_count.txt"
        ).read_text().strip() == str(gpus)
    finally:
        stop_run(run_dir, timeout=90, force=True)

    record = evaluate_snapshot(run_dir, snapshot="latest", data_dir=data, gpus=0)
    assert record["status"] == "completed"
    assert record["rewards"]["monomer_protein"] == pytest.approx(1.0, abs=1e-5)
    trial_dir = Path(record["trial_dir"])
    assert (trial_dir / "verifier/reward.json").is_file()
    assert (trial_dir / "verifier/report.json").is_file()
