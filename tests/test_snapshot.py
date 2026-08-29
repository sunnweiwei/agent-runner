from pathlib import Path
import os

import pytest

from agent_runner.snapshot import create_snapshot, inventory, verify_snapshot


def test_snapshot_is_content_addressed_and_latest_is_atomic(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    script = live / "infer.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    snapshots = tmp_path / "snapshots"

    first = create_snapshot(live, snapshots, settle_seconds=0)
    assert first is not None
    assert verify_snapshot(Path(first["path"]))["tree_sha256"] == first["tree_sha256"]
    assert (Path(first["submission_path"]) / "infer.sh").stat().st_mode & 0o111

    duplicate = create_snapshot(live, snapshots, settle_seconds=0)
    assert duplicate == first
    script.write_text("#!/bin/sh\necho updated\n")
    second = create_snapshot(live, snapshots, settle_seconds=0)
    assert second is not None
    assert second["snapshot_id"] != first["snapshot_id"]
    assert (
        (Path(first["submission_path"]) / "infer.sh").read_text().endswith("exit 0\n")
    )


def test_inventory_rejects_escaping_symlink(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("not part of submission")
    os.symlink(outside, live / "escape")
    with pytest.raises(ValueError, match="escapes"):
        inventory(live)
