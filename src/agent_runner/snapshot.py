from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import stat
import time
from uuid import uuid4

from .io import read_json, write_json_atomic


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str
    mode: int
    size: int | None = None
    sha256: str | None = None
    target: str | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[Entry, ...]:
    """Hash a tree without following symlinks; reject escaping symlinks."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Submission directory does not exist: {root}")
    entries: list[Entry] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if path.is_symlink():
            target = os.readlink(path)
            resolved = path.resolve(strict=False)
            if not resolved.is_relative_to(root):
                raise ValueError(
                    f"Submission symlink escapes its root: {relative} -> {target}"
                )
            entries.append(Entry(relative, "symlink", mode, target=target))
        elif path.is_dir():
            entries.append(Entry(relative, "directory", mode))
        elif path.is_file():
            entries.append(
                Entry(
                    relative, "file", mode, size=info.st_size, sha256=_file_sha256(path)
                )
            )
        else:
            raise ValueError(f"Unsupported special file in submission: {relative}")
    return tuple(entries)


def tree_sha256(entries: tuple[Entry, ...]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        fields = (
            entry.path,
            entry.kind,
            oct(entry.mode),
            str(entry.size),
            str(entry.sha256),
            str(entry.target),
        )
        digest.update("\0".join(fields).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def latest_snapshot(snapshots_dir: Path) -> dict | None:
    pointer = snapshots_dir / "latest.json"
    return read_json(pointer) if pointer.is_file() else None


def create_snapshot(
    source: Path,
    snapshots_dir: Path,
    *,
    settle_seconds: float = 0.2,
    attempts: int = 4,
) -> dict | None:
    """Copy one stable source state into an immutable, content-addressed directory."""
    source = source.resolve()
    snapshots_dir = snapshots_dir.resolve()
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        before = inventory(source)
        if not before:
            return None
        time.sleep(settle_seconds)
        if inventory(source) != before:
            continue

        digest = tree_sha256(before)
        previous = latest_snapshot(snapshots_dir)
        if previous and previous.get("tree_sha256") == digest:
            return previous

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        snapshot_id = f"{stamp}-{digest[:12]}"
        final = snapshots_dir / snapshot_id
        temporary = snapshots_dir / f".tmp-{uuid4().hex}"
        try:
            shutil.copytree(source, temporary / "submission", symlinks=True)
            copied = inventory(temporary / "submission")
            after = inventory(source)
            if before != copied or before != after:
                shutil.rmtree(temporary)
                continue
            manifest = {
                "schema_version": "agent-runner-snapshot/v1",
                "snapshot_id": snapshot_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tree_sha256": digest,
                "submission": "submission",
                "files": [asdict(entry) for entry in before],
            }
            write_json_atomic(temporary / "manifest.json", manifest)
            os.replace(temporary, final)
            pointer = {
                "schema_version": "agent-runner-latest/v1",
                "snapshot_id": snapshot_id,
                "tree_sha256": digest,
                "path": str(final),
                "submission_path": str(final / "submission"),
                "created_at": manifest["created_at"],
            }
            write_json_atomic(snapshots_dir / "latest.json", pointer)
            return pointer
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return None


def verify_snapshot(snapshot_dir: Path) -> dict:
    manifest = read_json(snapshot_dir / "manifest.json")
    entries = inventory(snapshot_dir / "submission")
    actual = tree_sha256(entries)
    if actual != manifest.get("tree_sha256"):
        raise ValueError(
            f"Snapshot checksum differs: expected {manifest.get('tree_sha256')}, got {actual}"
        )
    return manifest
