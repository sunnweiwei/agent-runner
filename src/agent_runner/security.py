from __future__ import annotations

from pathlib import Path
import stat


def validate_auth_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Authentication file does not exist: {path}")
    permissions = stat.S_IMODE(path.stat().st_mode)
    if permissions & 0o077:
        raise PermissionError(
            f"Authentication file must not be group/world accessible: {path} "
            f"(mode {permissions:o}; run chmod 600 {path})"
        )
    return path
