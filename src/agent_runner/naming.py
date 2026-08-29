from __future__ import annotations

import hashlib
from pathlib import Path
import re


def unique_trial_name(role: str, owner: Path) -> str:
    """Return a stable Harbor trial name unique to one runner-owned directory.

    Harbor uses the trial name as the Docker Compose project name.  A constant
    name therefore lets one run recreate or stop another run's containers.
    """
    prefix = re.sub(r"[^a-z0-9_-]", "-", role.lower()).strip("-_")
    if not prefix:
        raise ValueError("Trial role must contain an alphanumeric character")
    identity = str(owner.expanduser().resolve()).encode()
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return f"{prefix[:32]}-{digest}"
