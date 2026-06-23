"""Path resolution helpers for ROS share-dir / cwd-relative file inputs."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def resolve_share_file(pkg: str, rel: str | Path, override_text: str = "") -> Path:
    """Resolve a file path with the following priority:

    1. If ``override_text`` is non-empty: expand ``~`` and either return the
       absolute path or resolve it against the current working directory.
    2. Otherwise return ``<share-dir-of-pkg>/<rel>``.
    """

    if override_text:
        path = Path(override_text).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (Path.cwd() / path).resolve()
    share = Path(get_package_share_directory(pkg))
    return (share / rel).resolve()
