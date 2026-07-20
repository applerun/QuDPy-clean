"""Project-wide startup hook for Windows Conda DLL discovery."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_DIRECTORY_HANDLES: list[object] = []


def _bootstrap_conda_dll_paths() -> None:
    if os.name != "nt":
        return

    env_prefix = _find_conda_prefix()
    if env_prefix is None:
        return

    candidate_dirs = [
        env_prefix,
        env_prefix / "Library" / "mingw-w64" / "bin",
        env_prefix / "Library" / "usr" / "bin",
        env_prefix / "Library" / "bin",
        env_prefix / "Scripts",
        env_prefix / "bin",
    ]
    existing_dirs = [path for path in candidate_dirs if path.is_dir()]

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        for path in existing_dirs:
            try:
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(path)))
            except OSError:
                pass

    current_path = os.environ.get("PATH", "")
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    normalized_parts = {_normalize_path(part) for part in path_parts}
    missing_parts = [
        str(path)
        for path in existing_dirs
        if _normalize_path(str(path)) not in normalized_parts
    ]
    if missing_parts:
        os.environ["PATH"] = os.pathsep.join(missing_parts + path_parts)


def _find_conda_prefix() -> Path | None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        if _looks_like_conda_env(prefix):
            return prefix

    executable_dir = Path(sys.executable).resolve().parent
    if _looks_like_conda_env(executable_dir):
        return executable_dir
    return None


def _looks_like_conda_env(path: Path) -> bool:
    return (path / "conda-meta").is_dir() and (path / "Library" / "bin").is_dir()


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


_bootstrap_conda_dll_paths()
