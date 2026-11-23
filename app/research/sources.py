"""Pinned external sources, kept outside application imports and dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys

from app.utils.exceptions import GeneratorError

ROOT = Path(__file__).resolve().parents[2]
SOURCES = json.loads(Path(__file__).with_name("sources.json").read_text(encoding="utf-8"))


def worker_python() -> str:
    overlay = ROOT / ".research" / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return os.environ.get("RESEARCH_PYTHON", str(overlay) if overlay.exists() else sys.executable)


def source_path(name: str, verify: bool = False) -> Path:
    spec = SOURCES[name]
    path = Path(os.environ.get(spec["env"], str(ROOT / ".research" / "vendor" / name))).resolve()
    if not (path / ".git").exists():
        raise GeneratorError(f"{name} source is missing. Run python scripts/setup_research.py {name}.")
    if verify:
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != spec["commit"]:
            raise GeneratorError(f"{name} requires pinned source {spec['commit']}; found {actual}.")
        modified = subprocess.check_output(["git", "-C", str(path), "diff", "--name-only", "HEAD"], text=True)
        if modified.strip():
            raise GeneratorError(f"{name} source has unrecorded modifications; use a clean pinned checkout.")
    return path


def source_status(name: str) -> str | None:
    try:
        source_path(name)
        return None
    except GeneratorError as exc:
        return str(exc)
