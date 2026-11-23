"""Provision pinned research sources and their isolated Python overlay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.research.sources import ROOT, SOURCES, source_path, worker_python


def environment_python() -> Path:
    return ROOT / ".research" / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_environment(install: bool) -> Path:
    interpreter = environment_python()
    if not interpreter.exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                        str(ROOT / ".research" / "env")], check=True)
    if install:
        subprocess.run([str(interpreter), "-m", "pip", "install", "-r",
                        str(ROOT / "requirements-research.txt")], check=True)
    return interpreter


def ensure_source(name: str) -> Path:
    spec = SOURCES[name]
    destination = ROOT / ".research" / "vendor" / name
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", spec["url"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", spec["commit"]], check=True)
    return source_path(name, verify=True)


def dependency_check(interpreter: str) -> dict:
    code = ("import json,importlib.util; print(json.dumps({"
            "'torch':importlib.util.find_spec('torch') is not None,"
            "'einops':importlib.util.find_spec('einops') is not None}))")
    result = subprocess.run([interpreter, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Set up isolated advanced-model workers.")
    parser.add_argument("model", nargs="?", choices=[*SOURCES, "timegan_torch", "all"], default="diffusion_ts")
    parser.add_argument("--create-env", action="store_true",
                        help="Create .research/env and install the minimal overlay.")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check without cloning or installing.")
    args = parser.parse_args()
    names = [*SOURCES, "timegan_torch"] if args.model == "all" else [args.model]
    if args.check:
        statuses = {}
        for name in names:
            if name == "timegan_torch":
                statuses[name] = {"ready": True, "path": "local PyTorch implementation"}
                continue
            try:
                statuses[name] = {"ready": True, "path": str(source_path(name, verify=True))}
            except Exception as exc:
                statuses[name] = {"ready": False, "reason": str(exc)}
        interpreter = worker_python()
        print(json.dumps({"worker_python": interpreter,
                          "dependencies": dependency_check(interpreter),
                          "sources": statuses}, indent=2))
        return

    interpreter = ensure_environment(install=not args.skip_install) if args.create_env else Path(worker_python())
    paths = {name: str(ensure_source(name)) for name in names if name in SOURCES}
    if "timegan_torch" in names:
        paths["timegan_torch"] = "local PyTorch implementation"
    dependencies = dependency_check(str(interpreter))
    if not dependencies.get("torch"):
        raise SystemExit("Research worker cannot find PyTorch. Install PyTorch in the base Python and rerun with --create-env.")
    if "diffusion_ts" in names and not dependencies.get("einops"):
        raise SystemExit("Diffusion-TS requires einops. Rerun with --create-env without --skip-install.")
    print(json.dumps({"worker_python": str(interpreter), "dependencies": dependencies,
                      "sources": paths, "ready": True}, indent=2))


if __name__ == "__main__":
    main()
