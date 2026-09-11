"""Shared setup for the shell and Windows launchers."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
UV = os.environ.get("MODELSCOUT_UV") or shutil.which("uv")


def run(command):
    subprocess.run(command, check=True, stdout=sys.stderr)


def ready():
    try:
        return subprocess.run(
            [str(PYTHON), "-c", "import sys; sys.exit(sys.version_info < (3, 11))"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def main():
    os.chdir(ROOT)
    if sys.version_info < (3, 11):
        raise RuntimeError("ModelScout needs Python 3.11 or newer.")

    if not ready():
        if VENV.exists():
            # Keep a broken environment so local files are never discarded.
            backup = ROOT / ".local" / f"venv-{time.time_ns()}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            VENV.rename(backup)
            print(f"Kept the old environment at {backup}", file=sys.stderr)
        print("Creating ModelScout's Python environment...", file=sys.stderr)
        if UV:
            run([UV, "venv", str(VENV), "--python", sys.executable])
        else:
            run([sys.executable, "-m", "venv", str(VENV)])

    config = ROOT / "src" / "pyproject.toml"
    fingerprint = hashlib.sha256(
        config.read_bytes() + Path(__file__).read_bytes() + str(ROOT).encode()
    ).hexdigest()
    marker = VENV / ".modelscout-installed"
    installed = marker.exists() and marker.read_text() == fingerprint
    # Package metadata can survive missing files. An install alone won't repair those.
    imports_ok = subprocess.run(
        [str(PYTHON), "-c",
         "import fastapi, uvicorn, pydantic, httpx, rich, typer, psutil, jinja2; "
         "from modelscout.cli.main import app; from modelscout.web.app import app"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    installed = installed and imports_ok
    if installed:
        check = [UV, "pip", "check", "--python", str(PYTHON)] if UV else [str(PYTHON), "-m", "pip", "check"]
        installed = subprocess.run(check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not installed:
        print("Installing ModelScout and its Python dependencies...", file=sys.stderr)
        if UV:
            repair = ["--reinstall"] if not imports_ok else []
            run([UV, "pip", "install", "--python", str(PYTHON), "--quiet", *repair, "-e", str(ROOT / "src")])
        else:
            run([str(PYTHON), "-m", "ensurepip", "--upgrade"])
            repair = ["--force-reinstall"] if not imports_ok else []
            run([str(PYTHON), "-m", "pip", "install", "--quiet", *repair, "-e", str(ROOT / "src")])
        marker.write_text(fingerprint)

    args = [str(PYTHON), "-m", "modelscout", *sys.argv[1:]]
    if os.name == "nt":
        return subprocess.call(args)
    os.execv(str(PYTHON), args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ModelScout setup failed: {exc}", file=sys.stderr)
        print("Check your connection and Python's venv support, then run the launcher again.", file=sys.stderr)
        sys.exit(1)
