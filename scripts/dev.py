from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.runtime import RUNTIME_CONFIGS, python_binary_for


class ShutdownRequested(Exception):
    """Raised when the launcher receives a termination signal."""


def _raise_shutdown(signum, frame) -> None:
    raise ShutdownRequested()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start ModelBench and its local backend services.")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Reuse existing venvs without reinstalling dependencies.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create venvs and install dependencies without starting the services.",
    )
    return parser.parse_args()


def ensure_venv(name: str) -> Path:
    config = RUNTIME_CONFIGS[name]
    python_bin = python_binary_for(config.python_version)
    if not python_bin.exists():
        raise SystemExit(
            f"Required Python {config.python_version} was not found at {python_bin}. "
            "Install it with pyenv before running the launcher."
        )

    if not config.venv_path.exists():
        print(f"[setup] creating {name} venv at {config.venv_path}")
        subprocess.run(
            [str(python_bin), "-m", "venv", str(config.venv_path)],
            cwd=REPO_ROOT,
            check=True,
        )

    return config.venv_path / "bin" / "python"


def run_install(name: str, python_path: Path) -> None:
    config = RUNTIME_CONFIGS[name]
    env = os.environ.copy()
    env["PATH"] = f"{config.venv_path / 'bin'}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(config.venv_path)

    for command in config.install_commands:
        resolved = [str(config.venv_path / "bin" / command[0]), *command[1:]]
        print(f"[setup] {name}: {' '.join(command)}")
        subprocess.run(resolved, cwd=REPO_ROOT, env=env, check=True)


def wait_for_health(url: str, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)
    raise SystemExit(f"Timed out waiting for service health at {url}")


def spawn(name: str) -> subprocess.Popen:
    config = RUNTIME_CONFIGS[name]
    python_path = config.venv_path / "bin" / "python"
    env = os.environ.copy()
    env["PATH"] = f"{config.venv_path / 'bin'}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(config.venv_path)
    env["PYTHONPATH"] = str(REPO_ROOT)
    command = [str(python_path), *config.start_command[1:]]
    print(f"[start] {name}: {' '.join(config.start_command)}")
    return subprocess.Popen(command, cwd=REPO_ROOT, env=env)


def main() -> int:
    signal.signal(signal.SIGTERM, _raise_shutdown)
    signal.signal(signal.SIGINT, _raise_shutdown)
    args = parse_args()
    processes: list[subprocess.Popen] = []
    try:
        for name in ("modelbench", "ssrnet", "deepface"):
            python_path = ensure_venv(name)
            if not args.skip_install:
                run_install(name, python_path)

        if args.setup_only:
            print("[done] environments are ready")
            return 0

        for name in ("ssrnet", "deepface", "modelbench"):
            process = spawn(name)
            processes.append(process)
            wait_for_health(RUNTIME_CONFIGS[name].health_url)

        print("[ready] ModelBench is available at http://127.0.0.1:8000")
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    raise SystemExit(f"A child process exited unexpectedly with code {code}.")
            time.sleep(1.0)
    except (KeyboardInterrupt, ShutdownRequested):
        print("\n[stop] shutting down services")
        return 0
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in reversed(processes):
            if process.poll() is None:
                process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
