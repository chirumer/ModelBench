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

from services.runtime import RUNTIME_CONFIGS


STATE_DIR = REPO_ROOT / ".run"
PID_FILE = STATE_DIR / "modelbench-stack.pid"
LOG_FILE = STATE_DIR / "modelbench-stack.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start, stop, restart, or inspect the local ModelBench stack.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the full local stack in the background.")
    start_parser.add_argument(
        "--install",
        action="store_true",
        help="Run dependency installation before starting. Default is to reuse existing environments.",
    )
    start_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the stack to report healthy status.",
    )

    subparsers.add_parser("stop", help="Stop the background stack.")

    restart_parser = subparsers.add_parser("restart", help="Restart the background stack.")
    restart_parser.add_argument(
        "--install",
        action="store_true",
        help="Run dependency installation before restarting.",
    )
    restart_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the stack to report healthy status.",
    )

    subparsers.add_parser("status", help="Show whether the background stack is running.")
    return parser.parse_args()


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        return None


def write_pid(pid: int) -> None:
    ensure_state_dir()
    PID_FILE.write_text(f"{pid}\n")


def clear_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


def is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_health(timeout_s: float) -> None:
    health_url = RUNTIME_CONFIGS["modelbench"].health_url
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {health_url}")


def start_stack(*, install: bool, timeout_s: float) -> int:
    pid = read_pid()
    if is_running(pid):
        print(f"ModelBench stack is already running with pid {pid}.")
        print(f"Log file: {LOG_FILE}")
        return 0

    clear_pid()
    ensure_state_dir()

    command = [sys.executable, str(REPO_ROOT / "scripts" / "dev.py")]
    if not install:
        command.append("--skip-install")

    with LOG_FILE.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    write_pid(process.pid)

    try:
        deadline = time.time() + min(timeout_s, 10.0)
        while time.time() < deadline:
            code = process.poll()
            if code is not None:
                clear_pid()
                raise RuntimeError(f"Launcher exited early with code {code}. Check {LOG_FILE}.")
            time.sleep(0.5)

        wait_for_health(timeout_s)
    except Exception:
        stop_stack(silent=True)
        raise

    print(f"ModelBench stack started with pid {process.pid}.")
    print("App: http://127.0.0.1:8000")
    print(f"Log file: {LOG_FILE}")
    return 0


def stop_stack(*, silent: bool = False) -> int:
    pid = read_pid()
    if not is_running(pid):
        clear_pid()
        if not silent:
            print("ModelBench stack is not running.")
        return 0

    assert pid is not None
    os.kill(pid, signal.SIGTERM)

    deadline = time.time() + 20.0
    while time.time() < deadline:
        if not is_running(pid):
            clear_pid()
            if not silent:
                print(f"Stopped ModelBench stack pid {pid}.")
            return 0
        time.sleep(0.5)

    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass
    clear_pid()
    if not silent:
        print(f"Force-stopped ModelBench stack pid {pid}.")
    return 0


def status_stack() -> int:
    pid = read_pid()
    if not is_running(pid):
        clear_pid()
        print("ModelBench stack is stopped.")
        return 1

    health = "unreachable"
    try:
        with urllib.request.urlopen(RUNTIME_CONFIGS["modelbench"].health_url, timeout=2.0) as response:
            if 200 <= response.status < 300:
                health = "healthy"
    except (urllib.error.URLError, TimeoutError):
        health = "starting or unhealthy"

    print(f"ModelBench stack is running with pid {pid} ({health}).")
    print("App: http://127.0.0.1:8000")
    print(f"Log file: {LOG_FILE}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "start":
        return start_stack(install=args.install, timeout_s=args.timeout)
    if args.command == "stop":
        return stop_stack()
    if args.command == "restart":
        stop_stack(silent=True)
        return start_stack(install=args.install, timeout_s=args.timeout)
    if args.command == "status":
        return status_stack()
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
