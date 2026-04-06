from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYENV_PYTHON_ROOT = Path.home() / ".pyenv" / "versions"
DEFAULT_PYTHON_VERSION = "3.11.11"


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    python_version: str
    venv_path: Path
    port: int
    health_url: str
    install_commands: tuple[tuple[str, ...], ...]
    start_command: tuple[str, ...]
    cwd: Path = REPO_ROOT


RUNTIME_CONFIGS = {
    "modelbench": RuntimeConfig(
        name="modelbench",
        python_version=DEFAULT_PYTHON_VERSION,
        venv_path=REPO_ROOT / ".venvs" / "modelbench",
        port=8000,
        health_url="http://127.0.0.1:8000/api/health",
        install_commands=(
            ("pip", "install", "--upgrade", "pip"),
            ("pip", "install", "-r", "modelbench/requirements.txt"),
        ),
        start_command=(
            "python",
            "-m",
            "uvicorn",
            "modelbench.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ),
    ),
    "ssrnet": RuntimeConfig(
        name="ssrnet",
        python_version=DEFAULT_PYTHON_VERSION,
        venv_path=REPO_ROOT / ".venvs" / "ssrnet",
        port=8101,
        health_url="http://127.0.0.1:8101/health",
        install_commands=(
            ("pip", "install", "--upgrade", "pip"),
            ("pip", "install", "-r", "services/ssrnet_service/requirements.txt"),
        ),
        start_command=(
            "python",
            "-m",
            "uvicorn",
            "services.ssrnet_service.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8101",
        ),
    ),
    "deepface": RuntimeConfig(
        name="deepface",
        python_version=DEFAULT_PYTHON_VERSION,
        venv_path=REPO_ROOT / ".venvs" / "deepface",
        port=8102,
        health_url="http://127.0.0.1:8102/health",
        install_commands=(
            ("pip", "install", "--upgrade", "pip"),
            ("pip", "install", "-r", "services/deepface_service/requirements.txt"),
            ("pip", "install", "--no-deps", "-e", "./deepface"),
        ),
        start_command=(
            "python",
            "-m",
            "uvicorn",
            "services.deepface_service.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8102",
        ),
    ),
}


def python_binary_for(version: str) -> Path:
    return PYENV_PYTHON_ROOT / version / "bin" / "python"
