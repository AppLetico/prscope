"""
Web server bootstrap for prscope UI.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

# Load .env so API keys are available when server is started directly
# (e.g. uvicorn prscope.web.server:create_server_app). CLI also loads .env
# but the server subprocess or direct invocations do not import cli.
load_dotenv()
load_dotenv(Path.cwd() / ".env")
# Package root (e.g. prscope repo root when cwd is elsewhere)
_load_root = Path(__file__).resolve().parent.parent.parent
if _load_root != Path.cwd():
    load_dotenv(_load_root / ".env")

import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .api import create_app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward Loguru sinks."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """Configure loguru to intercept uvicorn/fastapi logs and write to file/console."""
    log_file = Path.home() / ".prscope" / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # Add console handler
    logger.add(sys.stderr, level="INFO", colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # Add file handler
    logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="1 week", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Intercept uvicorn loggers
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


def is_port_open(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def create_server_app() -> object:
    setup_logging()
    logger.info("Starting prscope API server...")
    
    app = create_app()
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    if static_dir.exists():
        # Serve static assets and always fall back to index.html for SPA routes.
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(_request: Request, full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            if not index_path.exists():
                raise HTTPException(status_code=404, detail="UI assets not found")

            requested = (static_dir / full_path).resolve()
            try:
                requested.relative_to(static_dir.resolve())
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="Not Found") from exc

            # Serve built files directly when they exist, otherwise serve SPA entrypoint.
            if requested.exists() and requested.is_file():
                return FileResponse(requested)
            return FileResponse(index_path)
    return app


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    uvicorn.run("prscope.web.server:create_server_app", host=host, port=port, log_level="info", factory=True)


def ensure_server_running(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> tuple[bool, str]:
    """
    Ensure web server is running.

    Returns (already_running, url).
    """
    url = f"http://{host}:{port}"
    if is_port_open(host, port):
        if open_browser:
            webbrowser.open(url)
        return True, url

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "prscope.web.server:create_server_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
    ]
    
    # We still redirect stdout/stderr to the log file so that any low-level crashes
    # (before loguru initializes) are captured, but loguru will handle the rest.
    log_file = Path.home() / ".prscope" / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "a") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    deadline = time.time() + 8
    while time.time() < deadline:
        if is_port_open(host, port):
            if open_browser:
                webbrowser.open(url)
            return False, url
        time.sleep(0.2)
    raise RuntimeError("Failed to start prscope web server.")
