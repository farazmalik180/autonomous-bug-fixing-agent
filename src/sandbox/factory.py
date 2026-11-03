"""Pick a sandbox backend: Docker when available, otherwise the local subprocess runner."""

from __future__ import annotations

import logging
import os

from sandbox.base import Sandbox, SandboxError
from sandbox.docker_runner import (
    DEFAULT_IMAGE,
    DockerSandbox,
    SandboxConfig,
    build_image,
    docker_client,
)
from sandbox.local_runner import LocalSandbox

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "docker", "local")


def _ensure_docker_image(image: str = DEFAULT_IMAGE) -> None:
    from docker.errors import ImageNotFound

    client = docker_client()
    try:
        client.images.get(image)
    except ImageNotFound:
        logger.info("building sandbox image %s (first run)", image)
        build_image(client, image)


def create_sandbox(backend: str | None = None) -> Sandbox:
    """Create a sandbox from ``backend`` or ``SANDBOX_BACKEND`` (``auto|docker|local``).

    ``auto`` prefers Docker (real isolation) and falls back to ``LocalSandbox`` with a
    warning when no daemon is reachable.
    """
    choice = (backend or os.environ.get("SANDBOX_BACKEND") or "auto").strip().lower()
    if choice not in BACKENDS:
        raise ValueError(f"Unknown sandbox backend {choice!r}; expected one of {BACKENDS}")

    if choice == "local":
        return LocalSandbox()

    if DockerSandbox.is_available():
        _ensure_docker_image()
        return DockerSandbox(SandboxConfig())
    if choice == "docker":
        raise SandboxError("SANDBOX_BACKEND=docker but the Docker daemon is not reachable.")

    logger.warning("Docker not available: falling back to LocalSandbox (NOT isolated)")
    return LocalSandbox()

