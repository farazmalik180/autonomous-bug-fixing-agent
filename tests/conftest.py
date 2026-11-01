"""Shared fixtures.

Tests marked ``docker`` need a running daemon; they are skipped otherwise. The
sandbox image is built once per session if missing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sandbox.docker_runner import (
    DEFAULT_IMAGE,
    DockerSandbox,
    SandboxConfig,
    build_image,
)

_DOCKER_UP = DockerSandbox.is_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _DOCKER_UP:
        return
    skip = pytest.mark.skip(reason="Docker daemon not reachable")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def sandbox_image() -> str:
    import docker
    from docker.errors import ImageNotFound

    client = docker.from_env()
    try:
        client.images.get(DEFAULT_IMAGE)
    except ImageNotFound:
        build_image(client)
    return DEFAULT_IMAGE


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    """An empty host working copy to mount into the sandbox."""
    d = tmp_path / "repo"
    d.mkdir()
    return d


@pytest.fixture
def sandbox(sandbox_image: str, repo_dir: Path) -> Iterator[DockerSandbox]:
    """A started sandbox with the network off, torn down after the test."""
    with DockerSandbox(SandboxConfig(image=sandbox_image, default_timeout_s=30)) as sb:
        sb.start(repo_dir)
        yield sb
