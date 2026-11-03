"""Docker sandbox runner.

Runs bash commands and test runners inside a long-lived, locked-down container.
One container per agent run, so edits and installed state persist across fix
iterations. See SPEC.md section 5.

Typical use::

    with DockerSandbox() as sandbox:
        sandbox.start(repo_copy_path)
        result = sandbox.run_tests("pytest -x -q")
        if not result.passed:
            ...
"""

from __future__ import annotations

import io
import logging
import os
import posixpath
import tarfile
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import docker
from docker.context import ContextAPI
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.types import Mount
from pydantic import BaseModel, Field

from sandbox.base import SandboxError, SandboxResult, tail_truncate

__all__ = [
    "DEFAULT_IMAGE",
    "DockerSandbox",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "build_image",
    "docker_client",
    "tail_truncate",
]

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "bugfix-agent-sandbox:latest"
DOCKERFILE_DIR = Path(__file__).resolve().parent

# Exit codes produced by coreutils `timeout`.
_TIMEOUT_EXIT = 124  # command timed out (SIGTERM honoured)
_KILLED_EXIT = 137  # SIGKILL (-k grace expired); 128 + 9
_KILL_GRACE_S = 2


class SandboxConfig(BaseModel):
    """Container resource limits and isolation settings."""

    image: str = DEFAULT_IMAGE
    workdir: str = "/workspace"
    mem_limit: str = "512m"
    nano_cpus: int = Field(default=1_000_000_000, gt=0)  # 1 CPU
    pids_limit: int = Field(default=256, gt=0)
    network_disabled: bool = True  # default-deny network
    default_timeout_s: int = Field(default=60, gt=0)
    max_output_bytes: int = Field(default=64_000, gt=0)  # per stream, tail-preserved
    user: str = "sandbox"  # non-root
    uid: int = 1000  # must match the uid of `user` in the image


def docker_client() -> docker.DockerClient:
    """Create a client for the *active Docker context*.

    ``docker.from_env()`` alone only knows ``DOCKER_HOST`` / the default pipe. Docker
    Desktop on Windows uses the ``desktop-linux`` context (a different pipe), so we
    resolve the current context when ``DOCKER_HOST`` is not set.
    """
    if not os.environ.get("DOCKER_HOST"):
        try:
            ctx = ContextAPI.get_current_context()
            host = getattr(ctx, "Host", None)
            if ctx is not None and host and ctx.Name != "default":
                return docker.DockerClient(base_url=host)
        except Exception:  # noqa: BLE001 - fall back to environment-based discovery
            logger.debug("could not resolve docker context", exc_info=True)
    return docker.from_env()


def build_image(
    client: docker.DockerClient | None = None,
    tag: str = DEFAULT_IMAGE,
    context_dir: Path = DOCKERFILE_DIR,
) -> None:
    """Build the sandbox image from ``src/sandbox/Dockerfile``."""
    client = client or _connect()
    try:
        client.images.build(path=str(context_dir), tag=tag, rm=True)
    except (DockerException, APIError) as exc:
        raise SandboxError(f"Failed to build sandbox image {tag!r}: {exc}") from exc


def _connect() -> docker.DockerClient:
    try:
        client = docker_client()
        client.ping()
        return client
    except DockerException as exc:
        raise SandboxError(f"Docker daemon is not reachable: {exc}") from exc


class DockerSandbox:
    """Isolated command runner backed by one Docker container."""

    name = "docker"
    isolated = True

    def __init__(
        self,
        config: SandboxConfig | None = None,
        client: docker.DockerClient | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self._client = client
        self._container: Any = None

    # ------------------------------------------------------------------ setup

    @staticmethod
    def is_available() -> bool:
        """True if the Docker daemon is reachable."""
        try:
            docker_client().ping()
            return True
        except Exception:  # noqa: BLE001 - any failure means "not available"
            return False


    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = _connect()
        return self._client

    @property
    def container_id(self) -> str | None:
        return self._container.id if self._container is not None else None

    def start(self, host_repo_path: Path) -> None:
        """Create the container and bind-mount ``host_repo_path`` at the workdir.

        ``host_repo_path`` should be a working copy (e.g. a git worktree), not the
        user's original checkout: the container has write access to it.
        """
        if self._container is not None:
            raise SandboxError("Sandbox already started; call stop() first.")
        host_path = Path(host_repo_path).resolve()
        if not host_path.is_dir():
            raise SandboxError(f"Repo path is not a directory: {host_path}")

        cfg = self.config
        try:
            self.client.images.get(cfg.image)
        except ImageNotFound as exc:
            raise SandboxError(
                f"Sandbox image {cfg.image!r} not found. Build it with "
                f"`docker build -t {cfg.image} src/sandbox` or sandbox.build_image()."
            ) from exc
        except (DockerException, APIError) as exc:
            raise SandboxError(f"Could not inspect image {cfg.image!r}: {exc}") from exc

        try:
            self._container = self.client.containers.run(
                cfg.image,
                command=["sleep", "infinity"],
                detach=True,
                working_dir=cfg.workdir,
                user=cfg.user,
                mem_limit=cfg.mem_limit,
                memswap_limit=cfg.mem_limit,  # no swap
                nano_cpus=cfg.nano_cpus,
                pids_limit=cfg.pids_limit,
                network_disabled=cfg.network_disabled,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                tmpfs={"/tmp": "size=64m,mode=1777"},
                mounts=[Mount(target=cfg.workdir, source=str(host_path), type="bind")],
                labels={"bugfix-agent": "sandbox"},
            )
        except (DockerException, APIError) as exc:
            raise SandboxError(f"Failed to start sandbox container: {exc}") from exc
        logger.info("sandbox started: %s", self._container.short_id)

    def stop(self) -> None:
        """Force-remove the container. Idempotent and never raises."""
        container, self._container = self._container, None
        if container is None:
            return
        try:
            container.remove(force=True, v=True)
        except NotFound:
            pass
        except (DockerException, APIError) as exc:
            logger.warning("failed to remove sandbox container: %s", exc)

    def __enter__(self) -> DockerSandbox:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -------------------------------------------------------------- execution

    def run(self, command: str, timeout_s: int | None = None) -> SandboxResult:
        """Run ``command`` via ``bash -c`` in the workdir, as the sandbox user.

        The timeout is enforced *inside* the container with coreutils ``timeout``
        so no orphaned processes are left behind.
        """
        container = self._require_container()
        timeout = timeout_s or self.config.default_timeout_s
        argv = [
            "timeout", "-k", str(_KILL_GRACE_S), str(timeout),
            "bash", "-c", command,
        ]  # fmt: skip

        start = time.perf_counter()
        try:
            exec_result = container.exec_run(
                argv,
                demux=True,
                workdir=self.config.workdir,
                user=self.config.user,
            )
        except (DockerException, APIError) as exc:
            raise SandboxError(f"Command execution failed: {exc}") from exc
        duration = time.perf_counter() - start

        out, err = exec_result.output or (None, None)
        stdout, out_trunc = tail_truncate(out or b"", self.config.max_output_bytes)
        stderr, err_trunc = tail_truncate(err or b"", self.config.max_output_bytes)

        code = exec_result.exit_code if exec_result.exit_code is not None else -1
        timed_out = code == _TIMEOUT_EXIT or (code == _KILLED_EXIT and duration >= timeout)
        return SandboxResult(
            exit_code=code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=round(duration, 3),
            truncated=out_trunc or err_trunc,
        )

    def run_tests(self, test_command: str, timeout_s: int | None = None) -> SandboxResult:
        """Run a test command (e.g. ``pytest -x -q``). Thin wrapper over :meth:`run`."""
        if not test_command.strip():
            raise ValueError("test_command must not be empty")
        return self.run(test_command, timeout_s)

    # ------------------------------------------------------------------ files

    def write_file(self, path: str, content: str) -> None:
        """Write a UTF-8 text file in the container (parents created).

        ``path`` is relative to the workdir or absolute within it; escaping the
        workdir raises ``ValueError``.
        """
        container = self._require_container()
        full = self._resolve(path)
        parent, name = posixpath.split(full)

        mkdir = container.exec_run(
            ["mkdir", "-p", parent], user=self.config.user, workdir=self.config.workdir
        )
        if mkdir.exit_code != 0:
            raise SandboxError(f"Could not create {parent}: {mkdir.output!r}")

        payload = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = self.config.uid
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(payload))
        buf.seek(0)
        try:
            ok = container.put_archive(parent, buf.getvalue())
        except (DockerException, APIError) as exc:
            raise SandboxError(f"Failed to write {full}: {exc}") from exc
        if not ok:
            raise SandboxError(f"Failed to write {full}")

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file from the container."""
        container = self._require_container()
        full = self._resolve(path)
        try:
            stream, _stat = container.get_archive(full)
        except NotFound as exc:
            raise FileNotFoundError(full) from exc
        except (DockerException, APIError) as exc:
            raise SandboxError(f"Failed to read {full}: {exc}") from exc

        buf = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=buf, mode="r") as tar:
            member = tar.next()
            extracted = tar.extractfile(member) if member is not None else None
            if extracted is None:
                raise IsADirectoryError(full)
            return extracted.read().decode("utf-8", errors="replace")

    # --------------------------------------------------------------- internal

    def _require_container(self) -> Any:
        if self._container is None:
            raise SandboxError("Sandbox not started; call start() first.")
        return self._container

    def _resolve(self, path: str) -> str:
        """Resolve ``path`` against the workdir, rejecting escapes."""
        workdir = posixpath.normpath(self.config.workdir)
        full = posixpath.normpath(posixpath.join(workdir, path))
        if full != workdir and not full.startswith(workdir + "/"):
            raise ValueError(f"Path escapes sandbox workdir: {path!r}")
        return full
