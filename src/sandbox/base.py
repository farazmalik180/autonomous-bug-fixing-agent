"""Backend-agnostic sandbox types: result/error models and the ``Sandbox`` protocol.

``DockerSandbox`` (full isolation) and ``LocalSandbox`` (subprocess, weaker isolation)
both implement :class:`Sandbox`; agent nodes depend only on this module.
See SPEC.md section 10.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SandboxError(Exception):
    """Infrastructure failure: daemon unreachable, image missing, container gone."""


class SandboxResult(BaseModel):
    """Outcome of one command executed in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_s: float = 0.0
    truncated: bool = False

    @property
    def passed(self) -> bool:
        """True only for a clean exit. A timeout is never a pass."""
        return self.exit_code == 0 and not self.timed_out


def tail_truncate(data: bytes, limit: int) -> tuple[str, bool]:
    """Decode ``data``, keeping the last ``limit`` bytes (failures print last).

    Returns ``(text, was_truncated)``. A marker line is prepended on truncation.
    """
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace"), False
    dropped = len(data) - limit
    tail = data[-limit:].decode("utf-8", errors="replace")
    return f"[... truncated {dropped} bytes ...]\n{tail}", True


@runtime_checkable
class Sandbox(Protocol):
    """Interface every sandbox backend implements."""

    #: Human-readable backend name, e.g. ``"docker"`` or ``"local"``.
    name: str
    #: True if the backend gives real isolation (the UI warns when it does not).
    isolated: bool

    def start(self, host_repo_path: Path) -> None: ...

    def run(self, command: str, timeout_s: int | None = None) -> SandboxResult: ...

    def run_tests(self, test_command: str, timeout_s: int | None = None) -> SandboxResult: ...

    def write_file(self, path: str, content: str) -> None: ...

    def read_file(self, path: str) -> str: ...

    def stop(self) -> None: ...

    def __enter__(self) -> Sandbox: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

