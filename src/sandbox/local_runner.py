"""Subprocess-based sandbox backend (weaker isolation than Docker).

Intended for hosts without a Docker daemon (e.g. Streamlit Community Cloud) and for fast
tests. It does **not** block the network, jail the filesystem, or cap memory. What it does:

* runs commands in a throw-away working copy, never the user's original repo;
* scrubs secrets (API keys, tokens, ...) from the child environment so generated code
  cannot read them;
* enforces timeouts and kills the *whole process tree* on expiry;
* tail-truncates output exactly like ``DockerSandbox``.

See SPEC.md section 10.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType

from sandbox.base import SandboxError, SandboxResult, tail_truncate

logger = logging.getLogger(__name__)

_TIMEOUT_EXIT = 124  # same convention as coreutils `timeout`

#: Environment variable names matching this are never passed to child processes.
SECRET_ENV_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|DATABASE_URL|GROQ|GITHUB|^GH_|AWS_|AZURE_|"
    r"OPENAI|ANTHROPIC|LANGSMITH|LANGCHAIN)",
    re.IGNORECASE,
)


def scrubbed_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment without secrets and with sane Python defaults."""
    env = {k: v for k, v in (base if base is not None else os.environ).items()
           if not SECRET_ENV_PATTERN.search(k)}  # fmt: skip
    env.pop("PYTHONPATH", None)  # don't leak the agent's own source tree into the target
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Make `python` / `pytest` resolve to the interpreter running the agent.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` and all of its descendants."""
    try:
        if os.name == "nt":
            subprocess.run(  # noqa: S603
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],  # noqa: S607
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


class LocalSandbox:
    """Runs commands as subprocesses inside a working-copy directory."""

    name = "local"
    isolated = False

    def __init__(self, default_timeout_s: int = 60, max_output_bytes: int = 64_000) -> None:
        self.default_timeout_s = default_timeout_s
        self.max_output_bytes = max_output_bytes
        self._root: Path | None = None

    # ------------------------------------------------------------------ setup

    @staticmethod
    def is_available() -> bool:
        return True

    @property
    def root(self) -> Path:
        if self._root is None:
            raise SandboxError("Sandbox not started; call start() first.")
        return self._root

    def start(self, host_repo_path: Path) -> None:
        if self._root is not None:
            raise SandboxError("Sandbox already started; call stop() first.")
        path = Path(host_repo_path).resolve()
        if not path.is_dir():
            raise SandboxError(f"Repo path is not a directory: {path}")
        self._root = path
        logger.info("local sandbox started in %s (NOT isolated)", path)

    def stop(self) -> None:
        self._root = None  # nothing to clean up: the working copy is owned by the caller

    def __enter__(self) -> LocalSandbox:
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
        root = self.root
        timeout = timeout_s or self.default_timeout_s
        popen_kwargs: dict = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        start = time.perf_counter()
        try:
            proc = subprocess.Popen(  # noqa: S602 - shell is the contract (`bash -c` equivalent)
                command,
                shell=True,
                cwd=root,
                env=scrubbed_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **popen_kwargs,
            )
        except OSError as exc:
            raise SandboxError(f"Failed to launch command: {exc}") from exc

        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = b"", b""
        duration = time.perf_counter() - start

        stdout, out_trunc = tail_truncate(out or b"", self.max_output_bytes)
        stderr, err_trunc = tail_truncate(err or b"", self.max_output_bytes)
        return SandboxResult(
            exit_code=_TIMEOUT_EXIT if timed_out else (proc.returncode if proc.returncode is not None else -1),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=round(duration, 3),
            truncated=out_trunc or err_trunc,
        )

    def run_tests(self, test_command: str, timeout_s: int | None = None) -> SandboxResult:
        if not test_command.strip():
            raise ValueError("test_command must not be empty")
        return self.run(test_command, timeout_s)

    # ------------------------------------------------------------------ files

    def _resolve(self, path: str) -> Path:
        root = self.root
        full = (root / path).resolve()
        if full != root and not full.is_relative_to(root):
            raise ValueError(f"Path escapes sandbox workdir: {path!r}")
        return full

    def write_file(self, path: str, content: str) -> None:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8", newline="")

    def read_file(self, path: str) -> str:
        full = self._resolve(path)
        if not full.exists():
            raise FileNotFoundError(str(full))
        if full.is_dir():
            raise IsADirectoryError(str(full))
        return full.read_text(encoding="utf-8", errors="replace")

