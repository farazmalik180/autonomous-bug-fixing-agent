"""Tests for DockerSandbox.

Pure tests (fake client) always run. Tests marked ``docker`` execute real
containers and are skipped when no daemon is reachable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import docker
import pytest
from docker.errors import ImageNotFound, NotFound

from sandbox.docker_runner import (
    DockerSandbox,
    SandboxConfig,
    SandboxError,
    SandboxResult,
    tail_truncate,
)

# --------------------------------------------------------------------------- pure


def test_tail_truncate_keeps_short_output():
    assert tail_truncate(b"hello", 100) == ("hello", False)


def test_tail_truncate_keeps_the_tail_and_marks_it():
    text, truncated = tail_truncate(b"A" * 100 + b"FAILED", 10)
    assert truncated
    assert text.endswith("AAAAFAILED")
    assert text.startswith("[... truncated 96 bytes ...]")


def test_tail_truncate_tolerates_split_multibyte_char():
    text, truncated = tail_truncate("é".encode() * 10, 5)  # cuts mid-character
    assert truncated and isinstance(text, str)


def test_result_passed_requires_clean_exit_and_no_timeout():
    assert SandboxResult(exit_code=0, stdout="", stderr="").passed
    assert not SandboxResult(exit_code=1, stdout="", stderr="").passed
    assert not SandboxResult(exit_code=0, stdout="", stderr="", timed_out=True).passed


def test_run_before_start_raises():
    with pytest.raises(SandboxError, match="not started"):
        DockerSandbox(client=MagicMock()).run("true")


def test_start_rejects_missing_repo_dir(tmp_path: Path):
    with pytest.raises(SandboxError, match="not a directory"):
        DockerSandbox(client=MagicMock()).start(tmp_path / "nope")


def test_start_reports_missing_image_with_build_hint(tmp_path: Path):
    client = MagicMock()
    client.images.get.side_effect = ImageNotFound("missing")
    with pytest.raises(SandboxError, match="docker build"):
        DockerSandbox(client=client).start(tmp_path)
    client.containers.run.assert_not_called()


def test_start_applies_isolation_settings(tmp_path: Path):
    client = MagicMock()
    DockerSandbox(client=client).start(tmp_path)
    kwargs = client.containers.run.call_args.kwargs
    assert kwargs["network_disabled"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges"]
    assert kwargs["user"] == "sandbox"
    assert kwargs["pids_limit"] == 256 and kwargs["mem_limit"] == "512m"


def test_stop_is_idempotent_and_swallows_missing_container(tmp_path: Path):
    client = MagicMock()
    client.containers.run.return_value.remove.side_effect = NotFound("gone")
    sb = DockerSandbox(client=client)
    sb.start(tmp_path)
    sb.stop()
    sb.stop()  # second call is a no-op
    assert sb.container_id is None


@pytest.mark.parametrize("bad", ["../escape.txt", "/etc/passwd", "a/../../b"])
def test_paths_cannot_escape_workdir(bad: str):
    sb = DockerSandbox(client=MagicMock())
    sb._container = MagicMock()
    with pytest.raises(ValueError, match="escapes"):
        sb.write_file(bad, "x")
    with pytest.raises(ValueError, match="escapes"):
        sb.read_file(bad)


def test_run_tests_rejects_empty_command():
    sb = DockerSandbox(client=MagicMock())
    sb._container = MagicMock()
    with pytest.raises(ValueError):
        sb.run_tests("   ")


def test_is_available_false_when_daemon_down(monkeypatch: pytest.MonkeyPatch):
    def boom(*_a, **_k):
        raise docker.errors.DockerException("down")

    monkeypatch.setattr(docker, "from_env", boom)
    assert DockerSandbox.is_available() is False


# ------------------------------------------------------------------------ docker

@pytest.mark.docker
class TestContainerExecution:
    def test_captures_stdout_and_exit_code(self, sandbox: DockerSandbox):
        r = sandbox.run("echo hello")
        assert r.exit_code == 0 and r.passed
        assert r.stdout.strip() == "hello"
        assert r.stderr == "" and not r.timed_out and not r.truncated

    def test_captures_stderr_and_nonzero_exit(self, sandbox: DockerSandbox):
        r = sandbox.run("echo oops >&2; exit 3")
        assert r.exit_code == 3 and not r.passed
        assert r.stderr.strip() == "oops"

    def test_runs_as_non_root_in_workdir(self, sandbox: DockerSandbox):
        r = sandbox.run("id -u; pwd")
        uid, cwd = r.stdout.split()
        assert uid != "0"
        assert cwd == "/workspace"

    def test_python_and_pytest_available(self, sandbox: DockerSandbox):
        r = sandbox.run("python --version && pytest --version")
        assert r.passed, r.stderr

    def test_timeout_is_enforced_inside_container(self, sandbox: DockerSandbox):
        r = sandbox.run("sleep 30", timeout_s=2)
        assert r.timed_out and not r.passed
        assert r.duration_s < 10

    def test_timeout_kills_processes_ignoring_sigterm(self, sandbox: DockerSandbox):
        r = sandbox.run("trap '' TERM; sleep 30", timeout_s=2)
        assert r.timed_out and r.duration_s < 15

    def test_large_output_is_tail_truncated(self, repo_dir: Path, sandbox_image: str):
        cfg = SandboxConfig(image=sandbox_image, max_output_bytes=200)
        with DockerSandbox(cfg) as sb:
            sb.start(repo_dir)
            r = sb.run("seq 1 5000; echo END")
        assert r.truncated
        assert r.stdout.rstrip().endswith("END")
        assert len(r.stdout) < 400

    def test_network_is_disabled_by_default(self, sandbox: DockerSandbox):
        r = sandbox.run(
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\""
        )
        assert not r.passed

    def test_cannot_gain_root(self, sandbox: DockerSandbox):
        r = sandbox.run("touch /etc/pwned")
        assert not r.passed


@pytest.mark.docker
class TestFilesAndMounts:
    def test_write_then_read_roundtrip_with_nested_dirs_and_unicode(
        self, sandbox: DockerSandbox
    ):
        sandbox.write_file("pkg/sub/mod.py", "x = 'héllo ✓'\n")
        assert sandbox.read_file("pkg/sub/mod.py") == "x = 'héllo ✓'\n"
        assert sandbox.run("python pkg/sub/mod.py").passed

    def test_written_file_is_visible_on_host_mount(
        self, sandbox: DockerSandbox, repo_dir: Path
    ):
        sandbox.write_file("note.txt", "from agent")
        assert (repo_dir / "note.txt").read_text() == "from agent"

    def test_host_file_is_visible_in_container(self, sandbox: DockerSandbox, repo_dir: Path):
        (repo_dir / "host.txt").write_text("from host")
        assert sandbox.run("cat host.txt").stdout == "from host"

    def test_container_can_write_to_workdir(self, sandbox: DockerSandbox, repo_dir: Path):
        assert sandbox.run("echo data > out.txt").passed
        assert (repo_dir / "out.txt").read_text().strip() == "data"

    def test_state_persists_across_runs(self, sandbox: DockerSandbox):
        sandbox.run("echo one > a.txt")
        assert sandbox.run("cat a.txt").stdout.strip() == "one"

    def test_read_missing_file_raises(self, sandbox: DockerSandbox):
        with pytest.raises(FileNotFoundError):
            sandbox.read_file("does/not/exist.txt")


@pytest.mark.docker
class TestLifecycle:
    def test_stop_removes_container(self, sandbox_image: str, repo_dir: Path):
        sb = DockerSandbox(SandboxConfig(image=sandbox_image))
        sb.start(repo_dir)
        cid = sb.container_id
        sb.stop()
        with pytest.raises(NotFound):
            docker.from_env().containers.get(cid)
        sb.stop()  # idempotent

    def test_context_manager_cleans_up_on_exception(self, sandbox_image: str, repo_dir: Path):
        cid = None
        with pytest.raises(RuntimeError):
            with DockerSandbox(SandboxConfig(image=sandbox_image)) as sb:
                sb.start(repo_dir)
                cid = sb.container_id
                raise RuntimeError("boom")
        with pytest.raises(NotFound):
            docker.from_env().containers.get(cid)

    def test_cannot_start_twice(self, sandbox: DockerSandbox, repo_dir: Path):
        with pytest.raises(SandboxError, match="already started"):
            sandbox.start(repo_dir)

    def test_missing_image_raises_sandbox_error(self, repo_dir: Path):
        sb = DockerSandbox(SandboxConfig(image="bugfix-agent-nonexistent:0"))
        with pytest.raises(SandboxError, match="not found"):
            sb.start(repo_dir)


# ------------------------------------------------- mock bug reproduction (e2e-ish)

BUGGY_MODULE = "def average(xs):\n    return sum(xs) / (len(xs) + 1)  # off-by-one bug\n"
REPRO_TEST = (
    "from calc import average\n\n"
    "def test_average_of_three():\n"
    "    assert average([2, 4, 6]) == 4\n"
)


@pytest.mark.docker
class TestMockBugReproduction:
    """Mirrors the agent loop: repro test fails, patch applied, repro test passes."""

    def test_repro_fails_then_passes_after_fix(self, sandbox: DockerSandbox):
        sandbox.write_file("calc.py", BUGGY_MODULE)
        sandbox.write_file("test_repro.py", REPRO_TEST)

        before = sandbox.run_tests("pytest -q -p no:cacheprovider")
        assert before.exit_code == 1 and not before.passed
        assert "assert 3.0 == 4" in before.stdout

        sandbox.write_file("calc.py", "def average(xs):\n    return sum(xs) / len(xs)\n")
        after = sandbox.run_tests("pytest -q -p no:cacheprovider")
        assert after.passed, after.stdout
        assert "1 passed" in after.stdout
