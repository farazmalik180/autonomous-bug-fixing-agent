"""Git helpers invoked ONLY after human approval (`commit_and_pr` node).

Copies approved changes back into the target repository (or creates a branch + commit
if it is a git repository and `apply_to_repo=True`). When running against a temporary
clone or non-git folder, it generates a deterministic commit SHA and patch bundle
without touching external remotes unless explicitly asked.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tools.file_tools import Workspace


class GitError(Exception):
    """Git operation failed."""


@dataclass(frozen=True)
class CommitResult:
    branch: str
    commit_sha: str
    pr_url: str | None = None
    applied_to_repo: bool = False


def is_git_repo(path: Path) -> bool:
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return res.returncode == 0 and res.stdout.strip() == "true"
    except OSError:
        return False


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if res.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout.strip()


def apply_workspace_to_repo(
    ws: Workspace,
    repo_path: Path,
    *,
    branch_name: str,
    commit_message: str,
    commit_in_git: bool = False,
) -> CommitResult:
    """Copy modified/new files from ``ws.work`` to ``repo_path`` and optionally commit."""
    changed = ws.changed_files()
    diff_text = ws.full_diff()
    synthetic_sha = hashlib.sha1(diff_text.encode("utf-8")).hexdigest()[:7]  # noqa: S324

    if not commit_in_git or not repo_path.is_dir():
        return CommitResult(branch=branch_name, commit_sha=synthetic_sha, applied_to_repo=False)

    for rel in changed:
        src = ws.work / Path(*PurePosixPath(rel).parts)
        dst = (repo_path / Path(*PurePosixPath(rel).parts)).resolve()
        if not dst.is_relative_to(repo_path.resolve()):
            raise GitError(f"Refusing to write outside repo: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    if not is_git_repo(repo_path):
        return CommitResult(branch=branch_name, commit_sha=synthetic_sha, applied_to_repo=True)

    _git(repo_path, "checkout", "-B", branch_name)
    _git(repo_path, "add", "--", *changed)
    _git(repo_path, "commit", "-m", commit_message)
    sha = _git(repo_path, "rev-parse", "--short", "HEAD")
    return CommitResult(branch=branch_name, commit_sha=sha, applied_to_repo=True)
