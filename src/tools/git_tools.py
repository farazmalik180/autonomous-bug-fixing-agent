"""Git & GitHub PR helpers invoked ONLY after human approval (`commit_and_pr` node or UI).

Supports:
1. Copying approved changes back into a local repository and creating a branch + commit.
2. Pushing a fix branch (`agent/fix-...`) and opening a real Pull Request on GitHub via the
   GitHub REST API using a user-supplied Personal Access Token (`github_token`).
"""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from tools.file_tools import Workspace


class GitError(Exception):
    """Git or GitHub API operation failed."""


@dataclass(frozen=True)
class CommitResult:
    branch: str
    commit_sha: str
    pr_url: str | None = None
    applied_to_repo: bool = False


@dataclass(frozen=True)
class GitHubIssueData:
    owner: str
    repo: str
    number: int
    title: str
    body: str
    html_url: str
    repo_clone_url: str


def parse_github_owner_repo(repo_url: str) -> tuple[str, str] | None:
    """Extract `(owner, repo)` from a GitHub HTTPS or SSH URL."""
    m = re.search(
        r"github\.com[:/](?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git|/.*)?$",
        repo_url.strip(),
    )
    if not m:
        return None
    return m.group("owner"), m.group("repo")


def parse_github_issue_url(issue_url: str) -> tuple[str, str, int] | None:
    """Extract `(owner, repo, issue_number)` from a GitHub Issue URL."""
    m = re.search(
        r"github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/issues/(?P<num>\d+)",
        issue_url.strip(),
    )
    if not m:
        return None
    return m.group("owner"), m.group("repo"), int(m.group("num"))


def fetch_github_issue(issue_url: str, github_token: str | None = None) -> GitHubIssueData:
    """Fetch a GitHub issue's title, body, number, and repository URL via the GitHub REST API."""
    parsed = parse_github_issue_url(issue_url)
    if not parsed:
        raise GitError(
            "Invalid GitHub Issue URL. Expected format: https://github.com/owner/repo/issues/12"
        )
    owner, repo, num = parsed
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token and github_token.strip():
        headers["Authorization"] = f"Bearer {github_token.strip()}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{num}"
    with httpx.Client(headers=headers, timeout=20.0) as client:
        resp = client.get(api_url)
        if resp.status_code != 200:
            raise GitError(f"Could not fetch issue #{num} from {owner}/{repo}: {resp.text[:200]}")
        data = resp.json()

    return GitHubIssueData(
        owner=owner,
        repo=repo,
        number=num,
        title=str(data.get("title") or f"Issue #{num}"),
        body=str(data.get("body") or ""),
        html_url=str(data.get("html_url") or issue_url.strip()),
        repo_clone_url=f"https://github.com/{owner}/{repo}.git",
    )


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


def create_github_pull_request(
    ws: Workspace,
    *,
    repo_url: str,
    github_token: str,
    issue_title: str,
    issue_body: str,
    rationale: str,
    branch_prefix: str = "agent/fix",
) -> CommitResult:
    """Create a branch, commit changed files, and open a Pull Request via the GitHub REST API.

    Uses GitHub's Git Data / Contents REST endpoints over HTTPS so it works in cloud
    containers without needing SSH keys or local git credentials.
    """
    parsed = parse_github_owner_repo(repo_url)
    if not parsed:
        raise GitError(f"Cannot parse GitHub owner/repo from URL: {repo_url!r}")
    owner, repo = parsed
    token = github_token.strip()
    if not token:
        raise GitError("A GitHub Personal Access Token (with `repo` or `contents:write` + `pull_requests:write` scope) is required.")

    changed = ws.changed_files()
    if not changed:
        raise GitError("No modified files in workspace to commit.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base_api = f"https://api.github.com/repos/{owner}/{repo}"

    with httpx.Client(headers=headers, timeout=30.0) as client:
        # 1. Get default branch name & its HEAD SHA
        repo_resp = client.get(base_api)
        if repo_resp.status_code != 200:
            raise GitError(f"GitHub API error fetching {owner}/{repo}: {repo_resp.text[:200]}")
        default_branch = repo_resp.json().get("default_branch", "main")

        ref_resp = client.get(f"{base_api}/git/ref/heads/{default_branch}")
        if ref_resp.status_code != 200:
            raise GitError(f"Could not read branch {default_branch!r}: {ref_resp.text[:200]}")
        base_sha = ref_resp.json()["object"]["sha"]

        # 2. Create a new branch ref
        branch_name = f"{branch_prefix}-{uuid.uuid4().hex[:6]}"
        create_ref = client.post(
            f"{base_api}/git/refs",
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
        )
        if create_ref.status_code not in (200, 201):
            raise GitError(
                f"Could not create branch {branch_name!r} (check token write permissions): "
                f"{create_ref.text[:200]}"
            )

        # 3. Commit each modified/new file onto the new branch via Contents API
        last_commit_sha = base_sha[:7]
        for rel in changed:
            full = ws.work / Path(*PurePosixPath(rel).parts)
            content_b64 = base64.b64encode(full.read_bytes()).decode("ascii")

            existing_sha: str | None = None
            existing = client.get(f"{base_api}/contents/{rel}", params={"ref": branch_name})
            if existing.status_code == 200 and isinstance(existing.json(), dict):
                existing_sha = existing.json().get("sha")

            put_payload: dict[str, str] = {
                "message": f"Fix: {issue_title} ({rel})",
                "content": content_b64,
                "branch": branch_name,
            }
            if existing_sha:
                put_payload["sha"] = existing_sha

            put_resp = client.put(f"{base_api}/contents/{rel}", json=put_payload)
            if put_resp.status_code not in (200, 201):
                raise GitError(f"Failed to commit {rel!r}: {put_resp.text[:200]}")
            last_commit_sha = put_resp.json().get("commit", {}).get("sha", last_commit_sha)[:7]

        # 4. Open the Pull Request
        pr_body = (
            f"## Automated Bug Fix\n\n"
            f"**Issue:** {issue_title}\n\n"
            f"{issue_body}\n\n"
            f"### Root-Cause & Fix Rationale\n{rationale}\n\n"
            f"### Files Changed\n" + "\n".join(f"- `{c}`" for c in changed)
        )
        pr_resp = client.post(
            f"{base_api}/pulls",
            json={
                "title": f"Fix: {issue_title}",
                "head": branch_name,
                "base": default_branch,
                "body": pr_body,
            },
        )
        if pr_resp.status_code not in (200, 201):
            raise GitError(f"Branch {branch_name!r} pushed, but PR creation failed: {pr_resp.text[:200]}")

        pr_url = pr_resp.json().get("html_url")
        return CommitResult(
            branch=branch_name,
            commit_sha=last_commit_sha,
            pr_url=pr_url,
            applied_to_repo=True,
        )
