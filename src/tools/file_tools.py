"""Path-jailed workspace for the agent: baseline + working copy of the target repo.

The user's repo is never modified during a run. ``Workspace.create`` copies it twice
(``base/`` stays pristine, ``work/`` is what the sandbox runs against). Every fix attempt
starts from the baseline (:meth:`Workspace.reset`), applies LLM-proposed search/replace
edits atomically, and diffs are always computed against the baseline. All paths coming
from the LLM go through :meth:`Workspace.resolve`, which rejects anything outside ``work/``.
"""

from __future__ import annotations

import difflib
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path, PurePosixPath

from agent.state import FileEdit

IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
    "dist", "build", ".eggs",
})  # fmt: skip
TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".json", ".yml", ".yaml",
})  # fmt: skip

MAX_FILES = 20_000
MAX_BYTES = 200 * 1024 * 1024
MAX_GREP_FILE_BYTES = 200 * 1024


class WorkspaceError(Exception):
    """Workspace could not be created or used (too large, missing, ...)."""


class PathEscapeError(WorkspaceError):
    """A path resolved outside the workspace."""


class EditError(WorkspaceError):
    """A proposed edit does not apply cleanly."""


def _ignored(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(".egg-info")


def _walk_files(root: Path) -> Iterator[Path]:
    """Yield files under ``root`` (relative), skipping ignored directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _ignored(d))
        for name in sorted(filenames):
            yield (Path(dirpath) / name).relative_to(root)


def clone_repo_to_temp(repo_url: str, timeout_s: int = 90) -> Path:
    """Shallow-clone a GitHub/Git URL into a temporary directory."""
    url = repo_url.strip()
    if not re.match(r"^(https?://|git@)", url):
        raise WorkspaceError(f"Invalid repository URL: {url!r}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="bugfix-clone-"))
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "clone", "--depth", "1", url, str(tmp_dir)],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if res.returncode != 0:
            raise WorkspaceError(f"git clone failed: {res.stderr.strip() or res.stdout.strip()}")
        return tmp_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def create_repo_from_uploads(uploaded_files: Sequence[ tuple[str, bytes]]) -> Path:
    """Build a temporary repository directory from uploaded `.zip` archives or source files.

    - Safe against Zip-Slip (`..` or absolute paths inside archives are rejected).
    - Automatically strips a single common top-level wrapper folder in `.zip` archives
      (e.g. GitHub's `repo-main/` zip layout) so imports resolve from the repo root.
    """
    if not uploaded_files:
        raise WorkspaceError("No files uploaded. Please upload a .zip archive or .py files.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="bugfix-upload-")).resolve()
    total_bytes = 0
    file_count = 0

    try:
        for raw_name, data in uploaded_files:
            name = Path(raw_name).name
            if not name:
                continue
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.infolist() if not m.is_dir()]
                    # Detect if all members share a single top-level directory
                    top_dirs = {
                        PurePosixPath(m.filename.replace("\\", "/")).parts[0]
                        for m in members
                        if len(PurePosixPath(m.filename.replace("\\", "/")).parts) > 1
                    }
                    has_root_files = any(
                        len(PurePosixPath(m.filename.replace("\\", "/")).parts) == 1
                        for m in members
                    )
                    strip_prefix = len(top_dirs) == 1 and not has_root_files

                    for m in members:
                        p = PurePosixPath(m.filename.replace("\\", "/"))
                        if p.is_absolute() or ".." in p.parts:
                            raise PathEscapeError(f"Unsafe path in zip archive: {m.filename!r}")
                        parts = p.parts[1:] if strip_prefix else p.parts
                        if not parts or any(_ignored(part) for part in parts[:-1]):
                            continue
                        file_count += 1
                        total_bytes += m.file_size
                        if file_count > MAX_FILES or total_bytes > MAX_BYTES:
                            raise WorkspaceError("Uploaded archive exceeds maximum size/file limit.")
                        dest = (tmp_dir / Path(*parts)).resolve()
                        if not dest.is_relative_to(tmp_dir):
                            raise PathEscapeError(f"Unsafe path in zip archive: {m.filename!r}")
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zf.read(m))
            else:
                p = PurePosixPath(name.replace("\\", "/"))
                if p.is_absolute() or ".." in p.parts:
                    raise PathEscapeError(f"Unsafe uploaded filename: {raw_name!r}")
                file_count += 1
                total_bytes += len(data)
                if file_count > MAX_FILES or total_bytes > MAX_BYTES:
                    raise WorkspaceError("Uploaded files exceed maximum size/file limit.")
                dest = (tmp_dir / p.name).resolve()
                if not dest.is_relative_to(tmp_dir):
                    raise PathEscapeError(f"Unsafe uploaded filename: {raw_name!r}")
                dest.write_bytes(data)

        if not any(_walk_files(tmp_dir)):
            raise WorkspaceError("Uploaded archive/files contained no usable source files.")
        return tmp_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.base = self.root / "base"
        self.work = self.root / "work"
        if not self.work.is_dir() or not self.base.is_dir():
            raise WorkspaceError(f"Not a workspace: {self.root}")

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    def create(cls, repo_path: Path) -> Workspace:
        """Copy ``repo_path`` into a fresh temp workspace (baseline + working copy)."""
        src = Path(repo_path).resolve()
        if not src.is_dir():
            raise WorkspaceError(f"Repo path is not a directory: {src}")

        files = total = 0
        for rel in _walk_files(src):
            files += 1
            total += (src / rel).stat().st_size
            if files > MAX_FILES or total > MAX_BYTES:
                raise WorkspaceError(
                    f"Repo too large for the agent (> {MAX_FILES} files or "
                    f"{MAX_BYTES // 2**20} MB, ignoring {sorted(IGNORED_DIRS)})."
                )

        root = Path(tempfile.mkdtemp(prefix="bugfix-ws-"))
        ignore = shutil.ignore_patterns(*IGNORED_DIRS, "*.egg-info")
        try:
            shutil.copytree(src, root / "base", ignore=ignore)
            shutil.copytree(root / "base", root / "work")
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return cls(root)

    @classmethod
    def open(cls, root: str | Path) -> Workspace:
        return cls(Path(root))

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # --------------------------------------------------------------------- paths

    def resolve(self, rel: str) -> Path:
        """Resolve a repo-relative path inside ``work/``; reject escapes."""
        if not rel or rel.strip() != rel:
            raise PathEscapeError(f"Invalid path: {rel!r}")
        p = PurePosixPath(rel.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", rel):
            raise PathEscapeError(f"Path escapes workspace: {rel!r}")
        full = (self.work / Path(*p.parts)).resolve()
        if not full.is_relative_to(self.work):
            raise PathEscapeError(f"Path escapes workspace: {rel!r}")
        return full

    def _rel(self, full: Path) -> str:
        return full.relative_to(self.work).as_posix()

    # ------------------------------------------------------------------ reading

    def list_tree(self, max_entries: int = 120) -> str:
        """Compact file listing of the working copy (ignored dirs skipped)."""
        entries = [p.as_posix() for p in _walk_files(self.work)]
        shown = entries[:max_entries]
        extra = len(entries) - len(shown)
        return "\n".join(shown) + (f"\n... (+{extra} more files)" if extra > 0 else "")

    def list_python_files(self) -> list[str]:
        """Return all ``.py`` files in the working copy."""
        return [p.as_posix() for p in _walk_files(self.work) if p.suffix == ".py"]

    def exists(self, rel: str) -> bool:
        return self.resolve(rel).is_file()

    def read(self, rel: str, max_chars: int | None = None) -> str:
        full = self.resolve(rel)
        if not full.is_file():
            raise FileNotFoundError(rel)
        text = full.read_text(encoding="utf-8", errors="replace")
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"
        return text

    def read_baseline(self, rel: str) -> str | None:
        """Content of ``rel`` in the pristine baseline, or ``None`` if it did not exist."""
        self.resolve(rel)  # jail check
        p = self.base / Path(*PurePosixPath(rel.replace("\\", "/")).parts)
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None

    def grep_counts(self, tokens: Iterable[str]) -> dict[str, int]:
        """Count literal occurrences of ``tokens`` per text file (relative path -> hits)."""
        toks = [t for t in dict.fromkeys(tokens) if len(t) >= 3]
        hits: dict[str, int] = {}
        if not toks:
            return hits
        for rel in _walk_files(self.work):
            full = self.work / rel
            if full.suffix not in TEXT_SUFFIXES or full.stat().st_size > MAX_GREP_FILE_BYTES:
                continue
            try:
                text = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            n = sum(text.count(t) for t in toks)
            if n:
                hits[rel.as_posix()] = n
        return hits

    # ------------------------------------------------------------------ writing

    def write(self, rel: str, content: str) -> None:
        full = self.resolve(rel)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8", newline="")

    def reset(self, paths: Iterable[str] | None = None) -> list[str]:
        """Restore baseline content of modified files in ``work/``.

        Files that do not exist in the baseline (e.g. the repro test) are left alone.
        Returns the paths that were restored.
        """
        restored = []
        candidates = list(paths) if paths is not None else [
            p.as_posix() for p in _walk_files(self.base)
        ]
        for rel in candidates:
            src = self.base / Path(*PurePosixPath(rel).parts)
            if not src.is_file():
                continue
            dst = self.resolve(rel)
            if not dst.is_file() or dst.read_bytes() != src.read_bytes():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                restored.append(rel)
        return restored

    def apply_edits(self, edits: Iterable[FileEdit]) -> dict[str, str]:
        """Apply search/replace edits atomically.

        All edits are computed in memory first; nothing is written unless every edit
        applies. Each ``search`` must match exactly once. Returns ``{path: new_content}``.
        """
        pending: dict[str, str] = {}
        for edit in edits:
            full = self.resolve(edit.path)
            rel = self._rel(full)
            if rel not in pending:
                if not full.is_file():
                    raise EditError(f"{edit.path}: file does not exist (edits can only modify files)")
                pending[rel] = full.read_text(encoding="utf-8", newline="")
            text = pending[rel]

            search, replace = edit.search, edit.replace
            if "\r\n" in text:  # keep the file's line endings
                search = search.replace("\r\n", "\n").replace("\n", "\r\n")
                replace = replace.replace("\r\n", "\n").replace("\n", "\r\n")

            count = text.count(search)
            if count == 0 and search.strip("\n") != search:
                search = search.strip("\n")
                count = text.count(search)
            if count == 0:
                first = search.strip().splitlines()[0][:80] if search.strip() else ""
                raise EditError(f"{edit.path}: search text not found (starts with {first!r})")
            if count > 1:
                raise EditError(
                    f"{edit.path}: search text matches {count} times; add surrounding lines "
                    "so it matches exactly once"
                )
            pending[rel] = text.replace(search, replace, 1)

        if not pending:
            raise EditError("no edits provided")
        for rel, content in pending.items():
            self.write(rel, content)
        return pending

    # -------------------------------------------------------------------- diffs

    def diff(self, rel: str) -> str:
        """Unified diff of ``rel`` (baseline -> working copy); '' if unchanged."""
        old = self.read_baseline(rel)
        new = self.read(rel) if self.resolve(rel).is_file() else None
        if old == new:
            return ""
        old_lines = (old or "").splitlines(keepends=True)
        new_lines = (new or "").splitlines(keepends=True)
        for lines in (old_lines, new_lines):
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n\\ No newline at end of file\n"
        return "".join(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{rel}" if old is not None else "/dev/null",
            tofile=f"b/{rel}" if new is not None else "/dev/null",
        ))  # fmt: skip

    def changed_files(self) -> list[str]:
        """Files in ``work/`` that differ from the baseline (modified or newly created)."""
        changed = []
        for rel in _walk_files(self.work):
            r = rel.as_posix()
            base = self.base / rel
            if not base.is_file() or base.read_bytes() != (self.work / rel).read_bytes():
                changed.append(r)
        return changed

    def full_diff(self) -> str:
        return "".join(self.diff(r) for r in self.changed_files())

    def export_zip_bytes(self) -> bytes:
        """Package the entire fixed working copy (``work/``) into an in-memory ``.zip`` archive."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in _walk_files(self.work):
                full = self.work / rel
                if full.is_file():
                    zf.write(full, arcname=rel.as_posix())
        return buf.getvalue()


def validate_repro_path(rel: str) -> str:
    """Repro tests must be ``test_*.py`` / ``*_test.py`` files; returns the normalized path."""
    p = PurePosixPath(rel.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or p.suffix != ".py":
        raise PathEscapeError(f"Invalid repro test path: {rel!r}")
    if not (p.name.startswith("test_") or p.name.endswith("_test.py")):
        raise PathEscapeError(f"Repro test file must be named test_*.py: {rel!r}")
    if any(_ignored(part) for part in p.parts[:-1]):
        raise PathEscapeError(f"Repro test is in an ignored directory: {rel!r}")
    return p.as_posix()
