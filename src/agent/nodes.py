"""Production LangGraph node functions (`src/agent/nodes.py`).

Every node takes `(state: AgentState, config: RunnableConfig)` and returns a partial
state update dict. Dependencies (`llm`, `sandbox_backend`, `commit_in_git`) are read
from `config["configurable"]`, with sensible defaults so tests can inject fakes while
production uses Groq + auto sandbox selection.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent.llm import (
    DetectedBugProposal,
    FixProposal,
    ReproTestProposal,
    create_groq_llm,
    invoke_structured,
    invoke_with_backoff,
)
from agent.prompts import (
    ANALYZE_SYSTEM_PROMPT,
    DETECT_BUG_SYSTEM_PROMPT,
    FIX_SYSTEM_PROMPT,
    REPRO_SYSTEM_PROMPT,
    build_fix_user_prompt,
    build_repro_user_prompt,
)
from agent.state import (
    AgentState,
    AgentStatus,
    ApprovalDecision,
    FilePatch,
    FixAttempt,
    RepoContext,
    RepoFinding,
    ReproTest,
    RunBudget,
    SandboxResult,
)
from sandbox.base import Sandbox
from sandbox.factory import create_sandbox
from tools.file_tools import (
    EditError,
    PathEscapeError,
    Workspace,
    validate_repro_path,
)
from tools.git_tools import apply_workspace_to_repo

logger = logging.getLogger(__name__)

MAX_RELEVANT_FILES = 4
MAX_FILE_SNIPPET_CHARS = 2200


def _get_llm(config: RunnableConfig) -> BaseChatModel:
    cfg = config.get("configurable", {})
    if "llm" in cfg and cfg["llm"] is not None:
        return cfg["llm"]
    return create_groq_llm(
        model=cfg.get("groq_model"),
        api_key=cfg.get("groq_api_key"),
    )


def _run_in_sandbox(ws: Workspace, command: str, config: RunnableConfig) -> SandboxResult:
    cfg = config.get("configurable", {})
    injected: Sandbox | None = cfg.get("sandbox")
    if injected is not None:
        return injected.run_tests(command)
    backend = cfg.get("sandbox_backend", "auto")
    with create_sandbox(backend) as sb:
        sb.start(ws.work)
        return sb.run_tests(command)


def _extract_tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    stop = {"the", "and", "for", "with", "that", "this", "from", "returns", "instead", "should", "when"}
    return [w for w in dict.fromkeys(words) if w.lower() not in stop]


# --------------------------------------------------------------------------- nodes


def ingest_issue(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Validate issue input, initialize workspace copy and budget."""
    issue = state["issue"]
    budget = state.get("budget") or RunBudget()
    ws_root = state.get("workspace_root")
    if not ws_root:
        ws = Workspace.create(issue.repo_path)
        ws_root = str(ws.root)
    return {
        "budget": budget,
        "iteration": state.get("iteration", 0),
        "repro_attempts": state.get("repro_attempts", 0),
        "workspace_root": ws_root,
        "status": AgentStatus.INGESTED,
    }


def inspect_repo(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Build directory tree and pick the top relevant Python files for the issue."""
    ws = Workspace.open(state["workspace_root"])
    issue = state["issue"]
    tree = ws.list_tree()

    tokens = _extract_tokens(f"{issue.title}\n{issue.body}")
    grep_hits = ws.grep_counts(tokens)
    py_files = ws.list_python_files()

    # Prefer source files over tests when ranking relevance, then fall back to all .py files
    ranked = sorted(
        py_files,
        key=lambda p: (
            0 if not (p.startswith("test") or "/test" in p or p.endswith("_test.py")) else 1,
            -grep_hits.get(p, 0),
            len(p),
        ),
    )
    selected = ranked[:MAX_RELEVANT_FILES]
    relevant = {p: ws.read(p, max_chars=MAX_FILE_SNIPPET_CHARS) for p in selected}

    repo = RepoContext(
        tree=tree,
        relevant_files=relevant,
        test_command="pytest -q -p no:cacheprovider",
        language="python",
    )

    # If the user left the default calc.py sample issue on a custom repository that has no calc.py,
    # or requested auto-detection, let the LLM audit the repo and formulate a real testable issue.
    updates: dict[str, Any] = {"repo": repo, "status": AgentStatus.INSPECTED}
    has_calc = any(p.endswith("calc.py") for p in py_files)
    is_default_calc_issue = "average()" in issue.title and "calc.average" in issue.body
    wants_auto = issue.title.strip().lower().startswith("auto") or (
        is_default_calc_issue and not has_calc and bool(relevant)
    )
    if wants_auto:
        try:
            llm = _get_llm(config)
            files_block = "\n\n".join(
                f"### File: `{p}`\n```python\n{src}\n```" for p, src in relevant.items()
            )
            detected = invoke_structured(
                llm,
                DetectedBugProposal,
                DETECT_BUG_SYSTEM_PROMPT,
                f"Repository Tree:\n```\n{tree}\n```\n\nSource Files:\n{files_block}",
            )
            updates["issue"] = issue.model_copy(
                update={"title": detected.title, "body": detected.body}
            )
            if detected.file_path in py_files and detected.file_path not in relevant:
                relevant[detected.file_path] = ws.read(
                    detected.file_path, max_chars=MAX_FILE_SNIPPET_CHARS
                )
            findings = list(detected.findings) if detected.findings else [
                RepoFinding(
                    severity="High",
                    file_path=detected.file_path,
                    title=detected.title,
                    description=detected.body,
                    suggested_fix="Apply automated search/replace fix verified by pytest.",
                )
            ]
            repo = repo.model_copy(update={"relevant_files": relevant, "findings": findings})
            updates["repo"] = repo
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-detect bug fallback skipped: %s", exc)
    if not updates["repo"].findings:
        primary_file = next(iter(relevant.keys()), "repository")
        updates["repo"] = updates["repo"].model_copy(
            update={
                "findings": [
                    RepoFinding(
                        severity="High",
                        file_path=primary_file,
                        title=updates.get("issue", issue).title,
                        description=updates.get("issue", issue).body,
                        suggested_fix="Apply targeted code fix and verify with regression test.",
                    )
                ]
            }
        )
    return updates


def write_repro_test(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Ask the LLM to write a minimal failing pytest reproduction test."""
    ws = Workspace.open(state["workspace_root"])
    llm = _get_llm(config)

    prev_out = ""
    if state.get("repro_result") is not None:
        r = state["repro_result"]
        prev_out = f"{r.stdout}\n{r.stderr}".strip()

    user_prompt = build_repro_user_prompt(state, last_repro_stdout=prev_out)
    proposal = invoke_structured(llm, ReproTestProposal, REPRO_SYSTEM_PROMPT, user_prompt)

    try:
        rel_path = validate_repro_path(proposal.path)
    except PathEscapeError:
        rel_path = "test_repro_bug.py"

    ws.write(rel_path, proposal.content)
    repro = ReproTest(path=rel_path, content=proposal.content, confirmed_failing=False)
    return {
        "repro_test": repro,
        "repro_attempts": state.get("repro_attempts", 0) + 1,
        "status": AgentStatus.REPRO_WRITTEN,
    }


def run_repro_test(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Run the reproduction test in the sandbox. We expect it to FAIL on buggy code."""
    ws = Workspace.open(state["workspace_root"])
    repro = state["repro_test"]
    cmd = f"pytest -q -p no:cacheprovider {repro.path}"
    res = _run_in_sandbox(ws, cmd, config)

    # Exit code 1 = tests collected and failed (desired). Exit codes 2/3/4 = syntax/import error.
    confirmed = (not res.passed) and (not res.timed_out) and res.exit_code == 1
    updated = repro.model_copy(update={"confirmed_failing": confirmed})
    return {
        "repro_test": updated,
        "repro_result": res,
        "status": AgentStatus.REPRO_CONFIRMED if confirmed else AgentStatus.REPRO_WRITTEN,
    }


def propose_fix(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Reset working copy to baseline and ask the LLM for search/replace edits."""
    ws = Workspace.open(state["workspace_root"])
    ws.reset()  # Always propose edits against clean baseline files

    current_files = {
        p: ws.read(p, max_chars=MAX_FILE_SNIPPET_CHARS)
        for p in state["repo"].relevant_files
        if ws.exists(p)
    }
    llm = _get_llm(config)
    user_prompt = build_fix_user_prompt(state, current_files)
    proposal = invoke_structured(llm, FixProposal, FIX_SYSTEM_PROMPT, user_prompt)

    next_iter = state.get("iteration", 0) + 1
    attempt = FixAttempt(
        iteration=next_iter,
        rationale=proposal.rationale,
        edits=proposal.edits,
    )
    return {
        "attempts": [attempt],
        "status": AgentStatus.FIXING,
    }


def apply_fix(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Apply the latest attempt's search/replace edits in the workspace and compute unified diffs."""
    ws = Workspace.open(state["workspace_root"])
    ws.reset()
    latest = state["attempts"][-1]
    repro = state.get("repro_test")

    for edit in latest.edits:
        if repro and edit.path.replace("\\", "/") == repro.path:
            updated = latest.model_copy(
                update={"patch_error": f"Refusing to modify reproduction test {repro.path!r}"}
            )
            return {"attempts": [updated], "iteration": latest.iteration}

    try:
        modified = ws.apply_edits(latest.edits)
    except (EditError, PathEscapeError) as exc:
        updated = latest.model_copy(update={"patch_error": str(exc)})
        return {"attempts": [updated], "iteration": latest.iteration}

    patches = [
        FilePatch(path=rel, unified_diff=ws.diff(rel))
        for rel in modified
        if ws.diff(rel).strip()
    ]
    if not patches:
        updated = latest.model_copy(update={"patch_error": "Edits produced no net change."})
        return {"attempts": [updated], "iteration": latest.iteration}

    updated = latest.model_copy(update={"patches": patches, "patch_error": None})
    return {"attempts": [updated]}


def run_tests(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Run the reproduction test + full test suite inside the sandbox."""
    ws = Workspace.open(state["workspace_root"])
    latest = state["attempts"][-1]
    res = _run_in_sandbox(ws, state["repo"].test_command, config)
    updated = latest.model_copy(update={"result": res})
    return {
        "attempts": [updated],
        "iteration": latest.iteration,
        "status": AgentStatus.TESTS_PASSING if res.passed else AgentStatus.FIXING,
    }


def analyze_failure(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Produce a short failure diagnosis for the next `propose_fix` iteration."""
    approval = state.get("approval")
    if approval and not approval.approved and approval.feedback:
        return {
            "diagnosis": f"Reviewer requested changes: {approval.feedback}",
            "approval": None,
        }

    latest = state["attempts"][-1]
    if latest.patch_error:
        return {
            "diagnosis": (
                f"Attempt {latest.iteration} edits could not be applied: {latest.patch_error}. "
                "Copy `search` blocks verbatim from the baseline file."
            )
        }

    out = ""
    if latest.result:
        out = (latest.result.stdout + "\n" + latest.result.stderr).strip()[
            -state["budget"].max_tool_output_chars :
        ]
    try:
        llm = _get_llm(config)
        msg = invoke_with_backoff(
            llm,
            [
                SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Issue: {state['issue'].title}\n"
                        f"Attempt {latest.iteration} rationale: {latest.rationale}\n"
                        f"Test output:\n{out}"
                    )
                ),
            ],
        )
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
    except Exception:  # noqa: BLE001
        text = f"Attempt {latest.iteration} failed tests; inspect assertion output and adjust logic."
    return {"diagnosis": text.strip()[:600]}


def human_approval(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Pause the graph with `interrupt()` for human review before any git commit."""
    attempt = state["attempts"][-1]
    decision = interrupt(
        {
            "iteration": attempt.iteration,
            "rationale": attempt.rationale,
            "diff": "\n".join(p.unified_diff for p in attempt.patches),
            "test_output": (attempt.result.stdout + "\n" + attempt.result.stderr).strip()
            if attempt.result
            else "",
            "repro_test": state["repro_test"].content,
        }
    )
    approval = ApprovalDecision.model_validate(decision)
    return {
        "approval": approval,
        "status": AgentStatus.APPROVED if approval.approved else AgentStatus.REJECTED,
    }


def commit_and_pr(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Commit approved changes (reachable ONLY after `human_approval`)."""
    ws = Workspace.open(state["workspace_root"])
    issue = state["issue"]
    cfg = config.get("configurable", {})
    commit_in_git = bool(cfg.get("commit_in_git", False))
    slug = f"agent/fix-{issue.number or 'issue'}"
    msg = f"Fix: {issue.title}"

    res = apply_workspace_to_repo(
        ws,
        issue.repo_path,
        branch_name=slug,
        commit_message=msg,
        commit_in_git=commit_in_git,
    )
    return {
        "branch": res.branch,
        "commit_sha": res.commit_sha,
        "pr_url": res.pr_url,
        "status": AgentStatus.COMMITTED,
    }


def escalate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Terminal failure handler when reproduction or fix iterations are exhausted, or rejected."""
    approval = state.get("approval")
    repro = state.get("repro_test")
    if approval is not None and not approval.approved and not approval.feedback:
        reason = "Human reviewer rejected the proposed fix."
    elif repro is not None and not repro.confirmed_failing:
        reason = (
            f"Could not reproduce the bug with a failing pytest test after "
            f"{state.get('repro_attempts', 0)} attempt(s)."
        )
    else:
        reason = (
            f"Exhausted {state.get('iteration', 0)}/{state['budget'].max_iterations} "
            "fix iterations without all tests passing."
        )
    return {"error": reason, "status": AgentStatus.ESCALATED}
