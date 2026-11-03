"""Prompt templates and active context-window pruning for Groq's free tier (8K TPM)."""

from __future__ import annotations

from agent.state import AgentState, FixAttempt

REPRO_SYSTEM_PROMPT = """You are a senior Python QA & bug-reproduction engineer.
Given a bug report and repository files, write a minimal, self-contained `pytest` test file
that FAILS on the current buggy code and will PASS once the bug is fixed.

Rules:
- Import directly from the repository modules shown in the context.
- Keep the test deterministic, fast, and focused on the exact bug described.
- Choose a filename like `test_repro_bug.py` in the repository root.
"""

FIX_SYSTEM_PROMPT = """You are an autonomous Python bug-fixing engineer.
Given a bug report, relevant repository files, a failing pytest reproduction test, and any prior
attempt history, propose a minimal, clean fix using exact search/replace edits.

Rules:
- Only edit existing source files (do NOT edit or weaken the reproduction test to make it pass).
- In each `FileEdit`, `search` MUST be an exact, verbatim substring of the current file content
  (including indentation) that appears ONCE in that file.
- `replace` is the exact replacement code with matching indentation.
- Preserve existing public signatures and unrelated behavior.
"""

ANALYZE_SYSTEM_PROMPT = """You are a senior debugging assistant.
Summarize in 2-3 concise sentences why the latest fix attempt failed (or what the human reviewer
requested) and what the next attempt should change.
"""


def summarize_past_attempts(attempts: list[FixAttempt], max_Summary_chars: int = 1200) -> str:
    """Prune past attempts into one-line summaries so context never blows Groq's 8K TPM limit."""
    if not attempts:
        return "None."
    lines: list[str] = []
    for a in attempts[:-1]:
        files = ", ".join(e.path for e in a.edits) or ", ".join(p.path for p in a.patches) or "none"
        if a.patch_error:
            outcome = f"patch failed ({a.patch_error[:100]})"
        elif a.result is not None:
            first_fail = next(
                (ln.strip() for ln in (a.result.stdout + "\n" + a.result.stderr).splitlines() if "FAILED" in ln or "Error" in ln or "assert " in ln),
                f"exit_code={a.result.exit_code}",
            )
            outcome = "PASSED" if a.result.passed else f"FAILED ({first_fail[:120]})"
        else:
            outcome = "not run"
        lines.append(f"- Iter {a.iteration} [{files}]: {a.rationale[:100]} -> {outcome}")
    text = "\n".join(lines)
    return text[-max_Summary_chars:]


def build_repro_user_prompt(state: AgentState, last_repro_stdout: str = "") -> str:
    issue = state["issue"]
    repo = state["repo"]
    budget = state["budget"]

    files_block = "\n\n".join(
        f"### File: `{path}`\n```python\n{snippet}\n```"
        for path, snippet in repo.relevant_files.items()
    )
    retry_note = ""
    if last_repro_stdout:
        trimmed = last_repro_stdout[-budget.max_tool_output_chars :]
        retry_note = (
            "\n\n## Previous Reproduction Attempt Did Not Fail As Expected\n"
            "The previous test passed or errored on import. Write a test that specifically "
            f"triggers the bug and fails with an assertion:\n```\n{trimmed}\n```"
        )
    return (
        f"## Issue: {issue.title}\n{issue.body}\n\n"
        f"## Repository Tree\n```\n{repo.tree}\n```\n\n"
        f"## Relevant Source Files\n{files_block}"
        f"{retry_note}"
    )


def build_fix_user_prompt(state: AgentState, current_files: dict[str, str]) -> str:
    issue = state["issue"]
    repro = state["repro_test"]
    budget = state["budget"]
    attempts = state.get("attempts", [])

    files_block = "\n\n".join(
        f"### File: `{path}` (exact baseline content)\n```python\n{content}\n```"
        for path, content in current_files.items()
    )

    repro_out = ""
    if state.get("repro_result"):
        res = state["repro_result"]
        combined = (res.stdout + "\n" + res.stderr).strip()[-budget.max_tool_output_chars :]
        repro_out = f"\n\n## Initial Reproduction Test Failure\n```\n{combined}\n```"

    latest_failure = ""
    if attempts:
        last = attempts[-1]
        if last.patch_error:
            latest_failure = (
                f"\n\n## Latest Attempt ({last.iteration}) Edit Application Error\n"
                f"`{last.patch_error}`\n"
                "Make sure `search` is copied verbatim from the file content above."
            )
        elif last.result and not last.result.passed:
            out = (last.result.stdout + "\n" + last.result.stderr).strip()[
                -budget.max_tool_output_chars :
            ]
            latest_failure = (
                f"\n\n## Latest Attempt ({last.iteration}) Test Output\n"
                f"Proposed rationale: {last.rationale}\n"
                f"```\n{out}\n```"
            )

    diagnosis = ""
    if state.get("diagnosis"):
        diagnosis = f"\n\n## Failure Diagnosis / Reviewer Feedback\n{state['diagnosis']}"

    approval = state.get("approval")
    if approval and not approval.approved and approval.feedback:
        diagnosis += f"\n\n## Human Reviewer Requested Changes\n{approval.feedback}"

    history = summarize_past_attempts(attempts)
    return (
        f"## Issue: {issue.title}\n{issue.body}\n\n"
        f"## Reproduction Test (`{repro.path}`)\n```python\n{repro.content}\n```"
        f"{repro_out}\n\n"
        f"## Past Attempts Summary (pruned)\n{history}"
        f"{latest_failure}"
        f"{diagnosis}\n\n"
        f"## Source Files Available to Edit\n{files_block}"
    )
