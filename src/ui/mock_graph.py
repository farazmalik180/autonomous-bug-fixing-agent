"""DEMO ONLY: a scripted LangGraph that mimics the real agent's control flow.

No LLM, no Docker, nothing is committed. It uses the real ``AgentState`` models, the
real conditional-routing shape from SPEC.md section 4, a real ``interrupt()`` before
``commit_and_pr``, and a real checkpointer, so the Streamlit UI exercises the same
pause/resume mechanics the production graph will use.

Scenario: ``calc.average()`` has an off-by-one bug. The scripted agent first proposes a
fix that is wrong (integer division), recovers on the second attempt, and, if the
reviewer requests changes, adds an empty-list guard on the third.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.state import (
    AgentState,
    AgentStatus,
    ApprovalDecision,
    FilePatch,
    FixAttempt,
    RepoContext,
    ReproTest,
    SandboxResult,
)

DEMO_ISSUE_TITLE = "average() returns wrong values"
DEMO_ISSUE_BODY = (
    "`calc.average([2, 4, 6])` returns `3.0` instead of `4`, and "
    "`calc.average([1, 2])` returns `1.0` instead of `1.5`."
)

BUGGY_SOURCE = "def average(xs):\n    return sum(xs) / (len(xs) + 1)\n"

REPRO_TEST = '''from calc import average


def test_average_of_three():
    assert average([2, 4, 6]) == 4


def test_average_of_two_is_fractional():
    assert average([1, 2]) == 1.5
'''

REPRO_FAIL_OUTPUT = """FAILED test_repro.py::test_average_of_three - assert 3.0 == 4
FAILED test_repro.py::test_average_of_two_is_fractional - assert 1.0 == 1.5
2 failed in 0.04s"""

# (rationale, new line(s) of the function body, diff, test passes?, pytest output)
_FIXES: list[tuple[str, str, bool, str]] = [
    (
        "The denominator has an extra +1. Remove it and use integer division.",
        "    return sum(xs) // len(xs)",
        False,
        "FAILED test_repro.py::test_average_of_two_is_fractional - assert 1 == 1.5\n"
        "1 failed, 1 passed in 0.04s",
    ),
    (
        "Integer division truncated the fractional result. Use true division.",
        "    return sum(xs) / len(xs)",
        True,
        "2 passed in 0.03s",
    ),
    (
        "Reviewer asked for edge-case handling: reject empty input explicitly "
        "instead of raising ZeroDivisionError.",
        "    if not xs:\n"
        '        raise ValueError("average() requires at least one value")\n'
        "    return sum(xs) / len(xs)",
        True,
        "2 passed in 0.03s",
    ),
]

_OLD_LINE = "    return sum(xs) / (len(xs) + 1)"


def _pause(config: RunnableConfig) -> None:
    time.sleep(float(config.get("configurable", {}).get("delay", 0.6)))


def _diff_for(new_body: str) -> str:
    old = f"-{_OLD_LINE}"
    new = "\n".join(f"+{line}" for line in new_body.split("\n"))
    n_new = len(new_body.split("\n"))
    return f"--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,{1 + n_new} @@\n def average(xs):\n{old}\n{new}\n"


# --------------------------------------------------------------------------- nodes


def ingest_issue(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    return {"status": AgentStatus.INGESTED}


def inspect_repo(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    repo = RepoContext(
        tree="calc.py\ntests/\n",
        relevant_files={"calc.py": BUGGY_SOURCE},
        test_command="pytest -q -p no:cacheprovider",
    )
    return {"repo": repo, "status": AgentStatus.INSPECTED}


def write_repro_test(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    return {
        "repro_test": ReproTest(path="test_repro.py", content=REPRO_TEST),
        "repro_attempts": state.get("repro_attempts", 0) + 1,
        "status": AgentStatus.REPRO_WRITTEN,
    }


def run_repro_test(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    repro = state["repro_test"].model_copy(update={"confirmed_failing": True})
    return {"repro_test": repro, "status": AgentStatus.REPRO_CONFIRMED}


def propose_fix(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    iteration = state.get("iteration", 0)
    rationale, body, _passes, _out = _FIXES[min(iteration, len(_FIXES) - 1)]
    attempt = FixAttempt(
        iteration=iteration + 1,
        rationale=rationale,
        patches=[FilePatch(path="calc.py", unified_diff=_diff_for(body))],
    )
    return {"attempts": [attempt], "status": AgentStatus.FIXING}


def run_tests(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    iteration = state.get("iteration", 0)
    _r, _b, passes, output = _FIXES[min(iteration, len(_FIXES) - 1)]
    result = SandboxResult(exit_code=0 if passes else 1, stdout=output, stderr="", duration_s=0.04)
    attempt = state["attempts"][-1].model_copy(update={"result": result})
    return {
        "attempts": [attempt],  # same iteration -> upserted by merge_attempts
        "iteration": iteration + 1,
        "status": AgentStatus.TESTS_PASSING if passes else AgentStatus.FIXING,
    }


def analyze_failure(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    return {}


def human_approval(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Pause the graph. Nothing before this line has side effects (the node re-runs on resume)."""
    attempt = state["attempts"][-1]
    decision = interrupt(
        {
            "iteration": attempt.iteration,
            "rationale": attempt.rationale,
            "diff": "\n".join(p.unified_diff for p in attempt.patches),
            "test_output": attempt.result.stdout if attempt.result else "",
            "repro_test": state["repro_test"].content,
        }
    )
    approval = ApprovalDecision.model_validate(decision)
    return {
        "approval": approval,
        "status": AgentStatus.APPROVED if approval.approved else AgentStatus.REJECTED,
    }


def commit_and_pr(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    diff = state["attempts"][-1].patches[0].unified_diff
    sha = hashlib.sha1(diff.encode()).hexdigest()[:7]  # noqa: S324 - demo id only
    return {"branch": "agent/fix-demo", "commit_sha": sha, "status": AgentStatus.COMMITTED}


def escalate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _pause(config)
    approval = state.get("approval")
    reason = (
        "Reviewer rejected the fix."
        if approval is not None and not approval.approved
        else "Iteration budget exhausted without a passing fix."
    )
    return {"error": reason, "status": AgentStatus.ESCALATED}


# -------------------------------------------------------------------------- routing


def _under_budget(state: AgentState) -> bool:
    return state["iteration"] < state["budget"].max_iterations


def route_after_tests(
    state: AgentState,
) -> Literal["human_approval", "analyze_failure", "escalate"]:
    result = state["attempts"][-1].result
    if result is not None and result.passed:
        return "human_approval"
    return "analyze_failure" if _under_budget(state) else "escalate"


def route_after_approval(
    state: AgentState,
) -> Literal["commit_and_pr", "analyze_failure", "escalate"]:
    approval = state["approval"]
    if approval is not None and approval.approved:
        return "commit_and_pr"
    if approval is not None and approval.feedback and _under_budget(state):
        return "analyze_failure"
    return "escalate"


def build_mock_graph(checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(AgentState)
    for fn in (
        ingest_issue, inspect_repo, write_repro_test, run_repro_test, propose_fix,
        run_tests, analyze_failure, human_approval, commit_and_pr, escalate,
    ):  # fmt: skip
        g.add_node(fn.__name__, fn)

    g.add_edge(START, "ingest_issue")
    g.add_edge("ingest_issue", "inspect_repo")
    g.add_edge("inspect_repo", "write_repro_test")
    g.add_edge("write_repro_test", "run_repro_test")
    g.add_edge("run_repro_test", "propose_fix")  # scripted: the bug always reproduces
    g.add_edge("propose_fix", "run_tests")
    g.add_conditional_edges("run_tests", route_after_tests)
    g.add_edge("analyze_failure", "propose_fix")
    g.add_conditional_edges("human_approval", route_after_approval)
    g.add_edge("commit_and_pr", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())

