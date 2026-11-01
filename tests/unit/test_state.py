"""Tests for state models, reducers, and checkpoint serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agent.state import (
    AgentState,
    AgentStatus,
    ApprovalDecision,
    FilePatch,
    FixAttempt,
    IssueInput,
    RunBudget,
    SandboxResult,
    initial_state,
)


@pytest.fixture
def issue(tmp_path: Path) -> IssueInput:
    return IssueInput(title="average() is wrong", body="average([2,4,6]) != 4", repo_path=tmp_path)


# ------------------------------------------------------------------------ models


def test_budget_defaults_match_groq_guardrails():
    b = RunBudget()
    assert b.max_iterations == 8
    assert b.max_repro_attempts == 2


@pytest.mark.parametrize("n", [0, 9, -1])
def test_budget_caps_iterations_at_eight(n: int):
    with pytest.raises(ValidationError):
        RunBudget(max_iterations=n)


def test_budget_rejects_tool_output_larger_than_context():
    with pytest.raises(ValidationError):
        RunBudget(max_context_tokens=500, max_tool_output_chars=4000)


def test_models_forbid_unknown_fields(tmp_path: Path):
    with pytest.raises(ValidationError):
        IssueInput(title="t", repo_path=tmp_path, bogus=1)


def test_issue_requires_title(tmp_path: Path):
    with pytest.raises(ValidationError):
        IssueInput(title="", repo_path=tmp_path)


def test_fix_attempt_iteration_is_one_based():
    with pytest.raises(ValidationError):
        FixAttempt(iteration=0)


def test_file_patch_requires_diff():
    with pytest.raises(ValidationError):
        FilePatch(path="a.py", unified_diff="")


def test_approval_timestamp_is_timezone_aware():
    assert ApprovalDecision(approved=True).decided_at.tzinfo is not None


def test_agent_status_values_are_strings():
    assert AgentStatus.ESCALATED == "escalated"


def test_sandbox_result_is_the_docker_runner_model():
    from sandbox.docker_runner import SandboxResult as Original

    assert SandboxResult is Original


# ------------------------------------------------------------------ initial state


def test_initial_state_is_clean(issue: IssueInput):
    s = initial_state(issue)
    assert s["iteration"] == 0 and s["repro_attempts"] == 0
    assert s["attempts"] == [] and s["messages"] == []
    assert s["status"] is AgentStatus.INGESTED
    assert s["approval"] is None and s["commit_sha"] is None and s["pr_url"] is None
    assert s["budget"] == RunBudget()


def test_initial_state_accepts_custom_budget(issue: IssueInput):
    assert initial_state(issue, RunBudget(max_iterations=3))["budget"].max_iterations == 3


def test_models_roundtrip_through_json(issue: IssueInput):
    attempt = FixAttempt(
        iteration=1,
        rationale="divide by len",
        patches=[FilePatch(path="calc.py", unified_diff="--- a\n+++ b\n")],
        result=SandboxResult(exit_code=1, stdout="F", stderr="", duration_s=0.5),
    )
    dumped = json.loads(json.dumps(attempt.model_dump(mode="json")))
    assert FixAttempt.model_validate(dumped) == attempt

    issue_dumped = json.loads(issue.model_dump_json())
    assert IssueInput.model_validate(issue_dumped) == issue  # Path survives as str


# ---------------------------------------------------------- reducers + checkpointer


def _build_graph():
    def first(state: AgentState) -> dict:
        return {
            "attempts": [FixAttempt(iteration=1, rationale="first")],
            "messages": [AIMessage(content="one", id="m1")],
            "iteration": 1,
        }

    def second(state: AgentState) -> dict:
        return {
            "attempts": [FixAttempt(iteration=2, rationale="second")],
            "messages": [AIMessage(content="two", id="m2")],
            "iteration": state["iteration"] + 1,
            "status": AgentStatus.FIXING,
        }

    g = StateGraph(AgentState)
    g.add_node("first", first)
    g.add_node("second", second)
    g.add_edge(START, "first")
    g.add_edge("first", "second")
    g.add_edge("second", END)
    return g


def test_reducers_append_attempts_and_messages(issue: IssueInput):
    graph = _build_graph().compile()
    out = graph.invoke(initial_state(issue) | {"messages": [HumanMessage(content="hi", id="h")]})
    assert [a.rationale for a in out["attempts"]] == ["first", "second"]
    assert [m.content for m in out["messages"]] == ["hi", "one", "two"]
    assert out["iteration"] == 2
    assert out["status"] == AgentStatus.FIXING


def test_state_checkpoints_with_memory_saver(issue: IssueInput):
    graph = _build_graph().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke(initial_state(issue), cfg)

    snap = graph.get_state(cfg)
    assert snap.values["iteration"] == 2
    attempts = snap.values["attempts"]
    assert len(attempts) == 2 and all(isinstance(a, FixAttempt) for a in attempts)
    assert isinstance(snap.values["issue"], IssueInput)
    assert snap.values["issue"].repo_path == issue.repo_path
