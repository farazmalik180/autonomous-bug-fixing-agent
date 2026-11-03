"""Tests for the scripted demo graph: control flow, HITL pause, and the attempts reducer."""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.types import Command

from agent.state import (
    AgentStatus,
    FixAttempt,
    IssueInput,
    SandboxResult,
    initial_state,
    merge_attempts,
)
from ui.mock_graph import DEMO_ISSUE_BODY, DEMO_ISSUE_TITLE, build_mock_graph

CFG = {"configurable": {"thread_id": "t", "delay": 0}}


@pytest.fixture
def graph():
    return build_mock_graph()


@pytest.fixture
def start_state():
    return initial_state(
        IssueInput(title=DEMO_ISSUE_TITLE, body=DEMO_ISSUE_BODY, repo_path=Path("demo"))
    )


def _nodes(graph, payload, cfg=CFG) -> list[str]:
    return [n for chunk in graph.stream(payload, cfg, stream_mode="updates") for n in chunk]


def _run_to_approval(graph, start_state):
    nodes = _nodes(graph, start_state)
    return nodes, graph.get_state(CFG)


# ----------------------------------------------------------------------- reducer


def _attempt(i: int, rationale: str = "", result: SandboxResult | None = None) -> FixAttempt:
    return FixAttempt(iteration=i, rationale=rationale, result=result)


def test_merge_attempts_appends_new_iterations_in_order():
    merged = merge_attempts([_attempt(1)], [_attempt(2)])
    assert [a.iteration for a in merged] == [1, 2]


def test_merge_attempts_upserts_same_iteration_keeping_history():
    res = SandboxResult(exit_code=1, stdout="F", stderr="")
    merged = merge_attempts([_attempt(1, "a"), _attempt(2, "b")], [_attempt(2, "b", res)])
    assert [a.iteration for a in merged] == [1, 2]
    assert merged[1].result == res and merged[0].result is None


def test_merge_attempts_handles_none():
    assert merge_attempts(None, None) == []


# ------------------------------------------------------------------- control flow


def test_graph_pauses_before_commit_and_a_failed_first_attempt_is_retried(graph, start_state):
    nodes, snap = _run_to_approval(graph, start_state)

    assert snap.next == ("human_approval",)  # paused, waiting for a human
    assert "commit_and_pr" not in nodes
    assert nodes.count("propose_fix") == 2 and nodes.count("analyze_failure") == 1
    assert snap.values["iteration"] == 2
    attempts = snap.values["attempts"]
    assert [a.result.passed for a in attempts] == [False, True]  # history kept, results attached
    assert snap.values.get("commit_sha") is None


def test_interrupt_payload_contains_review_material(graph, start_state):
    _run_to_approval(graph, start_state)
    payload = graph.get_state(CFG).tasks[0].interrupts[0].value
    assert "+    return sum(xs) / len(xs)" in payload["diff"]
    assert payload["test_output"].startswith("2 passed")
    assert "def test_average_of_three" in payload["repro_test"]


def test_approve_commits(graph, start_state):
    _run_to_approval(graph, start_state)
    nodes = _nodes(graph, Command(resume={"approved": True, "reviewer": "me"}))

    assert nodes[-1] == "commit_and_pr"
    final = graph.get_state(CFG)
    assert final.next == ()
    assert final.values["status"] == AgentStatus.COMMITTED
    assert len(final.values["commit_sha"]) == 7
    assert final.values["approval"].reviewer == "me"


def test_request_changes_loops_back_then_pauses_again(graph, start_state):
    _run_to_approval(graph, start_state)
    nodes = _nodes(graph, Command(resume={"approved": False, "feedback": "handle empty list"}))

    assert "analyze_failure" in nodes and "propose_fix" in nodes
    assert "commit_and_pr" not in nodes
    snap = graph.get_state(CFG)
    assert snap.next == ("human_approval",)  # a second approval is required
    assert snap.values["iteration"] == 3
    assert "ValueError" in snap.values["attempts"][-1].patches[0].unified_diff


def test_reject_without_feedback_escalates_and_never_commits(graph, start_state):
    _run_to_approval(graph, start_state)
    nodes = _nodes(graph, Command(resume={"approved": False}))

    assert nodes[-1] == "escalate" and "commit_and_pr" not in nodes
    final = graph.get_state(CFG).values
    assert final["status"] == AgentStatus.ESCALATED
    assert final["commit_sha"] is None and "rejected" in final["error"].lower()


def test_repeated_change_requests_stop_at_iteration_budget(graph, start_state):
    _run_to_approval(graph, start_state)
    for _ in range(20):
        if not graph.get_state(CFG).next:
            break
        _nodes(graph, Command(resume={"approved": False, "feedback": "again"}))

    final = graph.get_state(CFG)
    assert final.next == ()
    assert final.values["iteration"] <= final.values["budget"].max_iterations == 8
    assert final.values["status"] == AgentStatus.ESCALATED
    assert final.values["commit_sha"] is None

