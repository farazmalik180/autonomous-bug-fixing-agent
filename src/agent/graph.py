"""LangGraph assembly, pure conditional routers, and checkpointer factory (`src/agent/graph.py`)."""

from __future__ import annotations

import os
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    analyze_failure,
    apply_fix,
    commit_and_pr,
    escalate,
    human_approval,
    ingest_issue,
    inspect_repo,
    propose_fix,
    run_repro_test,
    run_tests,
    write_repro_test,
)
from agent.state import AgentState


def route_after_repro(
    state: AgentState,
) -> Literal["propose_fix", "write_repro_test", "escalate"]:
    """Route after running the reproduction test."""
    if state["repro_test"].confirmed_failing:
        return "propose_fix"
    if state.get("repro_attempts", 0) < state["budget"].max_repro_attempts:
        return "write_repro_test"
    return "escalate"


def route_after_apply(
    state: AgentState,
) -> Literal["run_tests", "analyze_failure", "escalate"]:
    """Route after applying edits: skip test execution if the search/replace failed."""
    latest = state["attempts"][-1]
    if not latest.patch_error:
        return "run_tests"
    if state.get("iteration", 0) < state["budget"].max_iterations:
        return "analyze_failure"
    return "escalate"


def route_after_tests(
    state: AgentState,
) -> Literal["human_approval", "analyze_failure", "escalate"]:
    """Route after running the test suite in the sandbox."""
    latest = state["attempts"][-1]
    if latest.result is not None and latest.result.passed:
        return "human_approval"
    if state.get("iteration", 0) < state["budget"].max_iterations:
        return "analyze_failure"
    return "escalate"


def route_after_approval(
    state: AgentState,
) -> Literal["commit_and_pr", "analyze_failure", "escalate"]:
    """Route after the human-in-the-loop interrupt."""
    approval = state.get("approval")
    if approval is not None and approval.approved:
        return "commit_and_pr"
    if (
        approval is not None
        and approval.feedback
        and state.get("iteration", 0) < state["budget"].max_iterations
    ):
        return "analyze_failure"
    return "escalate"


def get_checkpointer() -> BaseCheckpointSaver:
    """Return `PostgresSaver` if `CHECKPOINTER=postgres`, else `MemorySaver`."""
    kind = os.environ.get("CHECKPOINTER", "memory").strip().lower()
    if kind == "postgres":
        conn_str = os.environ.get("DATABASE_URL", "").strip()
        if not conn_str:
            raise RuntimeError("CHECKPOINTER=postgres requires DATABASE_URL to be set.")
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-not-found]

        saver = PostgresSaver.from_conn_string(conn_str)
        saver.setup()
        return saver
    return MemorySaver()


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the production bug-fixing `StateGraph` with HITL interrupt support."""
    g = StateGraph(AgentState)
    for fn in (
        ingest_issue,
        inspect_repo,
        write_repro_test,
        run_repro_test,
        propose_fix,
        apply_fix,
        run_tests,
        analyze_failure,
        human_approval,
        commit_and_pr,
        escalate,
    ):
        g.add_node(fn.__name__, fn)

    g.add_edge(START, "ingest_issue")
    g.add_edge("ingest_issue", "inspect_repo")
    g.add_edge("inspect_repo", "write_repro_test")
    g.add_edge("write_repro_test", "run_repro_test")
    g.add_conditional_edges("run_repro_test", route_after_repro)
    g.add_edge("propose_fix", "apply_fix")
    g.add_conditional_edges("apply_fix", route_after_apply)
    g.add_conditional_edges("run_tests", route_after_tests)
    g.add_edge("analyze_failure", "propose_fix")
    g.add_conditional_edges("human_approval", route_after_approval)
    g.add_edge("commit_and_pr", END)
    g.add_edge("escalate", END)

    return g.compile(checkpointer=checkpointer or get_checkpointer())
