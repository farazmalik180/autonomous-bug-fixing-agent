"""Streamlit front-end for the bug-fixing agent (DEMO MODE).

Drives ``ui.mock_graph`` -- a scripted agent with no LLM and no Docker -- through the same
LangGraph pause/resume mechanism the real agent uses. Nothing is ever committed.

Run:  streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

from agent.state import AgentState, FixAttempt, IssueInput, initial_state
from ui.mock_graph import (
    DEMO_ISSUE_BODY,
    DEMO_ISSUE_TITLE,
    build_mock_graph,
)

STEP_LABELS = {
    "ingest_issue": "Ingest issue",
    "inspect_repo": "Inspect repository",
    "write_repro_test": "Write reproduction test",
    "run_repro_test": "Run reproduction test (expect failure)",
    "propose_fix": "Propose fix",
    "run_tests": "Run tests in sandbox",
    "analyze_failure": "Analyze failure",
    "human_approval": "Human approval",
    "commit_and_pr": "Commit",
    "escalate": "Escalate",
}


@st.cache_resource
def get_graph():
    """One graph + MemorySaver per server process; runs are keyed by thread_id."""
    return build_mock_graph()


def _config() -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "delay": st.session_state.delay,
        }
    }


# ------------------------------------------------------------------------ running


def advance(payload: AgentState | Command) -> None:
    """Stream the graph until it finishes or pauses, recording each step, then rerun."""
    graph = get_graph()
    with st.status("Agent running...", expanded=True) as status:
        for chunk in graph.stream(payload, _config(), stream_mode="updates"):
            for node, update in chunk.items():
                if node.startswith("__"):
                    continue
                st.session_state.log.append((node, update or {}))
                status.update(label=f"Ran: {STEP_LABELS.get(node, node)}")
                st.write(f"✅ {STEP_LABELS.get(node, node)}")
        status.update(label="Agent paused or finished", state="complete", expanded=False)
    st.rerun()


def start_run() -> None:
    st.session_state.thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    st.session_state.log = []
    issue = IssueInput(title=DEMO_ISSUE_TITLE, body=DEMO_ISSUE_BODY, repo_path=Path("demo-repo"))
    advance(initial_state(issue))


# ---------------------------------------------------------------------- rendering


def render_event(node: str, update: dict[str, Any]) -> None:
    label = STEP_LABELS.get(node, node)
    attempts: list[FixAttempt] = update.get("attempts", [])

    if node == "propose_fix" and attempts:
        a = attempts[-1]
        with st.expander(f"🛠️ {label} (attempt {a.iteration})", expanded=False):
            st.write(a.rationale)
            st.code("\n".join(p.unified_diff for p in a.patches), language="diff")
    elif node == "run_tests" and attempts:
        a = attempts[-1]
        icon = "🟢" if a.result and a.result.passed else "🔴"
        with st.expander(f"{icon} {label} (attempt {a.iteration})", expanded=False):
            st.code(a.result.stdout if a.result else "", language="text")
    elif node == "write_repro_test" and "repro_test" in update:
        with st.expander(f"🧪 {label}", expanded=False):
            st.code(update["repro_test"].content, language="python")
    elif node == "inspect_repo" and "repo" in update:
        with st.expander(f"🔎 {label}", expanded=False):
            for path, src in update["repo"].relevant_files.items():
                st.caption(path)
                st.code(src, language="python")
    elif node == "run_repro_test":
        st.write(f"✅ {label}: bug reproduced")
    elif node == "analyze_failure":
        st.write(f"🔁 {label}: feeding the failure back into the next attempt")
    elif node == "human_approval":
        approval = update.get("approval")
        verdict = "approved" if approval and approval.approved else "changes requested / rejected"
        st.write(f"🧑 {label}: {verdict}")
        if approval and approval.feedback:
            st.caption(f"Feedback: {approval.feedback}")
    else:
        st.write(f"✅ {label}")


def render_approval(payload: dict[str, Any]) -> None:
    st.subheader("⏸️ Waiting for your approval")
    st.caption(
        "The agent is paused by a LangGraph `interrupt()`. "
        "Nothing is committed until you approve."
    )
    st.markdown(f"**Fix proposed (iteration {payload['iteration']}):** {payload['rationale']}")
    st.code(payload["diff"], language="diff")
    st.markdown("**Test results**")
    st.code(payload["test_output"], language="text")
    with st.expander("Reproduction test"):
        st.code(payload["repro_test"], language="python")

    feedback = st.text_area(
        "Feedback for the agent (required to request changes)",
        key=f"fb-{payload['iteration']}",
        placeholder="e.g. handle the empty-list case",
    )
    approve, changes, reject = st.columns(3)
    if approve.button("✅ Approve", type="primary", use_container_width=True):
        advance(Command(resume={"approved": True, "reviewer": "demo-user"}))
    if changes.button("✏️ Request changes", use_container_width=True):
        if feedback.strip():
            advance(
                Command(resume={"approved": False, "feedback": feedback.strip(),
                                "reviewer": "demo-user"})
            )  # fmt: skip
        else:
            st.warning("Add feedback so the agent knows what to change.")
    if reject.button("⛔ Reject", use_container_width=True):
        advance(Command(resume={"approved": False, "reviewer": "demo-user"}))


def render_outcome(values: dict[str, Any]) -> None:
    status = str(values.get("status"))
    if status == "committed":
        st.success(
            f"Approved. Demo commit `{values['commit_sha']}` on `{values['branch']}` "
            "(simulated: nothing was written to any repository)."
        )
        attempts: list[FixAttempt] = values["attempts"]
        patch = "\n".join(p.unified_diff for p in attempts[-1].patches)
        st.download_button("⬇️ Download patch", patch, file_name="fix.patch")
    elif status == "escalated":
        st.error(f"Escalated: {values.get('error')}")


# --------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="Autonomous Bug-Fixing Agent", page_icon="🐞", layout="wide")
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("log", [])
    st.session_state.setdefault("delay", 0.6)

    st.title("🐞 Autonomous Bug-Fixing Agent")
    st.warning(
        "**Demo mode.** This is a scripted agent: no LLM, no Docker, nothing is committed. "
        "It shows the real workflow, the pause-for-approval mechanism and the retry loop. "
        "The real agent is still under construction."
    )

    with st.sidebar:
        st.header("Issue")
        st.text_input("Title", DEMO_ISSUE_TITLE, disabled=True)
        st.text_area("Description", DEMO_ISSUE_BODY, disabled=True, height=120)
        st.caption("The scripted scenario is fixed; these fields are read-only.")
        st.slider("Step delay (s)", 0.0, 1.5, key="delay", step=0.1)
        if st.button("▶️ Start run", type="primary", use_container_width=True):
            start_run()

    if st.session_state.thread_id is None:
        st.info("Press **Start run** in the sidebar to watch the agent work on the sample bug.")
        return

    snap = get_graph().get_state(_config())
    values = snap.values
    left, right = st.columns([3, 1])
    left.subheader("Timeline")
    right.metric("Fix iterations", f"{values.get('iteration', 0)} / 8")

    for node, update in st.session_state.log:
        render_event(node, update)

    if snap.next:  # paused at human_approval
        render_approval(snap.tasks[0].interrupts[0].value)
    else:
        render_outcome(values)


main()

