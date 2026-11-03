"""Streamlit front-end for the Autonomous Bug-Fixing Agent.

Supports:
- **Live Agent (Groq LLM + Sandbox):** runs the real LangGraph pipeline against the built-in
  sample buggy repo, any local repository path, or any GitHub repository URL (shallow-cloned
  into an isolated temp workspace).
- **Scripted Demo Mode:** deterministic walkthrough with zero API calls.

Run:  streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

from agent.graph import build_graph
from agent.llm import AVAILABLE_GROQ_MODELS, load_dotenv_into_environ, resolve_groq_model
from agent.state import AgentState, FixAttempt, IssueInput, initial_state
from sandbox.docker_runner import DockerSandbox
from tools.file_tools import Workspace, WorkspaceError, clone_repo_to_temp
from ui.mock_graph import (
    DEMO_ISSUE_BODY,
    DEMO_ISSUE_TITLE,
    build_mock_graph,
)

load_dotenv_into_environ()

FIXTURE_REPO = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "buggy_repo"

STEP_LABELS = {
    "ingest_issue": "Ingest issue & prepare workspace",
    "inspect_repo": "Inspect repository & rank relevant files",
    "write_repro_test": "Write reproduction test (Groq LLM)",
    "run_repro_test": "Run reproduction test (expect failure)",
    "propose_fix": "Propose code fix (Groq LLM)",
    "apply_fix": "Apply search/replace edits & compute diff",
    "run_tests": "Run test suite in sandbox",
    "analyze_failure": "Analyze failure / reviewer feedback",
    "human_approval": "Human-in-the-loop approval",
    "commit_and_pr": "Commit approved patch",
    "escalate": "Escalate",
}


@st.cache_resource
def get_live_graph():
    return build_graph()


@st.cache_resource
def get_demo_graph():
    return build_mock_graph()


def _active_graph():
    return get_demo_graph() if st.session_state.get("run_mode") == "demo" else get_live_graph()


def _config() -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "delay": st.session_state.get("delay", 0.3),
            "groq_model": st.session_state.get("groq_model"),
            "groq_api_key": st.session_state.get("groq_api_key") or None,
            "sandbox_backend": st.session_state.get("sandbox_backend", "auto"),
            "commit_in_git": False,
        }
    }


# ------------------------------------------------------------------------ running


def advance(payload: AgentState | Command) -> None:
    """Stream the active graph until it pauses at `interrupt()` or finishes, then rerun."""
    graph = _active_graph()
    try:
        with st.status("Agent running...", expanded=True) as status:
            for chunk in graph.stream(payload, _config(), stream_mode="updates"):
                for node, update in chunk.items():
                    if node.startswith("__"):
                        continue
                    st.session_state.log.append((node, update or {}))
                    status.update(label=f"Ran: {STEP_LABELS.get(node, node)}")
                    st.write(f"✅ {STEP_LABELS.get(node, node)}")
            status.update(label="Agent paused or finished", state="complete", expanded=False)
    except Exception as exc:
        st.session_state.last_error = str(exc)
    st.rerun()


def start_run(
    *,
    mode: str,
    repo_source: str,
    repo_input: str,
    issue_title: str,
    issue_body: str,
) -> None:
    st.session_state.last_error = None
    st.session_state.run_mode = mode
    st.session_state.thread_id = f"{mode}-{uuid.uuid4().hex[:8]}"
    st.session_state.log = []

    if mode == "demo":
        issue = IssueInput(title=issue_title, body=issue_body, repo_path=Path("demo-repo"))
        advance(initial_state(issue))
        return

    try:
        if repo_source == "Built-in sample buggy repo":
            target_path = FIXTURE_REPO
        elif repo_source == "GitHub URL (shallow clone)":
            with st.spinner(f"Cloning {repo_input.strip()}..."):
                target_path = clone_repo_to_temp(repo_input)
        else:
            target_path = Path(repo_input.strip()).expanduser().resolve()
            if not target_path.is_dir():
                raise WorkspaceError(f"Local folder does not exist: {target_path}")
    except Exception as exc:
        st.session_state.last_error = str(exc)
        return

    issue = IssueInput(title=issue_title.strip(), body=issue_body.strip(), repo_path=target_path)
    advance(initial_state(issue))


# ---------------------------------------------------------------------- rendering


def render_event(node: str, update: dict[str, Any]) -> None:
    label = STEP_LABELS.get(node, node)
    attempts: list[FixAttempt] = update.get("attempts", [])

    if node == "propose_fix" and attempts:
        a = attempts[-1]
        with st.expander(f"🛠️ {label} (attempt {a.iteration})", expanded=False):
            st.write(a.rationale)
            if a.patches:
                st.code("\n".join(p.unified_diff for p in a.patches), language="diff")
            elif a.edits:
                for edit in a.edits:
                    st.caption(f"Edit `{edit.path}`")
                    st.code(f"- {edit.search}\n+ {edit.replace}", language="diff")
    elif node == "apply_fix" and attempts:
        a = attempts[-1]
        if a.patch_error:
            st.write(f"⚠️ {label}: {a.patch_error}")
        elif a.patches:
            with st.expander(f"🩹 {label} (attempt {a.iteration})", expanded=False):
                st.code("\n".join(p.unified_diff for p in a.patches), language="diff")
    elif node == "run_tests" and attempts:
        a = attempts[-1]
        icon = "🟢" if a.result and a.result.passed else "🔴"
        with st.expander(f"{icon} {label} (attempt {a.iteration})", expanded=False):
            out = ""
            if a.result:
                out = (a.result.stdout + "\n" + a.result.stderr).strip()
            st.code(out, language="text")
    elif node == "write_repro_test" and "repro_test" in update:
        with st.expander(f"🧪 {label} (`{update['repro_test'].path}`)", expanded=False):
            st.code(update["repro_test"].content, language="python")
    elif node == "inspect_repo" and "repo" in update:
        with st.expander(f"🔎 {label}", expanded=False):
            for path, src in update["repo"].relevant_files.items():
                st.caption(path)
                st.code(src, language="python")
    elif node == "run_repro_test":
        repro = update.get("repro_test")
        ok = repro.confirmed_failing if repro else True
        verdict = "bug reproduced (failed as expected)" if ok else "did not fail as expected"
        st.write(f"✅ {label}: {verdict}")
    elif node == "analyze_failure":
        diag = update.get("diagnosis") or "feeding the failure back into the next attempt"
        st.write(f"🔁 {label}: {diag}")
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
        advance(Command(resume={"approved": True, "reviewer": "user"}))
    if changes.button("✏️ Request changes", use_container_width=True):
        if feedback.strip():
            advance(
                Command(
                    resume={
                        "approved": False,
                        "feedback": feedback.strip(),
                        "reviewer": "user",
                    }
                )
            )
        else:
            st.warning("Add feedback so the agent knows what to change.")
    if reject.button("⛔ Reject", use_container_width=True):
        advance(Command(resume={"approved": False, "reviewer": "user"}))


def render_outcome(values: dict[str, Any]) -> None:
    status = str(values.get("status"))
    is_demo = st.session_state.get("run_mode") == "demo"
    if status == "committed":
        note = "Demo commit" if is_demo else "Approved commit"
        st.success(
            f"{note} `{values['commit_sha']}` on `{values['branch']}` "
            "(changes kept in the isolated workspace; download the patch below)."
        )
        attempts: list[FixAttempt] = values["attempts"]
        patch = "\n".join(p.unified_diff for p in attempts[-1].patches)
        if values.get("workspace_root"):
            try:
                patch = Workspace.open(values["workspace_root"]).full_diff() or patch
            except Exception:  # noqa: BLE001
                pass
        st.download_button("⬇️ Download patch", patch, file_name="fix.patch")
    elif status == "escalated":
        st.error(f"Escalated: {values.get('error')}")


# --------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="Autonomous Bug-Fixing Agent", page_icon="🐞", layout="wide")
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("log", [])
    st.session_state.setdefault("delay", 0.3)
    st.session_state.setdefault("run_mode", "demo")
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("groq_model", resolve_groq_model())
    st.session_state.setdefault("groq_api_key", "")
    st.session_state.setdefault("sandbox_backend", "auto")

    st.title("🐞 Autonomous Bug-Fixing Agent")

    docker_up = DockerSandbox.is_available()
    has_env_key = bool(os.environ.get("GROQ_API_KEY"))

    with st.sidebar:
        st.header("1. Execution Mode")
        mode_choice = st.radio(
            "Agent backend",
            options=["Live Groq Agent", "Scripted Demo (no API calls)"],
            index=0 if has_env_key else 1,
        )
        is_demo_choice = mode_choice.startswith("Scripted")

        st.header("2. Repository")
        repo_source = st.selectbox(
            "Repository source",
            options=[
                "Built-in sample buggy repo",
                "Local folder path",
                "GitHub URL (shallow clone)",
            ],
            disabled=is_demo_choice,
        )
        repo_input = ""
        if repo_source == "Local folder path":
            repo_input = st.text_input(
                "Local repository path",
                value=str(FIXTURE_REPO),
                disabled=is_demo_choice,
            )
        elif repo_source == "GitHub URL (shallow clone)":
            repo_input = st.text_input(
                "GitHub repository URL",
                placeholder="https://github.com/owner/repo.git",
                disabled=is_demo_choice,
            )
        else:
            st.caption(f"Using `{FIXTURE_REPO.name}` (`calc.py` with off-by-one bug)")

        st.header("3. Bug Report / Issue")
        issue_title = st.text_input(
            "Title",
            value=DEMO_ISSUE_TITLE,
            disabled=is_demo_choice,
        )
        issue_body = st.text_area(
            "Description",
            value=DEMO_ISSUE_BODY,
            height=110,
            disabled=is_demo_choice,
        )

        with st.expander("⚙️ LLM & Sandbox Settings", expanded=False):
            model_options = list(AVAILABLE_GROQ_MODELS)
            default_idx = (
                model_options.index(st.session_state.groq_model)
                if st.session_state.groq_model in model_options
                else 0
            )
            st.session_state.groq_model = st.selectbox(
                "Groq Model",
                options=model_options,
                index=default_idx,
            )
            st.session_state.groq_api_key = st.text_input(
                "Groq API Key override (optional if set in .env)",
                type="password",
                placeholder="Using GROQ_API_KEY from .env" if has_env_key else "gsk_...",
            )
            st.session_state.sandbox_backend = st.selectbox(
                "Sandbox backend",
                options=["auto", "local", "docker"],
                index=0,
                help=f"Docker daemon reachable: {docker_up}",
            )
            st.slider("Demo step delay (s)", 0.0, 1.5, key="delay", step=0.1)

        if st.button("▶️ Start run", type="primary", use_container_width=True):
            start_run(
                mode="demo" if is_demo_choice else "live",
                repo_source=repo_source,
                repo_input=repo_input,
                issue_title=issue_title,
                issue_body=issue_body,
            )

    if is_demo_choice:
        st.warning(
            "**Demo mode selected.** Runs the deterministic scripted agent (no LLM calls, no "
            "commits). Switch the sidebar to **Live Groq Agent** to run with Groq + sandbox."
        )
    else:
        sandbox_note = (
            "Docker container sandbox"
            if docker_up and st.session_state.sandbox_backend != "local"
            else "Local subprocess sandbox (secrets scrubbed, not container-isolated)"
        )
        st.info(
            f"**Live Groq mode (`{st.session_state.groq_model}`).** "
            f"Tests execute in: **{sandbox_note}**. Nothing is committed without your approval."
        )

    if st.session_state.last_error:
        st.error(f"Run error: {st.session_state.last_error}")

    if st.session_state.thread_id is None:
        st.info("Configure your repository & issue in the sidebar and press **Start run**.")
        return

    snap = _active_graph().get_state(_config())
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
