"""Unit tests for conditional edge routers, `Workspace` file tools, `LocalSandbox`,
and the full production `build_graph()` with an injected fake LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

from agent.graph import (
    build_graph,
    route_after_apply,
    route_after_approval,
    route_after_repro,
    route_after_tests,
)
from agent.state import (
    AgentState,
    AgentStatus,
    ApprovalDecision,
    FileEdit,
    FixAttempt,
    IssueInput,
    ReproTest,
    RunBudget,
    SandboxResult,
    initial_state,
)
from sandbox.local_runner import LocalSandbox, scrubbed_env
from tools.file_tools import (
    EditError,
    PathEscapeError,
    Workspace,
    create_repo_from_uploads,
    validate_repro_path,
)

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "fixtures" / "buggy_repo"


class ScriptedChatModel(BaseChatModel):
    """Minimal deterministic `BaseChatModel` returning pre-canned JSON responses."""

    responses: list[str]
    idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


# ------------------------------------------------------------------ pure routers


def test_route_after_repro_branches():
    base: AgentState = {"budget": RunBudget(max_repro_attempts=2), "repro_attempts": 1}
    assert (
        route_after_repro(
            {**base, "repro_test": ReproTest(path="test_a.py", content="x", confirmed_failing=True)}
        )
        == "propose_fix"
    )
    assert (
        route_after_repro(
            {
                **base,
                "repro_test": ReproTest(path="test_a.py", content="x", confirmed_failing=False),
            }
        )
        == "write_repro_test"
    )
    assert (
        route_after_repro(
            {
                **base,
                "repro_attempts": 2,
                "repro_test": ReproTest(path="test_a.py", content="x", confirmed_failing=False),
            }
        )
        == "escalate"
    )


def test_route_after_apply_skips_sandbox_on_patch_error():
    state: AgentState = {
        "budget": RunBudget(),
        "iteration": 1,
        "attempts": [FixAttempt(iteration=1, patch_error="search not found")],
    }
    assert route_after_apply(state) == "analyze_failure"
    state["attempts"] = [FixAttempt(iteration=1, patch_error=None)]
    assert route_after_apply(state) == "run_tests"


def test_route_after_tests_respects_budget_and_timeout():
    budget = RunBudget(max_iterations=8)
    passing = FixAttempt(iteration=1, result=SandboxResult(exit_code=0, stdout="ok", stderr=""))
    failing = FixAttempt(iteration=1, result=SandboxResult(exit_code=1, stdout="F", stderr=""))
    timed_out = FixAttempt(
        iteration=1, result=SandboxResult(exit_code=0, stdout="", stderr="", timed_out=True)
    )

    assert route_after_tests({"budget": budget, "iteration": 1, "attempts": [passing]}) == "human_approval"
    assert route_after_tests({"budget": budget, "iteration": 1, "attempts": [failing]}) == "analyze_failure"
    assert route_after_tests({"budget": budget, "iteration": 1, "attempts": [timed_out]}) == "analyze_failure"
    assert route_after_tests({"budget": budget, "iteration": 8, "attempts": [failing]}) == "escalate"


def test_route_after_approval_branches():
    budget = RunBudget(max_iterations=8)
    assert (
        route_after_approval(
            {"budget": budget, "iteration": 1, "approval": ApprovalDecision(approved=True)}
        )
        == "commit_and_pr"
    )
    assert (
        route_after_approval(
            {
                "budget": budget,
                "iteration": 1,
                "approval": ApprovalDecision(approved=False, feedback="add guard"),
            }
        )
        == "analyze_failure"
    )
    assert (
        route_after_approval(
            {"budget": budget, "iteration": 1, "approval": ApprovalDecision(approved=False)}
        )
        == "escalate"
    )


# --------------------------------------------------------- Workspace + LocalSandbox


def test_workspace_jails_paths_and_applies_edits():
    ws = Workspace.create(FIXTURE_REPO)
    try:
        with pytest.raises(PathEscapeError):
            ws.resolve("../outside.py")
        with pytest.raises(PathEscapeError):
            validate_repro_path("calc.py")

        ws.apply_edits(
            [
                FileEdit(
                    path="calc.py",
                    search="return sum(xs) / (len(xs) + 1)",
                    replace="return sum(xs) / len(xs)",
                )
            ]
        )
        assert "+    return sum(xs) / len(xs)" in ws.diff("calc.py")

        with pytest.raises(EditError):
            ws.apply_edits([FileEdit(path="calc.py", search="nonexistent line", replace="x")])

        ws.reset()
        assert ws.diff("calc.py") == ""

        zip_bytes = ws.export_zip_bytes()
        assert len(zip_bytes) > 100
    finally:
        ws.cleanup()


def test_parse_github_owner_repo():
    from tools.git_tools import parse_github_issue_url, parse_github_owner_repo

    assert parse_github_owner_repo("https://github.com/farazmalik180/SwiftSupport-Commerce.git") == (
        "farazmalik180",
        "SwiftSupport-Commerce",
    )
    assert parse_github_owner_repo("https://github.com/owner/repo") == ("owner", "repo")
    assert parse_github_owner_repo("not-a-url") is None

    assert parse_github_issue_url("https://github.com/farazmalik180/SwiftSupport-Commerce/issues/12") == (
        "farazmalik180",
        "SwiftSupport-Commerce",
        12,
    )
    assert parse_github_issue_url("https://github.com/owner/repo") is None


def test_create_repo_from_uploads_handles_zip_and_rejects_zip_slip():
    import io
    import shutil
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("myrepo-main/calc.py", "def add(a, b):\n    return a - b\n")
        zf.writestr("myrepo-main/sub/util.py", "X = 1\n")

    repo_dir = create_repo_from_uploads([("myrepo.zip", buf.getvalue())])
    try:
        assert (repo_dir / "calc.py").is_file()
        assert (repo_dir / "sub" / "util.py").is_file()
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)

    # Zip-slip attempt must raise PathEscapeError
    bad_buf = io.BytesIO()
    with zipfile.ZipFile(bad_buf, "w") as zf:
        zf.writestr("../../evil.py", "print('pwned')\n")
    with pytest.raises(PathEscapeError):
        create_repo_from_uploads([("evil.zip", bad_buf.getvalue())])


def test_local_sandbox_scrubs_secrets_and_runs_pytest(tmp_path: Path):
    env = scrubbed_env({"GROQ_API_KEY": "secret", "GITHUB_TOKEN": "tok", "SAFE_VAR": "1"})
    assert "GROQ_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["SAFE_VAR"] == "1"

    with LocalSandbox(default_timeout_s=15) as sb:
        sb.start(tmp_path)
        sb.write_file("test_sample.py", "def test_ok():\n    assert 2 + 2 == 4\n")
        res = sb.run_tests("pytest -q -p no:cacheprovider")
        assert res.passed, res.stdout + res.stderr


# ------------------------------------------------- Production graph end-to-end (fake LLM)


def test_production_graph_fixes_bug_and_pauses_at_approval():
    repro_json = json.dumps(
        {
            "path": "test_repro_bug.py",
            "content": (
                "from calc import average\n\n"
                "def test_average_repro():\n"
                "    assert average([2, 4, 6]) == 4.0\n"
            ),
        }
    )
    fix_json = json.dumps(
        {
            "rationale": "Remove off-by-one +1 in average() denominator.",
            "edits": [
                {
                    "path": "calc.py",
                    "search": "return sum(xs) / (len(xs) + 1)",
                    "replace": "return sum(xs) / len(xs)",
                }
            ],
        }
    )
    fake_llm = ScriptedChatModel(responses=[repro_json, fix_json])
    graph = build_graph()
    cfg = {
        "configurable": {
            "thread_id": "prod-e2e-1",
            "llm": fake_llm,
            "sandbox_backend": "local",
            "commit_in_git": False,
        }
    }
    issue = IssueInput(
        title="average() is off by one",
        body="average([2, 4, 6]) returns 3.0 instead of 4.0",
        repo_path=FIXTURE_REPO,
    )
    nodes = [
        n for chunk in graph.stream(initial_state(issue), cfg, stream_mode="updates") for n in chunk
    ]
    snap = graph.get_state(cfg)
    try:
        assert snap.next == ("human_approval",)
        assert "commit_and_pr" not in nodes
        assert snap.values["repro_test"].confirmed_failing is True
        assert snap.values["attempts"][-1].result.passed is True

        # Resume with approval
        after_nodes = [
            n
            for chunk in graph.stream(
                Command(resume={"approved": True, "reviewer": "tester"}),
                cfg,
                stream_mode="updates",
            )
            for n in chunk
        ]
        assert after_nodes[-1] == "commit_and_pr"
        final = graph.get_state(cfg).values
        assert final["status"] == AgentStatus.COMMITTED
        assert final["commit_sha"] is not None
    finally:
        ws_root = snap.values.get("workspace_root")
        if ws_root:
            Workspace.open(ws_root).cleanup()
