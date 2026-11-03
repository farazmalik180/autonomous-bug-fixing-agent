"""Agent state schema: Pydantic v2 models + the LangGraph ``AgentState`` TypedDict.

Nested objects are Pydantic models (validated, JSON-serializable for checkpointers);
the top-level state is a TypedDict because LangGraph needs per-key reducers.
See SPEC.md section 2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sandbox.docker_runner import SandboxResult

__all__ = [
    "AgentState",
    "AgentStatus",
    "ApprovalDecision",
    "FilePatch",
    "FixAttempt",
    "IssueInput",
    "RepoContext",
    "ReproTest",
    "RunBudget",
    "SandboxResult",
    "initial_state",
    "merge_attempts",
]


class AgentStatus(StrEnum):
    INGESTED = "ingested"
    INSPECTED = "inspected"
    REPRO_WRITTEN = "repro_written"
    REPRO_CONFIRMED = "repro_confirmed"
    FIXING = "fixing"
    TESTS_PASSING = "tests_passing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    COMMITTED = "committed"
    ESCALATED = "escalated"
    REJECTED = "rejected"


class _Model(BaseModel):
    """Base: reject unknown fields so LLM/node typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class IssueInput(_Model):
    title: str = Field(min_length=1)
    body: str = ""
    url: str | None = None
    number: int | None = Field(default=None, ge=1)
    repo_path: Path


class RepoContext(_Model):
    tree: str = ""  # truncated directory listing
    relevant_files: dict[str, str] = Field(default_factory=dict)  # path -> snippet
    test_command: str = "pytest -x -q"
    language: str = "python"


class ReproTest(_Model):
    path: str = Field(min_length=1)  # repo-relative
    content: str
    framework: Literal["pytest"] = "pytest"
    confirmed_failing: bool = False


class FilePatch(_Model):
    path: str = Field(min_length=1)  # repo-relative
    unified_diff: str = Field(min_length=1)


class FixAttempt(_Model):
    iteration: int = Field(ge=1)
    rationale: str = ""
    patches: list[FilePatch] = Field(default_factory=list)
    patch_error: str | None = None  # set by apply_fix when a diff fails to apply
    result: SandboxResult | None = None  # set by run_tests


class ApprovalDecision(_Model):
    approved: bool
    feedback: str | None = None
    reviewer: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunBudget(_Model):
    """Hard limits that keep a run inside Groq's free tier."""

    max_iterations: int = Field(default=8, ge=1, le=8)
    max_repro_attempts: int = Field(default=2, ge=1)
    max_context_tokens: int = Field(default=6000, ge=500)
    max_tool_output_chars: int = Field(default=4000, ge=200)

    @model_validator(mode="after")
    def _output_fits_context(self) -> RunBudget:
        # ~4 chars/token: a single tool output must not eat the whole window.
        if self.max_tool_output_chars // 4 >= self.max_context_tokens:
            raise ValueError("max_tool_output_chars is too large for max_context_tokens")
        return self


def merge_attempts(existing: list[FixAttempt] | None, new: list[FixAttempt] | None) -> list[FixAttempt]:
    """Reducer for ``attempts``: upsert by ``iteration``, ordered by iteration.

    Lets a later node (e.g. ``run_tests``) return an updated copy of an attempt
    with its sandbox result attached, while earlier attempts stay in the history.
    """
    merged = {a.iteration: a for a in existing or []}
    merged.update({a.iteration: a for a in new or []})
    return [merged[k] for k in sorted(merged)]


class AgentState(TypedDict, total=False):
    """Graph state. Nodes return partial updates and never mutate their input."""

    issue: IssueInput
    repo: RepoContext
    repro_test: ReproTest
    attempts: Annotated[list[FixAttempt], merge_attempts]  # audit trail, upserted by iteration
    messages: Annotated[list[AnyMessage], add_messages]  # LLM conversation (pruned)
    iteration: int  # fix iterations consumed, 0..budget.max_iterations
    repro_attempts: int  # repro-test generations consumed
    budget: RunBudget
    status: AgentStatus
    approval: ApprovalDecision | None
    branch: str | None
    commit_sha: str | None
    pr_url: str | None
    error: str | None  # terminal failure reason, set by escalate


def initial_state(issue: IssueInput, budget: RunBudget | None = None) -> AgentState:
    """Build the starting state for a run."""
    return AgentState(
        issue=issue,
        attempts=[],
        messages=[],
        iteration=0,
        repro_attempts=0,
        budget=budget or RunBudget(),
        status=AgentStatus.INGESTED,
        approval=None,
        branch=None,
        commit_sha=None,
        pr_url=None,
        error=None,
    )
