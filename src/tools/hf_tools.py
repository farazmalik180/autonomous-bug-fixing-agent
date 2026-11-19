"""Hugging Face Hub Dataset exporter (`src/tools/hf_tools.py`).

Publishes completed or audited bug-fixing runs to a public Hugging Face Dataset repository
(default: `MF180/bugfix-agent-benchmarks`) in SWE-Bench-style JSON format via the Hugging Face
Hub HTTP API (`httpx`).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from agent.llm import load_dotenv_into_environ

logger = logging.getLogger(__name__)

DEFAULT_HF_DATASET_REPO = "MF180/bugfix-agent-benchmarks"


def default_hf_token() -> str:
    load_dotenv_into_environ()
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    ).strip()


def build_benchmark_record(run_id: str, values: dict[str, Any]) -> dict[str, Any]:
    """Build a structured SWE-Bench-style dictionary from an `AgentState` snapshot."""
    issue = values.get("issue")
    repo_ctx = values.get("repo")
    repro = values.get("repro_test")
    attempts = values.get("attempts") or []

    findings_list = []
    if repo_ctx and getattr(repo_ctx, "findings", None):
        for f in repo_ctx.findings:
            findings_list.append(
                {
                    "severity": f.severity,
                    "file_path": f.file_path,
                    "title": f.title,
                    "description": f.description,
                    "suggested_fix": f.suggested_fix,
                }
            )

    latest_diff = ""
    latest_rationale = ""
    if attempts:
        last = attempts[-1]
        latest_rationale = last.rationale
        latest_diff = "\n".join(p.unified_diff for p in last.patches)

    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "repo": str(getattr(issue, "repo_path", "unknown")),
        "issue_title": str(getattr(issue, "title", "")),
        "issue_body": str(getattr(issue, "body", "")),
        "status": str(values.get("status", "unknown")),
        "iterations": int(values.get("iteration", 0)),
        "findings_count": len(findings_list),
        "findings": findings_list,
        "repro_test_path": getattr(repro, "path", ""),
        "repro_test": getattr(repro, "content", ""),
        "fix_rationale": latest_rationale,
        "patch_unified_diff": latest_diff,
        "commit_sha": values.get("commit_sha"),
    }


def export_run_to_hf_dataset(
    run_id: str,
    values: dict[str, Any],
    *,
    dataset_repo: str | None = None,
    hf_token: str | None = None,
    http_client: Any | None = None,
) -> str | None:
    """Upload a run's benchmark JSON record to a Hugging Face Dataset repo and return its URL."""
    token = (hf_token or default_hf_token()).strip()
    if not token:
        return None

    repo_id = (
        dataset_repo
        or os.environ.get("HF_DATASET_REPO")
        or DEFAULT_HF_DATASET_REPO
    ).strip()
    owner, name = repo_id.split("/", 1) if "/" in repo_id else ("MF180", repo_id)
    full_repo = f"{owner}/{name}"

    record = build_benchmark_record(run_id, values)
    encoded_json = base64.b64encode(json.dumps(record, indent=2).encode("utf-8")).decode("ascii")

    ndjson_lines = [
        json.dumps(
            {
                "key": "header",
                "value": {"summary": f"Add bugfix benchmark run {run_id}"},
            }
        ),
        json.dumps(
            {
                "key": "file",
                "value": {
                    "content": encoded_json,
                    "path": f"runs/{run_id}.json",
                    "encoding": "base64",
                },
            }
        ),
    ]
    ndjson_body = "\n".join(ndjson_lines) + "\n"

    headers = {"Authorization": f"Bearer {token}"}
    close_client = False
    if http_client is None:
        http_client = httpx.Client(timeout=25.0)
        close_client = True

    try:
        # Ensure dataset repo exists
        http_client.post(
            "https://huggingface.co/api/repos/create",
            headers=headers,
            json={"type": "dataset", "name": name, "private": False},
        )
        commit_resp = http_client.post(
            f"https://huggingface.co/api/datasets/{full_repo}/commit/main",
            headers={**headers, "Content-Type": "application/x-ndjson"},
            content=ndjson_body,
        )
        if commit_resp.status_code not in (200, 201):
            logger.warning("HF dataset commit returned %s: %s", commit_resp.status_code, commit_resp.text[:200])
            return None
        return f"https://huggingface.co/datasets/{full_repo}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF dataset export failed: %s", exc)
        return None
    finally:
        if close_client:
            http_client.close()
