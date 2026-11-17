"""AWS Lambda Function URL handler for GitHub Issue webhooks (`deploy/lambda_webhook.py`).

Receives a GitHub `issues` webhook (or a manual JSON POST `{"repo_url": "https://github.com/owner/repo"}`),
logs the queued scan into the `BugFixAgentRuns` Amazon DynamoDB table, and returns the live
EC2 Bug-Fixing Agent URL (`http://98.89.45.201`) to inspect and approve the fix.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import boto3  # type: ignore[import-untyped]


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    region = os.environ.get("AWS_REGION", "us-east-1")
    table_name = os.environ.get("AWS_DYNAMODB_TABLE", "BugFixAgentRuns")
    agent_url = os.environ.get("AGENT_WEB_URL", "http://98.89.45.201")

    raw_body = event.get("body") or "{}"
    try:
        payload = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except json.JSONDecodeError:
        payload = {}

    # Support both GitHub Issues webhook payloads and simple {"repo_url": "..."} JSON
    issue = payload.get("issue") or {}
    repository = payload.get("repository") or {}
    repo_url = (
        payload.get("repo_url")
        or repository.get("html_url")
        or "https://github.com/farazmalik180/autonomous-bug-fixing-agent"
    )
    issue_title = (
        payload.get("issue_title")
        or issue.get("title")
        or f"Webhook scan for {repo_url}"
    )

    run_id = f"webhook-{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(UTC).isoformat()

    saved = False
    try:
        ddb = boto3.client("dynamodb", region_name=region)
        ddb.put_item(
            TableName=table_name,
            Item={
                "run_id": {"S": run_id},
                "created_at": {"S": created_at},
                "repo": {"S": str(repo_url)},
                "issue_title": {"S": str(issue_title)},
                "status": {"S": "queued_via_lambda_webhook"},
                "iterations": {"N": "0"},
                "findings_count": {"N": "1"},
            },
        )
        saved = True
    except Exception as exc:  # noqa: BLE001
        saved = False
        error_msg = str(exc)
    else:
        error_msg = ""

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "ok": True,
                "service": "Autonomous Bug-Fixing Agent — AWS Lambda Webhook",
                "run_id": run_id,
                "repo_url": repo_url,
                "issue_title": issue_title,
                "dynamodb_logged": saved,
                "agent_dashboard_url": agent_url,
                "error": error_msg or None,
            },
            indent=2,
        ),
    }
