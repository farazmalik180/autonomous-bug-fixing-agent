"""AWS Free-Tier cloud integrations: Amazon S3 report archiving & Amazon DynamoDB run history.

Designed to work seamlessly on AWS EC2 (via IAM Instance Profile) or locally (via environment
variables), while degrading gracefully to a no-op when AWS is not configured so unit tests
and offline demo runs require zero cloud credentials.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudArchiveResult:
    s3_bucket: str | None = None
    s3_pdf_key: str | None = None
    s3_pdf_url: str | None = None
    s3_zip_url: str | None = None
    dynamodb_saved: bool = False
    sns_published: bool = False
    cloudwatch_published: bool = False
    error: str | None = None


def is_aws_configured() -> bool:
    """Return True if an S3 bucket, DynamoDB table, or SNS topic is configured in the environment."""
    return bool(
        os.environ.get("AWS_S3_BUCKET", "").strip()
        or os.environ.get("AWS_DYNAMODB_TABLE", "").strip()
        or os.environ.get("AWS_SNS_TOPIC_ARN", "").strip()
    )


def _aws_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    ).strip()


def upload_bytes_to_s3(
    data: bytes,
    key: str,
    content_type: str,
    *,
    bucket: str | None = None,
    expires_in: int = 604800,
    s3_client: Any | None = None,
) -> str | None:
    """Upload raw bytes to Amazon S3 and return a 7-day pre-signed download URL."""
    target_bucket = (bucket or os.environ.get("AWS_S3_BUCKET") or "").strip()
    if not target_bucket:
        return None

    if s3_client is None:
        import boto3  # type: ignore[import-untyped]

        s3_client = boto3.client("s3", region_name=_aws_region())

    s3_client.put_object(
        Bucket=target_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    url: str = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": target_bucket, "Key": key},
        ExpiresIn=expires_in,
    )
    return url


def save_run_to_dynamodb(
    run_id: str,
    values: dict[str, Any],
    *,
    s3_pdf_url: str | None = None,
    s3_zip_url: str | None = None,
    table_name: str | None = None,
    ddb_client: Any | None = None,
) -> bool:
    """Persist a summary record of an agent run into Amazon DynamoDB (`BugFixAgentRuns`)."""
    target_table = (table_name or os.environ.get("AWS_DYNAMODB_TABLE") or "").strip()
    if not target_table:
        return False

    if ddb_client is None:
        import boto3  # type: ignore[import-untyped]

        ddb_client = boto3.client("dynamodb", region_name=_aws_region())

    issue = values.get("issue")
    repo_ctx = values.get("repo")
    findings = getattr(repo_ctx, "findings", []) if repo_ctx else []
    repo_label = str(getattr(issue, "repo_path", "unknown"))
    title_label = str(getattr(issue, "title", "Repository Scan"))

    item: dict[str, dict[str, str]] = {
        "run_id": {"S": run_id},
        "created_at": {"S": datetime.now(UTC).isoformat()},
        "repo": {"S": repo_label},
        "issue_title": {"S": title_label},
        "status": {"S": str(values.get("status", "unknown"))},
        "iterations": {"N": str(int(values.get("iteration", 0)))},
        "findings_count": {"N": str(len(findings))},
    }
    if values.get("commit_sha"):
        item["commit_sha"] = {"S": str(values["commit_sha"])}
    if s3_pdf_url:
        item["s3_pdf_url"] = {"S": s3_pdf_url}
    if s3_zip_url:
        item["s3_zip_url"] = {"S": s3_zip_url}

    ddb_client.put_item(TableName=target_table, Item=item)
    return True


def list_runs_from_dynamodb(
    limit: int = 10,
    *,
    table_name: str | None = None,
    ddb_client: Any | None = None,
) -> list[dict[str, str]]:
    """Fetch recent agent runs from Amazon DynamoDB, sorted newest-first."""
    target_table = (table_name or os.environ.get("AWS_DYNAMODB_TABLE") or "").strip()
    if not target_table:
        return []

    try:
        if ddb_client is None:
            import boto3  # type: ignore[import-untyped]

            ddb_client = boto3.client("dynamodb", region_name=_aws_region())

        resp = ddb_client.scan(TableName=target_table, Limit=max(limit * 3, 25))
        rows: list[dict[str, str]] = []
        for raw in resp.get("Items", []):
            rows.append(
                {
                    "run_id": raw.get("run_id", {}).get("S", ""),
                    "created_at": raw.get("created_at", {}).get("S", ""),
                    "repo": raw.get("repo", {}).get("S", ""),
                    "issue_title": raw.get("issue_title", {}).get("S", ""),
                    "status": raw.get("status", {}).get("S", ""),
                    "iterations": raw.get("iterations", {}).get("N", "0"),
                    "findings_count": raw.get("findings_count", {}).get("N", "0"),
                    "s3_pdf_url": raw.get("s3_pdf_url", {}).get("S", ""),
                }
            )
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows[:limit]
    except Exception as exc:  # noqa: BLE001
        logger.warning("DynamoDB list_runs failed: %s", exc)
        return []


def publish_sns_notification(
    run_id: str,
    values: dict[str, Any],
    *,
    s3_pdf_url: str | None = None,
    topic_arn: str | None = None,
    sns_client: Any | None = None,
) -> bool:
    """Publish an email/alert notification to Amazon SNS (`BugFixAgentAlerts`) when a run finishes or pauses."""
    target_arn = (topic_arn or os.environ.get("AWS_SNS_TOPIC_ARN") or "").strip()
    if not target_arn:
        return False

    try:
        if sns_client is None:
            import boto3  # type: ignore[import-untyped]

            sns_client = boto3.client("sns", region_name=_aws_region())

        issue = values.get("issue")
        repo_ctx = values.get("repo")
        repro = values.get("repro_test")
        attempts = values.get("attempts") or []
        findings = getattr(repo_ctx, "findings", []) if repo_ctx else []
        status = str(values.get("status", "unknown"))
        title = str(getattr(issue, "title", "Repository Scan"))

        lines = [
            "================================================================",
            "AUTONOMOUS BUG-FIXING AGENT — FULL DIAGNOSTIC & FIX REPORT",
            "================================================================",
            f"Run ID:            {run_id}",
            f"Status:            {status.upper()}",
            f"Primary Issue:     {title}",
            f"Issues Discovered: {len(findings)}",
            f"Fix Iterations:    {values.get('iteration', 0)} / 8",
        ]
        if s3_pdf_url:
            lines.extend(
                [
                    "",
                    "📄 DOWNLOAD PRINTABLE PDF REPORT (Amazon S3 7-Day Link):",
                    s3_pdf_url,
                ]
            )
        lines.extend(
            [
                "",
                "----------------------------------------------------------------",
                f"1. REPOSITORY ISSUES AUDIT ({len(findings)} ISSUE(S) FOUND)",
                "----------------------------------------------------------------",
            ]
        )
        for idx, f in enumerate(findings, start=1):
            lines.append(f"#{idx} [{f.severity.upper()}] {f.file_path} — {f.title}")
            lines.append(f"   Details: {f.description}")
            if f.suggested_fix:
                lines.append(f"   Recommended Fix: {f.suggested_fix}")
            lines.append("")

        if repro:
            lines.extend(
                [
                    "----------------------------------------------------------------",
                    f"2. REPRODUCTION TEST ({repro.path})",
                    "----------------------------------------------------------------",
                    repro.content.strip(),
                    "",
                ]
            )

        if attempts:
            last = attempts[-1]
            diff_text = "\n".join(p.unified_diff for p in last.patches)
            lines.extend(
                [
                    "----------------------------------------------------------------",
                    f"3. VERIFIED CODE FIX (Iteration {last.iteration})",
                    "----------------------------------------------------------------",
                    f"Rationale: {last.rationale}",
                    "",
                    diff_text.strip(),
                    "",
                ]
            )

        lines.extend(
            [
                "================================================================",
                "Live CloudFront HTTPS Dashboard: https://d3gctx9v0f6g4u.cloudfront.net",
                "Hugging Face Dataset: https://huggingface.co/datasets/MF180/bugfix-agent-benchmarks",
                "================================================================",
            ]
        )

        sns_client.publish(
            TopicArn=target_arn,
            Subject=f"[BugFix Agent Report] {status.upper()}: {title[:55]}",
            Message="\n".join(lines),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("SNS publish failed: %s", exc)
        return False


def subscribe_email_to_sns(
    email: str,
    *,
    topic_arn: str | None = None,
    sns_client: Any | None = None,
) -> str:
    """Subscribe an email address to the Amazon SNS alert topic (`BugFixAgentAlerts`)."""
    target_arn = (topic_arn or os.environ.get("AWS_SNS_TOPIC_ARN") or "").strip()
    if not target_arn:
        raise RuntimeError("AWS_SNS_TOPIC_ARN is not configured.")
    if sns_client is None:
        import boto3  # type: ignore[import-untyped]

        sns_client = boto3.client("sns", region_name=_aws_region())

    resp = sns_client.subscribe(
        TopicArn=target_arn,
        Protocol="email",
        Endpoint=email.strip(),
    )
    return str(resp.get("SubscriptionArn") or "pending confirmation")


def publish_cloudwatch_metrics(
    values: dict[str, Any],
    *,
    elapsed_s: float = 0.0,
    namespace: str = "AutonomousBugFixingAgent",
    cw_client: Any | None = None,
) -> bool:
    """Publish operational metrics (`BugsDetected`, `FixIterations`, `RunDurationSeconds`) to Amazon CloudWatch."""
    try:
        if cw_client is None:
            import boto3  # type: ignore[import-untyped]

            cw_client = boto3.client("cloudwatch", region_name=_aws_region())

        repo_ctx = values.get("repo")
        findings = getattr(repo_ctx, "findings", []) if repo_ctx else []
        iterations = int(values.get("iteration", 0))

        metric_data = [
            {
                "MetricName": "BugsDetected",
                "Value": float(len(findings)),
                "Unit": "Count",
            },
            {
                "MetricName": "FixIterations",
                "Value": float(iterations),
                "Unit": "Count",
            },
            {
                "MetricName": "RunDurationSeconds",
                "Value": float(max(0.0, elapsed_s)),
                "Unit": "Seconds",
            },
        ]
        cw_client.put_metric_data(Namespace=namespace, MetricData=metric_data)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("CloudWatch put_metric_data failed: %s", exc)
        return False


def archive_run_to_aws(
    run_id: str,
    values: dict[str, Any],
    pdf_bytes: bytes | None = None,
    zip_bytes: bytes | None = None,
    elapsed_s: float = 0.0,
) -> CloudArchiveResult:
    """One-stop helper: upload PDF & ZIP to S3, log run to DynamoDB, publish CloudWatch metrics & SNS alert."""
    if not is_aws_configured():
        return CloudArchiveResult()

    bucket = os.environ.get("AWS_S3_BUCKET", "").strip() or None
    pdf_key = f"reports/{run_id}/bugfix-report.pdf"
    zip_key = f"reports/{run_id}/fixed-repository.zip"
    pdf_url: str | None = None
    zip_url: str | None = None

    try:
        if bucket and pdf_bytes:
            pdf_url = upload_bytes_to_s3(
                pdf_bytes, pdf_key, "application/pdf", bucket=bucket
            )
        if bucket and zip_bytes:
            zip_url = upload_bytes_to_s3(
                zip_bytes, zip_key, "application/zip", bucket=bucket
            )
        ddb_ok = save_run_to_dynamodb(
            run_id,
            values,
            s3_pdf_url=pdf_url,
            s3_zip_url=zip_url,
        )
        cw_ok = publish_cloudwatch_metrics(values, elapsed_s=elapsed_s)
        sns_ok = publish_sns_notification(run_id, values, s3_pdf_url=pdf_url)
        return CloudArchiveResult(
            s3_bucket=bucket,
            s3_pdf_key=pdf_key if pdf_url else None,
            s3_pdf_url=pdf_url,
            s3_zip_url=zip_url,
            dynamodb_saved=ddb_ok,
            sns_published=sns_ok,
            cloudwatch_published=cw_ok,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("AWS archive failed: %s", exc)
        return CloudArchiveResult(error=str(exc))
