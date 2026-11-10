"""Generate a clean, printable PDF report summarizing the bug, reproduction test,
fix attempts, unified diffs, and test results.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fpdf import FPDF

from agent.state import FixAttempt, IssueInput, RepoContext, RepoFinding, ReproTest


def _latin1(text: str) -> str:
    """Sanitize arbitrary text so built-in PDF fonts (Helvetica/Courier) never fail."""
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2192": "->",
        "\u2713": "[OK]",
        "\u2705": "[PASS]",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 7, "Autonomous Bug-Fixing Agent - Diagnostic & Fix Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def _section_heading(pdf: FPDF, title: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(25, 45, 90)
    pdf.cell(0, 8, _latin1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(200, 210, 225)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(20, 20, 20)


def _code_block(pdf: FPDF, code: str, max_lines: int = 80) -> None:
    lines = code.strip().splitlines()
    if len(lines) > max_lines:
        lines = [*lines[:max_lines], f"... ({len(lines) - max_lines} more lines truncated)"]
    pdf.set_font("Courier", "", 8.5)
    pdf.set_fill_color(245, 247, 250)
    pdf.set_text_color(30, 30, 30)
    text = _latin1("\n".join(lines) or "(empty)")
    pdf.multi_cell(0, 4.5, text, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def generate_pdf_report(values: dict[str, Any]) -> bytes:
    """Build a PDF report from the current `AgentState` values and return raw PDF bytes."""
    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # Title Banner
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 25, 55)
    pdf.cell(0, 10, "Autonomous Bug-Fixing Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    status = str(values.get("status", "unknown")).upper()
    iterations = values.get("iteration", 0)
    pdf.cell(
        0,
        6,
        _latin1(f"Generated: {ts}   |   Status: {status}   |   Fix Iterations: {iterations} / 8"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    if values.get("commit_sha"):
        pdf.cell(
            0,
            6,
            _latin1(f"Branch: {values.get('branch', '-')}   |   Commit SHA: {values.get('commit_sha')}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    # 1. Repository Issues Audit (All Issues Found)
    repo_ctx: RepoContext | None = values.get("repo")
    findings: list[RepoFinding] = list(repo_ctx.findings) if repo_ctx and repo_ctx.findings else []
    if findings:
        _section_heading(pdf, f"1. Repository Issues Audit ({len(findings)} issue(s) found)")
        for idx, f in enumerate(findings, start=1):
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.multi_cell(
                0,
                5.5,
                _latin1(f"Issue #{idx} [{f.severity.upper()}] - {f.file_path}: {f.title}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_font("Helvetica", "", 9.5)
            details = f.description or "(no description)"
            if f.suggested_fix:
                details += f"\nRecommended Fix: {f.suggested_fix}"
            pdf.multi_cell(0, 5, _latin1(details), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)

    # 2. Primary Bug Selected for Automated Reproduction & Fix
    issue: IssueInput | None = values.get("issue")
    if issue is not None:
        _section_heading(pdf, "2. Primary Issue Selected for Automated Fix")
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, _latin1(f"Title: {issue.title}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            0,
            5.5,
            _latin1(f"Repository: {issue.repo_path}\n\nDescription:\n{issue.body or '(no description)'}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    # 3. Reproduction Test
    repro: ReproTest | None = values.get("repro_test")
    if repro is not None:
        _section_heading(pdf, f"3. Reproduction Test ({repro.path})")
        pdf.set_font("Helvetica", "", 10)
        repro_status = "Confirmed failing on baseline code" if repro.confirmed_failing else "Not confirmed failing"
        pdf.cell(0, 6, _latin1(f"Reproduction Status: {repro_status}"), new_x="LMARGIN", new_y="NEXT")
        _code_block(pdf, repro.content)

    # 4. Fix Attempts & Unified Diffs
    attempts: list[FixAttempt] = values.get("attempts") or []
    _section_heading(pdf, f"4. Fix Iterations & Code Changes ({len(attempts)} attempt(s))")
    if not attempts:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 6, "No fix attempts recorded.", new_x="LMARGIN", new_y="NEXT")
    else:
        for att in attempts:
            passed = bool(att.result and att.result.passed)
            verdict = "PASSED" if passed else ("PATCH ERROR" if att.patch_error else "FAILED")
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 110, 50) if passed else pdf.set_text_color(160, 40, 40)
            pdf.cell(
                0,
                7,
                _latin1(f"Iteration {att.iteration}: {verdict}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_text_color(20, 20, 20)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(
                0,
                5.5,
                _latin1(f"Root-Cause & Fix Rationale: {att.rationale}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            if att.patch_error:
                pdf.set_font("Helvetica", "I", 9.5)
                pdf.multi_cell(0, 5, _latin1(f"Edit Error: {att.patch_error}"), new_x="LMARGIN", new_y="NEXT")
            if att.patches:
                diff_text = "\n".join(p.unified_diff for p in att.patches)
                _code_block(pdf, diff_text)
            if att.result:
                out = (att.result.stdout + "\n" + att.result.stderr).strip()
                if out:
                    pdf.set_font("Helvetica", "B", 9.5)
                    pdf.cell(0, 5.5, "Sandbox Test Output:", new_x="LMARGIN", new_y="NEXT")
                    _code_block(pdf, out, max_lines=35)

    # 4. Terminal Error / Escalation Note (if any)
    if values.get("error"):
        _section_heading(pdf, "4. Escalation / Error Details")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _latin1(str(values["error"])), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
