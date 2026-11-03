"""Headless smoke tests for the Streamlit demo UI (no browser needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "streamlit_app.py")


def _start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.slider[0].set_value(0.0)
    _button(at, "Start run").click()
    return at.run()


def _button(at: AppTest, label_part: str):
    return next(b for b in at.button if label_part in b.label)


def _subheaders(at: AppTest) -> list[str]:
    return [s.value for s in at.subheader]


def test_initial_page_shows_demo_banner_and_prompt():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Demo mode" in w.value for w in at.warning)
    assert any("Start run" in i.value for i in at.info)


def test_run_pauses_for_approval_with_fix_and_tests():
    at = _start()
    assert not at.exception
    assert any("Waiting for your approval" in s for s in _subheaders(at))
    code = " ".join(c.value for c in at.code)
    assert "sum(xs) / len(xs)" in code and "2 passed" in code
    assert at.metric[0].value == "2 / 8"
    assert not any("commit" in str(s.value).lower() for s in at.success)  # nothing committed yet


def test_approve_shows_simulated_commit():
    at = _start()
    _button(at, "Approve").click()
    at.run()
    assert not at.exception
    assert any("Demo commit" in s.value for s in at.success)
    assert not any("Waiting for your approval" in s for s in _subheaders(at))


def test_request_changes_requires_feedback_then_loops_back():
    at = _start()
    _button(at, "Request changes").click()
    at.run()
    assert any("Add feedback" in w.value for w in at.warning)  # blocked, still paused
    assert any("Waiting for your approval" in s for s in _subheaders(at))

    at.text_area[0].set_value("handle the empty list")
    _button(at, "Request changes").click()
    at.run()
    assert not at.exception
    assert at.metric[0].value == "3 / 8"
    assert any("ValueError" in c.value for c in at.code)
    assert any("Waiting for your approval" in s for s in _subheaders(at))


def test_reject_escalates():
    at = _start()
    _button(at, "Reject").click()
    at.run()
    assert not at.exception
    assert any("Escalated" in e.value for e in at.error)

