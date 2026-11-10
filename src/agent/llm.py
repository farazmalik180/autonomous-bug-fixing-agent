"""Groq LLM client with free-tier guardrails: tenacity exponential backoff, model fallback,
and structured JSON output parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import (
     retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from agent.state import FileEdit, RepoFinding

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default to active Groq models; fall back automatically if llama-3.3-70b-versatile is set
# in .env from older templates.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
AVAILABLE_GROQ_MODELS = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
)
DEPRECATED_MODEL_MAP = {
    "llama-3.3-70b-versatile": DEFAULT_GROQ_MODEL,
    "llama3-70b-8192": DEFAULT_GROQ_MODEL,
}


class ReproTestProposal(BaseModel):
    """Structured output for `write_repro_test`."""

    path: str = Field(
        default="test_repro_bug.py",
        description="Relative path for the reproduction test file (must start with test_ and end with .py).",
    )
    content: str = Field(
        min_length=10,
        description="Complete, runnable pytest file that reproduces the reported bug.",
    )


class FixProposal(BaseModel):
    """Structured output for `propose_fix`."""

    rationale: str = Field(
        min_length=1,
        description="Concise explanation of the root cause and how these edits fix it.",
    )
    edits: list[FileEdit] = Field(
        min_length=1,
        description="Exact search/replace blocks in existing repository files.",
    )


class DetectedBugProposal(BaseModel):
    """Structured output when scanning a repository for all issues + a primary testable bug."""

    title: str = Field(
        min_length=3,
        description="Short issue title naming the file/function and the primary unit-testable bug.",
    )
    body: str = Field(
        min_length=10,
        description=(
            "Detailed bug description naming the exact module/function, the input that triggers "
            "the bug, the current wrong behavior, and the expected correct behavior."
        ),
    )
    file_path: str = Field(
        min_length=1,
        description="Relative path of the primary Python file containing the bug.",
    )
    findings: list[RepoFinding] = Field(
        default_factory=list,
        description="Complete audit list of all bugs, edge-case flaws, and security/reliability issues found across the repository.",
    )


def load_dotenv_into_environ(dotenv_path: Path | None = None) -> None:
    """Load key=value pairs from `.env` (handling UTF-8 BOM safely) without printing values."""
    candidates = [dotenv_path] if dotenv_path else [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    for path in candidates:
        if path and path.is_file():
            raw = path.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                raw = raw[3:]
            for line in raw.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and v and not os.environ.get(k):
                    os.environ[k] = v
            break


def resolve_groq_model(requested: str | None = None) -> str:
    """Resolve model name, transparently upgrading deprecated model IDs."""
    load_dotenv_into_environ()
    raw = (requested or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()
    return DEPRECATED_MODEL_MAP.get(raw, raw)


def default_groq_api_key() -> str:
    load_dotenv_into_environ()
    return os.environ.get("GROQ_API_KEY", "").strip()


def create_groq_llm(
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ChatGroq:
    """Instantiate `ChatGroq` configured for Groq's free tier (8,000 TPM limit)."""
    key = (api_key or default_groq_api_key()).strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env or enter it in the Streamlit sidebar."
        )
    resolved_model = resolve_groq_model(model)
    return ChatGroq(
        model=resolved_model,
        api_key=key,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=1,  # outer tenacity handles retries with exponential backoff
    )


def _is_retryable_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(k in name for k in ("ratelimit", "timeout", "connection", "apitimeout", "internalserver")):
        return True
    return any(
        code in msg
        for code in (
            "429",
            "413",
            "rate limit",
            "rate_limit_exceeded",
            "tokens per minute",
            "502",
            "503",
            "504",
            "timed out",
        )
    )


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_random_exponential(multiplier=2.0, min=2.0, max=40.0),
    stop=stop_after_attempt(6),
    reraise=True,
)
def invoke_with_backoff(llm: BaseChatModel, messages: list[BaseMessage]) -> BaseMessage:
    """Call ``llm.invoke(messages)`` with tenacity exponential backoff on 429/413 TPM & transient errors."""
    return llm.invoke(messages)


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from an LLM response (stripping <think> tags and fences)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _message_text(msg: BaseMessage) -> str:
    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
    if content and content.strip():
        return content
    # Fallback if reasoning model placed output in additional_kwargs
    extra = getattr(msg, "additional_kwargs", {}) or {}
    for k in ("reasoning_content", "content"):
        if isinstance(extra.get(k), str) and extra[k].strip():
            return extra[k]
    return ""


def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
) -> T:
    """Invoke the LLM and parse its output into ``schema`` (Pydantic v2).

    Uses native `with_structured_output(schema, method="json_mode")` when supported by
    `ChatGroq` so reasoning models never return empty content, with a resilient manual
    JSON-extraction + repair fallback for custom/fake test models.
    """
    if isinstance(llm, ChatGroq):
        try:
            structured_llm = llm.with_structured_output(schema, method="json_mode")
            schema_hint = json.dumps(schema.model_json_schema(), indent=2)
            res = invoke_with_backoff(  # type: ignore[arg-type]
                structured_llm,  # type: ignore[arg-type]
                [
                    SystemMessage(
                        content=(
                            f"{system_prompt.strip()}\n\n"
                            "Respond with a valid JSON object matching this schema:\n"
                            f"{schema_hint}"
                        )
                    ),
                    HumanMessage(content=user_prompt[-10_000:]),
                ],
            )
            if isinstance(res, schema):
                return res
            if isinstance(res, dict):
                return schema.model_validate(res)
        except Exception as native_err:
            logger.warning("Native json_mode structured output failed (%s); falling back", native_err)

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system_prompt.strip()}\n\n"
        "Be concise in internal reasoning. Respond ONLY with a single valid JSON object "
        "matching this JSON schema (no markdown prose outside the JSON):\n"
        f"{schema_json}"
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_prompt),
    ]
    response = invoke_with_backoff(llm, messages)
    content = _message_text(response)
    try:
        data = _extract_json_object(content)
        return schema.model_validate(data)
    except Exception as first_err:
        logger.warning("First structured parse failed (%s); asking LLM to repair JSON", first_err)
        repair_messages: list[BaseMessage] = [
            *messages,
            AIMessage(content=content or "{}"),
            HumanMessage(
                content=(
                    f"Your previous response could not be parsed ({first_err}). "
                    "Do NOT include reasoning or markdown fences. Output ONLY the raw JSON object "
                    f"with keys {list(schema.model_fields.keys())}."
                )
            ),
        ]
        repaired = invoke_with_backoff(llm, repair_messages)
        repaired_text = _message_text(repaired)
        return schema.model_validate(_extract_json_object(repaired_text))

