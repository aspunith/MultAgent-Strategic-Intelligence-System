"""
MASIS LLM Utilities — Model instantiation, rate limiting, and structured output helpers.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from masis.config import get_config

T = TypeVar("T", bound=BaseModel)


# ──────────────────────────────────────────────────────────────
# Rate Limiter
# ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, max_rpm: int):
        self._max_rpm = max_rpm
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        while True:
            with self._lock:
                now = time.time()
                # Purge timestamps older than 60 seconds
                while self._timestamps and self._timestamps[0] < now - 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_rpm:
                    self._timestamps.append(now)
                    return
            time.sleep(0.5)


_rate_limiter: RateLimiter | None = None


def _get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        cfg = get_config()
        _rate_limiter = RateLimiter(cfg.agents.rate_limit_rpm)
    return _rate_limiter


# ──────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────

def get_primary_llm(**kwargs: Any) -> ChatOpenAI:
    """Large reasoning model for complex tasks (Supervisor, Skeptic)."""
    cfg = get_config().models
    return ChatOpenAI(
        model=cfg.primary_model,
        temperature=kwargs.pop("temperature", cfg.temperature_reasoning),
        **kwargs,
    )


def get_secondary_llm(**kwargs: Any) -> ChatOpenAI:
    """Smaller/cheaper model for extraction and drafting (Researcher summaries, Synthesizer)."""
    cfg = get_config().models
    return ChatOpenAI(
        model=cfg.secondary_model,
        temperature=kwargs.pop("temperature", cfg.temperature_creative),
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────
# Structured Invocation
# ──────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def invoke_llm(
    llm: ChatOpenAI,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Invoke an LLM with rate limiting and retry logic. Returns raw text."""
    _get_rate_limiter().acquire()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def invoke_llm_structured(
    llm: ChatOpenAI,
    system_prompt: str,
    user_prompt: str,
    output_schema: type[T],
) -> T:
    """Invoke an LLM and parse output into a Pydantic model via structured output."""
    _get_rate_limiter().acquire()
    structured_llm = llm.with_structured_output(output_schema)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    return structured_llm.invoke(messages)
