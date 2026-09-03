"""Async LLM client (OpenAI-compat) → Ollama.

Key design points:
- Uses AsyncOpenAI with `base_url` pointing at Ollama's /v1 endpoint.
- Structured output via prompt-injected JSON schema + Pydantic validation (Ollama
  does not implement OpenAI's `response_format=json_schema` server-side).
- Retries on parse/validation failure up to N times; the prompt gets a short
  corrective suffix on each retry.
- Tracks tokens (in/out) and call count on a passed-in RunContext.
- Distinguishes LLMUnavailableError (connect refused, 5xx) from
  LLMTimeoutError (httpx timeout) from LLMSchemaError (bad JSON).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel

from comparables.core.context import RunContext
from comparables.core.exceptions import (
    LLMTimeoutError,
    LLMUnavailableError,
)
from comparables.llm.structured import LLMSchemaError, parse_strict, schema_instructions

T = TypeVar("T", bound=BaseModel)


# Corrective suffix appended on retry after a parse/validation failure.
_RETRY_SUFFIX = (
    "\n\nYour previous reply was not parseable as the required JSON. "
    "Return ONLY the JSON object, no prose, no fences, with exactly the keys "
    "and types from the schema."
)


class LLMClient:
    """Async LLM client wrapping an OpenAI-compat endpoint (Ollama)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_s,
                max_retries=0,  # we handle retries ourselves for clearer errors
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        ctx: RunContext | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        think: bool = False,
    ) -> tuple[str, int, int]:
        """Free-form completion. Returns (text, tokens_in, tokens_out)."""
        text, in_t, out_t = await self._call(
            system=system, user=user, temperature=temperature, max_tokens=max_tokens, think=think
        )
        if ctx is not None:
            ctx.inc_llm(in_t, out_t)
        return text, in_t, out_t

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        ctx: RunContext | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        think: bool = False,
    ) -> BaseModel:
        """Structured completion: returns a validated Pydantic model.

        Retries on parse/validation failure up to `max_retries` times.
        """
        schema_sys = schema_instructions(schema_model)
        full_system = f"{system}\n\n{schema_sys}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            sys_msg = full_system if attempt == 0 else full_system + _RETRY_SUFFIX
            try:
                text, in_t, out_t = await self._call(
                    system=sys_msg, user=user, temperature=temperature, max_tokens=max_tokens, think=think
                )
                if ctx is not None:
                    ctx.inc_llm(in_t, out_t)
                return parse_strict(text, schema_model)
            except LLMSchemaError as exc:
                last_err = exc
                continue
        raise LLMSchemaError(
            f"LLM failed to produce valid {schema_model.__name__} after "
            f"{self.max_retries + 1} attempts; last error: {last_err}"
        )

    # ─── Internals ──────────────────────────────────────────────────────
    async def _call(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        think: bool = False,
    ) -> tuple[str, int, int]:
        t0 = time.perf_counter()
        # Qwen3.5 / DeepSeek-R1 emit chain-of-thought that eats tokens. For
        # structured-output tasks we want the final answer only. The Ollama
        # OpenAI-compat API honors `extra_body["think"] = false`.
        extra_body: dict = {}
        if not think:
            extra_body["think"] = False
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        except APITimeoutError as exc:
            # NOTE: APITimeoutError subclasses APIConnectionError, so it must
            # be matched FIRST.
            raise LLMTimeoutError(f"LLM timed out after {self.timeout_s}s") from exc
        except (APIConnectionError, InternalServerError) as exc:
            raise LLMUnavailableError(f"LLM unavailable: {exc}") from exc
        except RateLimitError as exc:
            raise LLMUnavailableError(f"LLM rate-limited: {exc}") from exc

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        in_t = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out_t = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        # Latency observability: not exposed via return, but available on the
        # client wrapper for metrics later. Keep it cheap.
        _ = time.perf_counter() - t0
        return text, in_t, out_t

    async def ping(self) -> bool:
        """Lightweight reachability check. Used by health/ready."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False


__all__ = ["LLMClient"]
