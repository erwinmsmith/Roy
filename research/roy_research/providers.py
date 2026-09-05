from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .io import write_jsonl
from .token_ledger import PersistentTokenLedger


class ProviderCircuitOpenError(RuntimeError):
    """The provider is unavailable in a way that should stop the whole shard."""


class ProviderPaymentRequiredError(ProviderCircuitOpenError):
    """The provider rejected the account before processing the request."""


class ProviderRetryExhaustedError(ProviderCircuitOpenError):
    """A transient provider failure persisted through bounded backoff retries."""


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    raw: Dict[str, Any]


class DeepSeekClient:
    def __init__(
        self,
        ledger: PersistentTokenLedger,
        model: str = "deepseek-v4-flash",
        base_url: str | None = None,
        timeout: float = 60.0,
        event_log: Path | None = None,
        max_retries: int = 4,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 30.0,
    ) -> None:
        self.ledger = ledger
        self.model = model
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.timeout = timeout
        self.event_log = event_log
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("provider retry delays cannot be negative")
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.event_lock = Lock()
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        metadata: Dict[str, Any] | None = None,
        json_mode: bool = False,
        thinking: str | None = None,
    ) -> Completion:
        # UTF-8 bytes are a conservative upper bound for ordinary tokenizer input
        # units; reserve before dispatch so concurrent callers cannot cross the cap.
        prompt_reserve = max(1, sum(len(message["content"].encode("utf-8")) for message in messages) + 1024)
        reservation = prompt_reserve + max_tokens
        request_id = str(uuid.uuid4())
        request_value = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            request_value["response_format"] = {"type": "json_object"}
        if thinking is not None:
            request_value["thinking"] = {"type": thinking}
        payload = json.dumps(request_value).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.ledger.reserve(reservation)
            started = time.monotonic()
            settled = False
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                usage = raw.get("usage", {})
                total = min(int(usage.get("total_tokens", reservation)), reservation)
                self.ledger.settle(reservation, total)
                settled = True
                choice = (raw.get("choices") or [{}])[0]
                completion = Completion(
                    content=str(choice.get("message", {}).get("content", "")),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=total,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    raw=raw,
                )
                self._record_event({
                    "schema_version": 2,
                    "request_id": request_id,
                    "provider_attempt": attempt + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": "deepseek",
                    "base_url": self.base_url,
                    "request": request_value,
                    "response": raw,
                    "reservation": reservation,
                    "latency_ms": completion.latency_ms,
                    "ledger": self.ledger.snapshot(),
                    "metadata": metadata or {},
                    "status": "completed",
                })
                return completion
            except Exception as error:
                last_error = error
                status = error.code if isinstance(error, urllib.error.HTTPError) else None
                # Explicit HTTP rejection codes are known not to have produced a
                # completion. Ambiguous disconnects may have been processed, so
                # retain the conservative full-reservation charge for those.
                rejected_before_completion = status in {401, 402, 403, 429}
                if not settled:
                    if rejected_before_completion:
                        self.ledger.release(reservation)
                    else:
                        self.ledger.settle(reservation, reservation)
                retryable = (
                    status == 429
                    or (status is not None and 500 <= status <= 599)
                    or (
                        status is None
                        and isinstance(
                            error, (urllib.error.URLError, TimeoutError, ConnectionError),
                        )
                    )
                )
                self._record_event({
                    "schema_version": 2,
                    "request_id": request_id,
                    "provider_attempt": attempt + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": "deepseek",
                    "base_url": self.base_url,
                    "request": request_value,
                    "reservation": reservation,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "ledger": self.ledger.snapshot(),
                    "metadata": metadata or {},
                    "status": "failed",
                    "http_status": status,
                    "retryable": retryable,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                if status == 402:
                    raise ProviderPaymentRequiredError(
                        "DeepSeek API returned HTTP 402; stop the shard and restore account credit"
                    ) from error
                if status in {401, 403}:
                    raise ProviderCircuitOpenError(
                        f"DeepSeek API rejected credentials with HTTP {status}; stop the shard"
                    ) from error
                if not retryable:
                    raise
                if attempt >= self.max_retries:
                    raise ProviderRetryExhaustedError(
                        f"DeepSeek provider remained unavailable after {attempt + 1} attempts: {error}"
                    ) from error
                time.sleep(self._retry_delay(attempt, error))
        assert last_error is not None  # pragma: no cover
        raise last_error

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        retry_after = None
        if isinstance(error, urllib.error.HTTPError):
            retry_after = error.headers.get("Retry-After") if error.headers else None
        try:
            explicit = float(retry_after) if retry_after is not None else 0.0
        except ValueError:
            explicit = 0.0
        exponential = min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2 ** attempt),
        )
        # Independent processes otherwise retry in lockstep and reproduce the 429.
        return max(explicit, exponential * random.uniform(0.75, 1.25))

    def _record_event(self, event: Dict[str, Any]) -> None:
        if self.event_log is not None:
            with self.event_lock:
                write_jsonl(self.event_log, [event], append=self.event_log.exists())


class OpenAICompatibleClient:
    """Ledgered client for any endpoint implementing OpenAI chat completions."""

    def __init__(
        self,
        ledger: PersistentTokenLedger,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        event_log: Path | None = None,
        max_retries: int = 4,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 30.0,
        max_output_tokens: int | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("OpenAI-compatible model cannot be empty")
        if not base_url.strip():
            raise ValueError("OpenAI-compatible base URL cannot be empty")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("provider retry delays cannot be negative")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not configured")
        self.ledger = ledger
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.event_log = event_log
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.max_output_tokens = max_output_tokens
        self.event_lock = Lock()
        self.client = sdk_client or OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        metadata: Dict[str, Any] | None = None,
        json_mode: bool = False,
        thinking: str | None = None,
    ) -> Completion:
        requested_max_tokens = max_tokens
        max_tokens = min(max_tokens, self.max_output_tokens or max_tokens)
        prompt_reserve = max(
            1,
            sum(len(message["content"].encode("utf-8")) for message in messages) + 1024,
        )
        reservation = prompt_reserve + max_tokens
        request_id = str(uuid.uuid4())
        request_value: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            request_value["response_format"] = {"type": "json_object"}
        # `thinking` is a Roy execution hint. It is intentionally not forwarded
        # as a vendor-specific request field on a generic OpenAI endpoint.
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.ledger.reserve(reservation)
            started = time.monotonic()
            settled = False
            try:
                response = self.client.chat.completions.create(**request_value)
                raw = response.model_dump(mode="json")
                usage = raw.get("usage") or {}
                total = min(int(usage.get("total_tokens") or reservation), reservation)
                self.ledger.settle(reservation, total)
                settled = True
                choice = (raw.get("choices") or [{}])[0]
                completion = Completion(
                    content=str((choice.get("message") or {}).get("content") or ""),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=total,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    raw=raw,
                )
                self._record_event({
                    "schema_version": 2,
                    "request_id": request_id,
                    "provider_attempt": attempt + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": "openai_compatible",
                    "base_url": self.base_url,
                    "request": request_value,
                    "requested_max_tokens": requested_max_tokens,
                    "requested_thinking": thinking,
                    "response": raw,
                    "reservation": reservation,
                    "latency_ms": completion.latency_ms,
                    "ledger": self.ledger.snapshot(),
                    "metadata": metadata or {},
                    "status": "completed",
                })
                return completion
            except Exception as error:
                last_error = error
                status = error.status_code if isinstance(error, APIStatusError) else None
                rejected_before_completion = status in {400, 401, 402, 403, 404, 422, 429}
                if not settled:
                    if rejected_before_completion:
                        self.ledger.release(reservation)
                    else:
                        self.ledger.settle(reservation, reservation)
                retryable = (
                    status == 429
                    or (status is not None and 500 <= status <= 599)
                    or isinstance(error, (APIConnectionError, APITimeoutError, TimeoutError))
                )
                self._record_event({
                    "schema_version": 2,
                    "request_id": request_id,
                    "provider_attempt": attempt + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": "openai_compatible",
                    "base_url": self.base_url,
                    "request": request_value,
                    "requested_max_tokens": requested_max_tokens,
                    "requested_thinking": thinking,
                    "reservation": reservation,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "ledger": self.ledger.snapshot(),
                    "metadata": metadata or {},
                    "status": "failed",
                    "http_status": status,
                    "retryable": retryable,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                if status == 402:
                    raise ProviderPaymentRequiredError(
                        "OpenAI-compatible API returned HTTP 402; stop the shard"
                    ) from error
                if status in {401, 403}:
                    raise ProviderCircuitOpenError(
                        f"OpenAI-compatible API rejected credentials with HTTP {status}"
                    ) from error
                if not retryable:
                    raise
                if attempt >= self.max_retries:
                    raise ProviderRetryExhaustedError(
                        "OpenAI-compatible provider remained unavailable after "
                        f"{attempt + 1} attempts: {error}"
                    ) from error
                time.sleep(self._retry_delay(attempt, error))
        assert last_error is not None  # pragma: no cover
        raise last_error

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        retry_after = 0.0
        if isinstance(error, APIStatusError) and error.response is not None:
            value = error.response.headers.get("Retry-After")
            try:
                retry_after = float(value) if value is not None else 0.0
            except ValueError:
                retry_after = 0.0
        exponential = min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2 ** attempt),
        )
        return max(retry_after, exponential * random.uniform(0.75, 1.25))

    def _record_event(self, event: Dict[str, Any]) -> None:
        if self.event_log is not None:
            with self.event_lock:
                write_jsonl(self.event_log, [event], append=self.event_log.exists())
