from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .io import write_jsonl
from .token_ledger import PersistentTokenLedger


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
    ) -> None:
        self.ledger = ledger
        self.model = model
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.timeout = timeout
        self.event_log = event_log
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        metadata: Dict[str, Any] | None = None,
    ) -> Completion:
        # UTF-8 bytes are a conservative upper bound for ordinary tokenizer input
        # units; reserve before dispatch so concurrent callers cannot cross the cap.
        prompt_reserve = max(1, sum(len(message["content"].encode("utf-8")) for message in messages) + 1024)
        reservation = prompt_reserve + max_tokens
        self.ledger.reserve(reservation)
        request_id = str(uuid.uuid4())
        request_value = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        payload = json.dumps(request_value).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        settled = False
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            usage = raw.get("usage", {})
            total = int(usage.get("total_tokens", reservation))
            if total > reservation:
                total = reservation
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
                "schema_version": 1,
                "request_id": request_id,
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
            # A timed-out or disconnected request may still have been processed by
            # the provider. Charge the full reservation so a retry/restart can
            # never make the persistent hard cap optimistic.
            if not settled:
                self.ledger.settle(reservation, reservation)
            self._record_event({
                "schema_version": 1,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": "deepseek",
                "base_url": self.base_url,
                "request": request_value,
                "reservation": reservation,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "ledger": self.ledger.snapshot(),
                "metadata": metadata or {},
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            })
            raise

    def _record_event(self, event: Dict[str, Any]) -> None:
        if self.event_log is not None:
            write_jsonl(self.event_log, [event], append=self.event_log.exists())
