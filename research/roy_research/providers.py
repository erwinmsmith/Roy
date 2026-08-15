from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    ) -> None:
        self.ledger = ledger
        self.model = model
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Completion:
        # UTF-8 bytes are a conservative upper bound for ordinary tokenizer input
        # units; reserve before dispatch so concurrent callers cannot cross the cap.
        prompt_reserve = max(1, sum(len(message["content"].encode("utf-8")) for message in messages) + 1024)
        reservation = prompt_reserve + max_tokens
        self.ledger.reserve(reservation)
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            usage = raw.get("usage", {})
            total = int(usage.get("total_tokens", reservation))
            if total > reservation:
                total = reservation
            self.ledger.settle(reservation, total)
            choice = (raw.get("choices") or [{}])[0]
            return Completion(
                content=str(choice.get("message", {}).get("content", "")),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=total,
                latency_ms=int((time.monotonic() - started) * 1000),
                raw=raw,
            )
        except Exception:
            self.ledger.release(reservation)
            raise
