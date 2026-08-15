from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from .io import atomic_json


@dataclass
class TokenLedgerState:
    limit: int = 10_000_000
    used: int = 0
    reserved: int = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used - self.reserved


class PersistentTokenLedger:
    def __init__(self, path: Path, limit: int = 10_000_000) -> None:
        if limit <= 0:
            raise ValueError("token limit must be positive")
        self.path = path
        self.lock = Lock()
        if path.exists():
            import json
            value = json.loads(path.read_text(encoding="utf-8"))
            if int(value["limit"]) != limit:
                raise ValueError("existing token ledger has a different limit")
            self.state = TokenLedgerState(
                limit=limit,
                used=int(value["used"]),
                reserved=int(value.get("reserved", 0)),
            )
            if self.state.remaining < 0:
                raise ValueError("existing token ledger exceeds its hard limit")
        else:
            self.state = TokenLedgerState(limit=limit)
            self._persist()

    def reserve(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("token reservation must be non-negative")
        with self.lock:
            if amount > self.state.remaining:
                self._persist()
                raise RuntimeError(
                    f"token budget exhausted: requested={amount}, remaining={self.state.remaining}"
                )
            self.state.reserved += amount
            self._persist()

    def settle(self, reserved: int, actual: int) -> None:
        if reserved < 0 or actual < 0 or actual > reserved:
            raise ValueError("invalid token settlement")
        with self.lock:
            if reserved > self.state.reserved:
                raise ValueError("settlement exceeds outstanding reservation")
            self.state.reserved -= reserved
            self.state.used += actual
            self._persist()

    def release(self, reserved: int) -> None:
        if reserved < 0:
            raise ValueError("release must be non-negative")
        with self.lock:
            if reserved > self.state.reserved:
                raise ValueError("release exceeds outstanding reservation")
            self.state.reserved -= reserved
            self._persist()

    def snapshot(self) -> Dict[str, Any]:
        value = asdict(self.state)
        value["remaining"] = self.state.remaining
        value["exhausted"] = self.state.remaining == 0
        return value

    def _persist(self) -> None:
        atomic_json(self.path, self.snapshot())
