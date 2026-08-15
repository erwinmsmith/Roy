#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from roy_research.token_ledger import PersistentTokenLedger


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="OpenAI-compatible DeepSeek hard-budget proxy")
    value.add_argument("--ledger", type=Path, required=True)
    value.add_argument("--limit", type=int, default=10_000_000)
    value.add_argument("--port", type=int, default=18080)
    value.add_argument("--upstream", default="https://api.deepseek.com")
    return value


def main() -> None:
    args = parser().parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required")
    ledger = PersistentTokenLedger(args.ledger, args.limit)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"ok": True, "ledger": ledger.snapshot()})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in ("/chat/completions", "/v1/chat/completions"):
                self._json(404, {"error": "unsupported endpoint"})
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                payload: Dict[str, Any] = json.loads(body)
                if payload.get("stream"):
                    raise ValueError("streaming is disabled so usage can be settled atomically")
                reserve = len(body) + int(payload.get("max_tokens", 1024)) + 1024
                ledger.reserve(reserve)
            except RuntimeError as error:
                self._json(429, {"error": str(error), "ledger": ledger.snapshot()})
                return
            except Exception as error:
                self._json(400, {"error": str(error)})
                return
            request = urllib.request.Request(
                f"{args.upstream.rstrip('/')}/chat/completions",
                data=body,
                method="POST",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    response_body = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
                decoded = json.loads(response_body)
                actual = int(decoded.get("usage", {}).get("total_tokens", reserve))
                ledger.settle(reserve, min(actual, reserve))
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except urllib.error.HTTPError as error:
                ledger.release(reserve)
                response_body = error.read()
                self.send_response(error.code)
                self.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as error:
                ledger.release(reserve)
                self._json(502, {"error": str(error)})

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
