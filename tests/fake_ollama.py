"""A real HTTP server on a local port that answers like Ollama.

The clients are tested through a socket, not a mocked ``urlopen``, so request
framing, status codes and response parsing are all exercised.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Reply = Callable[[dict[str, Any]], tuple[int, bytes]]


class FakeOllama:
    def __init__(self, reply: Reply):
        self.requests: list[tuple[str, dict[str, Any]]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.requests.append((self.path, body))
                status, data = reply(body)
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeOllama:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def json_reply(payload: Any, status: int = 200) -> Reply:
    return lambda _body: (status, json.dumps(payload).encode("utf-8"))


def raw_reply(data: bytes, status: int = 200) -> Reply:
    return lambda _body: (status, data)
