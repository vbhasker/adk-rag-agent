"""A tiny, dependency-free web server: serves the lesson site and a JSON API.

Routes are plain Python functions that take a dict (JSON body or query string) and return
something JSON-serialisable. That keeps each chapter's server.py focused on the RAG logic.
"""

from __future__ import annotations

import json
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlparse

Route = Callable[[dict[str, Any]], Any]


class _Handler(SimpleHTTPRequestHandler):
    routes: dict[str, Route] = {}

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, params: dict[str, Any]) -> None:
        route = self.routes.get(urlparse(self.path).path)
        if route is None:
            self._send_json(404, {"error": f"unknown API route {self.path}"})
            return
        try:
            self._send_json(200, route(params))
        except Exception as exc:  # show the error in the playground instead of a blank page
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            self._dispatch(dict(parse_qsl(url.query)))
        else:
            super().do_GET()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        self._dispatch(json.loads(raw or b"{}"))

    def log_message(self, fmt: str, *args: Any) -> None:
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)


def serve(routes: dict[str, Route], site_dir: Path, port: int) -> None:
    handler = type("Handler", (_Handler,), {"routes": routes})
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(site_dir)))
    print(f"\n  Lesson + playground running at  http://localhost:{port}\n  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")
