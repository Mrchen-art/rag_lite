# -*- coding: utf-8 -*-
"""HTTP 服务：提供静态页面 + /api/ask + /api/stats。

刻意使用标准库 http.server 而非 Flask/FastAPI：
整个项目零第三方依赖，任何装了 Python 3.7+ 的机器都能直接跑起来，
演示和复现成本最低。
"""
import json
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

STATS = {"queries": 0, "started_at": time.time()}


def make_server(pipeline, web_dir, host="127.0.0.1", port=8000):
    class RagHandler(SimpleHTTPRequestHandler):
        pipe = pipeline
        web_root = web_dir

        def __init__(self, *args, **kwargs):
            kwargs["directory"] = self.web_root
            super().__init__(*args, **kwargs)

        def log_message(self, fmt, *args):  # 保持控制台干净
            pass

        def _send_json(self, obj, code=200):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/api/stats"):
                st = self.pipe.stats()
                st.update(STATS)
                st["uptime_sec"] = int(time.time() - STATS["started_at"])
                return self._send_json(st)
            if self.path.startswith("/api/suggest"):
                return self._send_json({"questions": self.pipe.suggestions})
            return super().do_GET()

        def do_POST(self):
            if not self.path.startswith("/api/ask"):
                return self._send_json({"error": "not found"}, 404)
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                return self._send_json({"error": "invalid json"}, 400)

            query = (body.get("query") or "").strip()
            if not query:
                return self._send_json({"error": "empty query"}, 400)

            top_k = int(body.get("top_k", 3) or 3)
            mode = body.get("mode", "hybrid")
            if mode not in ("bm25", "vector", "hybrid"):
                mode = "hybrid"

            STATS["queries"] += 1
            result = self.pipe.answer(query, top_k=top_k, mode=mode)
            return self._send_json(result)

    return ThreadingHTTPServer((host, port), RagHandler)
