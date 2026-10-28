"""Local web UI for RAG.chatbot, served alongside the static results dashboard.

Backend: a stdlib http.server (no new web-framework dependency, consistent with
dashboard/index.html's own "no server needed" design -- this is the one piece of
the project that genuinely needs one, since BM25/embeddings/LLM calls can't run in
the browser). It serves dashboard/ as static files, so dashboard/index.html (the
fidelity/utility/privacy report) and dashboard/chatbot.html (this file's UI) are
both reachable from one server and link to each other, plus three JSON routes the
chatbot page calls:
    GET  /api/info   -> {"provider", "n_chunks", "n_companies"}
    POST /api/chat    {"question": "..."} -> {"answer", "provider", "sources": [...]}
    GET  /api/demo   -> RAG.chatbot.demo_rows's data (the scripted scenario-analysis
                         walkthrough, structured for the page to render)

Retrieval and generation reuse RAG.chatbot as-is; the index and generator are built
once at startup and shared across requests via the server instance (threaded, since
fastembed/requests calls are not instant). Run from implementations/synthetic_10k:
    uv run python -m RAG.chatbot_server
then open http://localhost:8001/chatbot.html (linked from the dashboard's nav, and
vice versa). Ctrl-C to stop.
"""

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from method_2.utils import setup_logging

from RAG.chatbot import answer_question, build_index, demo_rows, get_generator


logger = setup_logging("chatbot_server")

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
HOST = "localhost"
PORT = 8001


class ChatbotServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, chunks_by_id, bm25, semantic, provider, generate, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks_by_id = chunks_by_id
        self.bm25 = bm25
        self.semantic = semantic
        self.provider = provider
        self.generate = generate


class Handler(SimpleHTTPRequestHandler):
    server: ChatbotServer

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s %s", self.address_string(), fmt % args)

    def _send_json(self, obj: object, status: int = 200) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming convention)
        if self.path == "/api/info":
            srv = self.server
            self._send_json(
                {
                    "provider": srv.provider,
                    "n_chunks": len(srv.chunks_by_id),
                    "n_companies": len({c.ticker for c in srv.chunks_by_id.values()}),
                },
            )
            return
        if self.path == "/api/demo":
            srv = self.server
            try:
                rows = demo_rows(srv.chunks_by_id, srv.bm25, srv.semantic, srv.provider, srv.generate)
            except Exception:
                logger.exception("GET /api/demo failed")
                self._send_json({"error": "demo generation failed; see server logs"}, status=500)
                return
            self._send_json({"provider": srv.provider, "rows": rows})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return
        question = (body.get("question") or "").strip()
        if not question:
            self._send_json({"error": "question must not be empty"}, status=400)
            return

        srv = self.server
        try:
            answer, passages, provider = answer_question(
                question, srv.chunks_by_id, srv.bm25, srv.semantic, srv.provider, srv.generate,
            )
        except Exception:
            logger.exception("POST /api/chat failed for question: %r", question)
            self._send_json({"error": "chat failed; see server logs"}, status=500)
            return

        self._send_json(
            {
                "answer": answer,
                "provider": provider,
                "sources": [
                    {
                        "chunk_id": c.chunk_id,
                        "ticker": c.ticker,
                        "company_name": c.company_name,
                        "fiscal_year": c.fiscal_year,
                        "item": c.item,
                        "text": c.text,
                    }
                    for c in passages
                ],
            },
        )


def main() -> None:
    chunks_by_id, bm25, semantic = build_index()
    provider, generate = get_generator()
    if provider == "retrieval_only":
        logger.warning("No DEEPSEEK_API_KEY or ANTHROPIC_API_KEY set -- running retrieval-only (no synthesized answers)")
    else:
        logger.info("Generation backend: %s", provider)

    server = ChatbotServer(
        (HOST, PORT),
        Handler,
        chunks_by_id=chunks_by_id,
        bm25=bm25,
        semantic=semantic,
        provider=provider,
        generate=generate,
    )
    logger.info("Serving dashboard + chatbot at http://%s:%d/ (chatbot.html, index.html). Ctrl-C to stop.", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping")
        server.shutdown()


if __name__ == "__main__":
    main()
