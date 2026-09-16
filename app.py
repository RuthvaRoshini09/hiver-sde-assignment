"""
app.py
------
Modern Web UI Server for AppleSupport Agent.

Provides a clean HTTP server using Python standard library (zero extra dependencies).
Serves the web UI from web/ and exposes POST /api/analyze to query the support agent.

Usage:
  python app.py
  python app.py --port 8000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import agent pipeline (strictly untouched)
from src.support_agent import run_agent, _load_classifier, _load_retrieval_index

WEB_DIR = PROJECT_ROOT / "web"


class AgentRequestHandler(SimpleHTTPRequestHandler):
    """Handles static web assets and dynamic /api/analyze POST requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Health endpoint
        if path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            resp_data = {
                "status": "ready",
                "service": "AppleSupport AI Agent",
                "model": "Baseline 1 (TF-IDF + Logistic Regression)",
                "corpus_size": 102670
            }
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        # Default root path -> serve index.html
        if path == "/" or path == "":
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/api/analyze", "/analyze"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length <= 0:
                    self._send_json_error(400, "Missing request body")
                    return

                body_bytes = self.rfile.read(content_length)
                try:
                    payload = json.loads(body_bytes.decode("utf-8"))
                except Exception:
                    self._send_json_error(400, "Invalid JSON payload")
                    return

                raw_message = payload.get("message", "").strip()
                if not raw_message:
                    self._send_json_error(400, "Customer message cannot be empty")
                    return

                # Execute untouched agent pipeline
                result = run_agent(raw_message)

                # Send dynamic JSON response
                response_payload = {
                    "status": "success",
                    "decision": result.get("decision", "UNKNOWN"),
                    "escalation_reason": result.get("escalation_reason", ""),
                    "predicted_intent": result.get("predicted_intent", ""),
                    "confidence": result.get("confidence", 0.0),
                    "top_candidates": result.get("top_candidates", []),
                    "retrieved_cases": result.get("retrieved_cases", []),
                    "drafted_reply": result.get("drafted_reply", "")
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode("utf-8"))

            except Exception as e:
                self._send_json_error(500, f"Internal pipeline error: {str(e)}")
            return

        # Method not allowed for other POST paths
        self._send_json_error(404, "Endpoint not found")

    def _send_json_error(self, code: int, message: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "error", "error": message}).encode("utf-8"))

    def log_message(self, format, *args):
        # Clean server log format
        sys.stderr.write(f"[Server] {args[0]} - {args[1]} ({args[2]})\n")


def main():
    parser = argparse.ArgumentParser(description="AppleSupport Agent Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    print("=" * 70)
    print("  AppleSupport AI Agent — Web UI Server")
    print("=" * 70)
    print("Pre-loading classifier and retrieval index into memory...")
    try:
        _load_classifier()
        _load_retrieval_index()
        print("  ✓ Models and retrieval index pre-loaded successfully.")
    except Exception as e:
        print(f"  ! Warning during pre-load: {e}")

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, AgentRequestHandler)

    url = f"http://{args.host}:{args.port}"
    print(f"\nServer running at: {url}")
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.shutdown()
        print("Server stopped.")


if __name__ == "__main__":
    main()
