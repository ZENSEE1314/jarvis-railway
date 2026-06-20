"""Desktop JARVIS UI for Windows.

This starts a local web UI with normal chat, dashboard, calendar, and PC action
support. It uses the Railway JARVIS server as the online brain.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pc_jarvis import DEFAULT_SERVER, DEFAULT_WORK_DIR, JarvisApi, PcActions, poll_once


def resource_path(name: str) -> Path:
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        return Path(bundle_dir) / name
    return Path(__file__).with_name(name)


DESKTOP_HTML = resource_path("desktop.html")


class DesktopState:
    def __init__(self, server_url: str, token: str, work_dir: Path, speak: bool):
        self.worker_id = os.getenv("JARVIS_WORKER_ID", "pc-jarvis-desktop")
        self.api = JarvisApi(server_url, token)
        self.actions = PcActions(work_dir, speak_enabled=speak)
        self.polling = False
        self.poll_thread: threading.Thread | None = None

    def local_action(self, text: str) -> str:
        lowered = text.lower().strip()
        if re.match(r"^(open|launch|start)\s+", lowered):
            return self.actions.run_command(text)
        if lowered in {"open work folder", "show work folder", "where is my work folder"}:
            return self.actions.run_command("work folder")
        if re.match(r"^(save note|note)\s+", lowered):
            return self.actions.run_command(text)
        return ""

    def start_polling(self, interval: int = 20) -> None:
        if self.polling:
            return
        self.polling = True

        def loop() -> None:
            import time

            while self.polling:
                poll_once(self.api, self.actions, self.worker_id, "")
                time.sleep(interval)

        self.poll_thread = threading.Thread(target=loop, daemon=True)
        self.poll_thread.start()

    def stop_polling(self) -> None:
        self.polling = False


def response(handler: BaseHTTPRequestHandler, status: int, body: Any, content_type: str = "application/json") -> None:
    if isinstance(body, (dict, list)):
        raw = json.dumps(body).encode("utf-8")
    elif isinstance(body, str):
        raw = body.encode("utf-8")
    else:
        raw = bytes(body)
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.end_headers()
    handler.wfile.write(raw)


class DesktopHandler(BaseHTTPRequestHandler):
    state: DesktopState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        response(self, 204, b"")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/desktop"}:
            response(self, 200, DESKTOP_HTML.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/dashboard":
            response(self, 200, self.state.api.request("GET", "/api/dashboard"))
            return
        if parsed.path == "/api/health":
            response(self, 200, self.state.api.request("GET", "/health"))
            return
        response(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/chat":
            text = str(payload.get("text", "")).strip()
            if not text:
                response(self, 400, {"error": "No text"})
                return

            local_result = self.state.local_action(text)
            data = self.state.api.chat(text)
            ai_text = data.get("response") or data.get("error") or ""
            if local_result:
                ai_text = f"{ai_text}\n\nPC action: {local_result}".strip()
                self.state.api.activity(self.state.worker_id, "pc_action", local_result, {"text": text})
            if payload.get("speak"):
                self.state.actions.say(ai_text)
            response(self, 200, {"response": ai_text, "local": local_result, "dashboard": data.get("dashboard")})
            return

        if parsed.path == "/api/speak":
            self.state.actions.say(str(payload.get("text", "")))
            response(self, 200, {"ok": True})
            return

        if parsed.path == "/api/poll/start":
            self.state.start_polling()
            response(self, 200, {"polling": True})
            return

        if parsed.path == "/api/poll/stop":
            self.state.stop_polling()
            response(self, 200, {"polling": False})
            return

        if parsed.path == "/api/poll/once":
            count = poll_once(self.state.api, self.state.actions, self.state.worker_id, "")
            response(self, 200, {"processed": count})
            return

        response(self, 404, {"error": "Not found"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the JARVIS desktop UI.")
    parser.add_argument("--server", default=os.getenv("JARVIS_SERVER_URL", DEFAULT_SERVER))
    parser.add_argument("--token", default=os.getenv("JARVIS_WORKER_TOKEN", ""))
    parser.add_argument("--work-dir", default=os.getenv("JARVIS_WORK_DIR", str(DEFAULT_WORK_DIR)))
    parser.add_argument("--port", type=int, default=int(os.getenv("JARVIS_DESKTOP_PORT", "8765")))
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--speak", action="store_true", default=True)
    args = parser.parse_args()

    DesktopHandler.state = DesktopState(args.server, args.token, Path(args.work_dir), args.speak)
    DesktopHandler.state.api.activity(
        DesktopHandler.state.worker_id,
        "desktop_online",
        f"Desktop JARVIS online. Work folder: {DesktopHandler.state.actions.work_dir}",
    )

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DesktopHandler)
    url = f"http://127.0.0.1:{args.port}/desktop"
    print(f"Desktop JARVIS running at {url}")
    print("Close this window to stop it.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
