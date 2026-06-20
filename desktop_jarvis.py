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
from datetime import datetime
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


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


class LocalBrain:
    def __init__(self, work_dir: Path):
        self.root = work_dir / "brain"
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.root / "memory.json"
        self.chat_file = self.root / "chat_history.json"

    def _read(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default

    def _write(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def remember(self, text: str, source: str = "user") -> dict[str, Any] | None:
        clean = text.strip()
        if len(clean) < 3:
            return None
        key = normalize(clean)[:200]
        memories = self._read(self.memory_file, [])
        now = datetime.now().isoformat(timespec="seconds")
        for item in memories:
            if item["key"] == key:
                item["text"] = clean
                item["source"] = source
                item["updated_at"] = now
                item["count"] = int(item.get("count", 1)) + 1
                self._write(self.memory_file, memories)
                return item
        item = {"key": key, "text": clean, "source": source, "created_at": now, "updated_at": now, "count": 1}
        memories.append(item)
        self._write(self.memory_file, memories[-2000:])
        return item

    def log_chat(self, user_text: str, reply: str, mode: str) -> None:
        history = self._read(self.chat_file, [])
        history.append(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "user": user_text,
                "reply": reply,
                "mode": mode,
            }
        )
        self._write(self.chat_file, history[-1000:])

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        terms = [term for term in normalize(query).split() if len(term) > 2]
        memories = self._read(self.memory_file, [])
        scored = []
        for item in memories:
            haystack = item.get("key", "")
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].get("updated_at", "")), reverse=True)
        return [item for _, item in scored[:limit]]

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        return self._read(self.memory_file, [])[-limit:]


class DesktopState:
    def __init__(self, server_url: str, token: str, work_dir: Path, speak: bool):
        self.worker_id = os.getenv("JARVIS_WORKER_ID", "pc-jarvis-desktop")
        self.api = JarvisApi(server_url, token)
        self.actions = PcActions(work_dir, speak_enabled=speak)
        self.brain = LocalBrain(work_dir)
        self.polling = False
        self.poll_thread: threading.Thread | None = None

    def offline_reply(self, text: str) -> tuple[str, bool]:
        lowered = text.lower().strip()
        self.brain.remember(text, "user")

        if lowered in {"hi", "hello", "hey", "jarvis"}:
            return "Yes Sir. I am here, listening from your PC.", True
        if lowered in {"time", "what time is it", "what is the time"}:
            return f"It is {datetime.now().strftime('%I:%M %p')}.", True
        if lowered in {"date", "today", "what date is it", "what is today"}:
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.", True

        if re.match(r"^(open|launch|start)\s+", lowered):
            return self.actions.run_command(text), True
        if any(phrase in lowered for phrase in ["open work folder", "show work folder", "where is my work folder"]):
            return self.actions.run_command("work folder"), True
        if re.match(r"^(save note|take note|note)\s+", lowered):
            note_text = re.sub(r"^(save note|take note|note)\s+", "note ", text, flags=re.I)
            result = self.actions.run_command(note_text)
            self.brain.remember(text, "note")
            return f"Saved the note here: {result}", True
        remember_match = re.match(r"^(remember|remember that)\s+(.+)$", text, flags=re.I)
        if remember_match:
            memory = remember_match.group(2).strip()
            self.brain.remember(memory, "explicit")
            return f"I will remember: {memory}", True
        if any(phrase in lowered for phrase in ["what do you remember", "search memory", "find memory", "what did i ask"]):
            query = re.sub(r"(what do you remember about|what do you remember|search memory|find memory|what did i ask)", "", text, flags=re.I).strip()
            matches = self.brain.search(query or text) if query else self.brain.recent()
            if not matches:
                return "I do not have matching local memory yet.", True
            lines = [f"- {item['text']} ({item.get('updated_at', '')})" for item in matches]
            return "Here is what I found in my local brain:\n" + "\n".join(lines), True
        return "", False

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

            local_reply, handled = self.state.offline_reply(text)
            if handled:
                self.state.brain.log_chat(text, local_reply, "offline")
                try:
                    self.state.api.activity(self.state.worker_id, "offline_reply", local_reply, {"text": text})
                except Exception:
                    pass
                response(self, 200, {"response": local_reply, "local": True, "mode": "offline"})
                return

            memory_matches = self.state.brain.search(text, limit=3)
            memory_context = ""
            if memory_matches:
                memory_context = "\n\nLocal memory context:\n" + "\n".join(f"- {item['text']}" for item in memory_matches)

            data = self.state.api.chat(text + memory_context)
            ai_text = data.get("response") or data.get("error") or ""
            if not ai_text or "Both Ollama and Gemini are unavailable" in ai_text or "connectivity issues" in ai_text:
                matches = self.state.brain.search(text) or self.state.brain.recent(5)
                if matches:
                    ai_text = "I am offline from the online model, but my local brain found:\n" + "\n".join(
                        f"- {item['text']}" for item in matches
                    )
                else:
                    ai_text = "The online model is unavailable. I saved this to my local brain and can still handle PC actions, notes, memory, files, and reminders."
            self.state.brain.remember(ai_text, "assistant")
            self.state.brain.log_chat(text, ai_text, "online")
            response(self, 200, {"response": ai_text, "local": False, "mode": "online", "dashboard": data.get("dashboard")})
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
    DesktopHandler.state.brain.remember(
        f"Desktop JARVIS started. Work folder: {DesktopHandler.state.actions.work_dir}",
        "system",
    )
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
