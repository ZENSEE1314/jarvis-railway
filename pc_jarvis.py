"""Local PC worker for J.A.R.V.I.S.

Run this on your Windows PC. Railway stays the online brain/dashboard; this
worker is the local hands that can open apps/files/URLs, save outputs, and
report progress back.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SERVER = "https://jarvis-railway-production-42ff.up.railway.app"
DEFAULT_WORK_DIR = Path.home() / "Documents" / "Jarvis Work"


def now_slug() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def today_dir(root: Path) -> Path:
    path = root / datetime.now().strftime("%Y-%m-%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


class JarvisApi:
    def __init__(self, server_url: str, token: str = ""):
        self.server_url = server_url.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.server_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            return {"error": f"HTTP {exc.code}: {text}"}
        except Exception as exc:
            return {"error": str(exc)}

    def chat(self, text: str) -> dict[str, Any]:
        return self.request("POST", "/api/chat", {"text": text, "session": "pc-jarvis"})

    def create_task(self, text: str, agent_id: str = "coder") -> dict[str, Any]:
        return self.request(
            "POST",
            "/api/tasks",
            {
                "title": text[:100],
                "description": text,
                "agent_id": agent_id,
                "source": "pc-jarvis",
            },
        )

    def pending_tasks(self, agent_id: str = "", limit: int = 5) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"agent_id": agent_id, "limit": str(limit)})
        data = self.request("GET", f"/api/worker/tasks?{query}")
        return data.get("tasks", [])

    def claim(self, task_id: str, worker_id: str, detail: str) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/worker/tasks/{task_id}/claim",
            {"worker_id": worker_id, "event": "worker_claim", "detail": detail},
        )

    def complete(self, task_id: str, status: str, note: str, output_path: str = "") -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/worker/tasks/{task_id}/complete",
            {"status": status, "note": note, "output_path": output_path or None},
        )

    def activity(self, worker_id: str, event: str, detail: str, metadata: dict[str, Any] | None = None) -> None:
        self.request(
            "POST",
            "/api/worker/activity",
            {"worker_id": worker_id, "event": event, "detail": detail, "metadata": metadata or {}},
        )


class PcActions:
    def __init__(self, work_dir: Path, speak_enabled: bool = False):
        self.work_dir = work_dir
        self.speak_enabled = speak_enabled
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def say(self, text: str) -> None:
        if not self.speak_enabled or os.name != "nt" or not text:
            return
        safe = text[:700].replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.Speak('{safe}')"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def listen_once(self) -> str:
        try:
            import speech_recognition as sr  # type: ignore
        except Exception:
            return "Voice input needs optional packages: pip install SpeechRecognition pyaudio"

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=20)
        try:
            return recognizer.recognize_google(audio)
        except Exception as exc:
            return f"Could not understand voice: {exc}"

    def run_command(self, command: str) -> str:
        lowered = command.lower().strip()
        if lowered.startswith(("open ", "launch ")):
            return self.open_target(command.split(" ", 1)[1].strip())
        if lowered.startswith(("note ", "save note ")):
            text = re.sub(r"^(save note|note)\s+", "", command, flags=re.I).strip()
            return self.save_note(text)
        if lowered in {"where work", "work folder", "open work folder"}:
            self.open_target(str(self.work_dir))
            return f"Opened work folder: {self.work_dir}"
        if lowered.startswith("remember "):
            return f"Memory request noted: {command[9:].strip()}"
        return self.save_note(f"Manual command captured for follow-up:\n\n{command}")

    def open_target(self, target: str) -> str:
        if not target:
            return "No target provided."
        if re.match(r"^https?://", target, re.I):
            webbrowser.open(target)
            return f"Opened URL: {target}"

        expanded = Path(os.path.expandvars(os.path.expanduser(target.strip('"'))))
        if expanded.exists():
            os.startfile(str(expanded))  # type: ignore[attr-defined]
            return f"Opened path: {expanded}"

        try:
            subprocess.Popen(target, shell=True)
            return f"Launched: {target}"
        except Exception as exc:
            return f"Could not open {target}: {exc}"

    def save_note(self, text: str, task_id: str = "") -> str:
        folder = today_dir(self.work_dir)
        filename = f"{now_slug()}_{task_id or 'note'}.md"
        path = folder / filename
        path.write_text(f"# JARVIS Note\n\n{text.strip()}\n", encoding="utf-8")
        return str(path)

    def execute_task(self, task: dict[str, Any]) -> tuple[str, str, str]:
        description = task.get("description") or task.get("title") or ""
        task_id = task.get("id", "task")
        folder = today_dir(self.work_dir) / task_id
        folder.mkdir(parents=True, exist_ok=True)

        result = self.run_command(description)
        summary_path = folder / "result.md"
        summary_path.write_text(
            "\n".join(
                [
                    "# JARVIS PC Task Result",
                    "",
                    f"- Task: {task.get('title', '')}",
                    f"- Agent: {task.get('agent_id', '')}",
                    f"- Status: done",
                    f"- Result: {result}",
                    "",
                    "## Original Request",
                    "",
                    description,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return "done", result, str(summary_path)


def foreground_window_title() -> str:
    if os.name != "nt":
        return ""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.strip()


def run_awareness(api: JarvisApi, worker_id: str, interval: int) -> None:
    print("PC awareness is ON: foreground window titles only. No keystrokes, passwords, or screen capture.")
    last_title = ""
    while True:
        title = foreground_window_title()
        if title and title != last_title:
            print(f"[aware] {title}")
            api.activity(worker_id, "pc_activity", f"Active window: {title}", {"window_title": title})
            last_title = title
        time.sleep(interval)


def poll_once(api: JarvisApi, actions: PcActions, worker_id: str, agent_id: str) -> int:
    tasks = api.pending_tasks(agent_id=agent_id, limit=3)
    if not tasks:
        return 0
    completed = 0
    for task in tasks:
        task_id = task["id"]
        print(f"Claiming {task_id}: {task.get('title')}")
        api.claim(task_id, worker_id, "PC worker started task.")
        try:
            status, note, output = actions.execute_task(task)
        except Exception as exc:
            status, note, output = "error", f"PC worker error: {exc}", ""
        api.complete(task_id, status, note, output)
        print(f"{status.upper()}: {note}")
        if output:
            print(f"Saved: {output}")
        completed += 1
    return completed


def interactive(api: JarvisApi, actions: PcActions, worker_id: str) -> None:
    print("PC JARVIS ready. Type 'help' for commands, 'quit' to exit.")
    actions.say("PC JARVIS is ready.")
    while True:
        try:
            text = input("JARVIS PC> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            return
        if text.lower() == "help":
            print("Commands: chat <text>, task <text>, open <url/path/app>, note <text>, listen, poll, work folder, status")
            continue
        if text.lower() == "listen":
            heard = actions.listen_once()
            print(f"Heard: {heard}")
            if heard.startswith("Voice input needs") or heard.startswith("Could not"):
                actions.say(heard)
                continue
            data = api.chat(heard)
            response = data.get("response") or data.get("error") or ""
            print(response)
            actions.say(response)
            continue
        if text.lower() == "status":
            health = api.request("GET", "/health")
            print(json.dumps(health, indent=2))
            continue
        if text.lower() == "poll":
            count = poll_once(api, actions, worker_id, "")
            print(f"Processed {count} task(s).")
            continue
        if text.lower().startswith("chat "):
            data = api.chat(text[5:].strip())
            response = data.get("response") or data.get("error") or ""
            print(response)
            actions.say(response)
            continue
        if text.lower().startswith("task "):
            data = api.create_task(text[5:].strip())
            print(json.dumps(data, indent=2))
            continue
        result = actions.run_command(text)
        api.activity(worker_id, "pc_command", result, {"command": text})
        print(result)
        actions.say(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PC JARVIS worker.")
    parser.add_argument("--server", default=os.getenv("JARVIS_SERVER_URL", DEFAULT_SERVER))
    parser.add_argument("--token", default=os.getenv("JARVIS_WORKER_TOKEN", ""))
    parser.add_argument("--worker-id", default=os.getenv("JARVIS_WORKER_ID", "pc-jarvis"))
    parser.add_argument("--work-dir", default=os.getenv("JARVIS_WORK_DIR", str(DEFAULT_WORK_DIR)))
    parser.add_argument("--poll", action="store_true", help="Continuously poll Railway tasks.")
    parser.add_argument("--agent", default=os.getenv("JARVIS_WORKER_AGENT", ""))
    parser.add_argument("--interval", type=int, default=int(os.getenv("JARVIS_WORKER_INTERVAL", "20")))
    parser.add_argument("--awareness", action="store_true", help="Log foreground window titles to JARVIS memory.")
    parser.add_argument("--speak", action="store_true", help="Speak responses using Windows text-to-speech.")
    args = parser.parse_args()

    api = JarvisApi(args.server, args.token)
    actions = PcActions(Path(args.work_dir), speak_enabled=args.speak)
    api.activity(args.worker_id, "pc_worker_online", f"PC worker online. Work folder: {actions.work_dir}")

    if args.awareness:
        run_awareness(api, args.worker_id, max(args.interval, 5))
        return 0

    if args.poll:
        print(f"Polling {args.server} every {args.interval}s. Work folder: {actions.work_dir}")
        while True:
            poll_once(api, actions, args.worker_id, args.agent)
            time.sleep(args.interval)

    interactive(api, actions, args.worker_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
