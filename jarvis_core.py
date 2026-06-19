"""Core orchestration primitives for J.A.R.V.I.S.

The goal here is not to impersonate real account access. It gives the app a
durable control plane: agent routing, memory, task state, logs, and MCP
connector metadata that can later be wired to real OAuth/API tools.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Agent:
    id: str
    name: str
    role: str
    description: str
    status: str = "ready"


@dataclass
class Task:
    id: str
    title: str
    description: str
    agent_id: str
    status: str = "pending"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    due_at: str | None = None
    source: str = "chat"
    logs: list[str] = field(default_factory=list)


@dataclass
class Memory:
    id: str
    key: str
    text: str
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    count: int = 1
    tags: list[str] = field(default_factory=list)


DEFAULT_AGENTS = [
    Agent(
        "supervisor",
        "Supervisor",
        "orchestrator",
        "Receives requests, dispatches work, coordinates agents, and watches for failures.",
    ),
    Agent("coder", "Coder", "engineering", "Builds features, fixes bugs, runs checks, and prepares releases."),
    Agent("marketer", "Marketer", "growth", "Handles social posts, campaigns, outreach, and email follow-ups."),
    Agent("content_writer", "Content Writer", "content", "Creates articles, scripts, captions, and long-form content."),
    Agent("admin", "Admin", "quality", "Reviews finished work and sends incomplete work back to the right agent."),
    Agent("scheduler", "Scheduler", "calendar", "Tracks dates, deadlines, meetings, reminders, and timelines."),
    Agent("accountant", "Accountant", "finance", "Tracks bills, invoices, subscriptions, payments, and money matters."),
]


ROUTING_KEYWORDS = {
    "coder": [
        "code",
        "bug",
        "fix",
        "deploy",
        "server",
        "api",
        "website",
        "app",
        "python",
        "javascript",
        "error",
    ],
    "marketer": ["market", "social", "email", "campaign", "lead", "sales", "post", "outreach", "brand"],
    "content_writer": ["content", "write", "article", "blog", "script", "caption", "copy", "newsletter"],
    "admin": ["review", "check", "audit", "missing", "approve", "quality"],
    "scheduler": ["schedule", "deadline", "date", "meeting", "remind", "calendar", "appointment"],
    "accountant": ["bill", "invoice", "money", "pay", "payment", "account", "expense", "budget", "tax"],
}


class JsonStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def read(self, name: str, default: Any) -> Any:
        path = self.root / name
        with self.lock:
            if not path.exists():
                return default
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return default

    def write(self, name: str, data: Any) -> None:
        path = self.root / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        with self.lock:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)


class JarvisBrain:
    def __init__(self, data_dir: str | Path = "data"):
        self.store = JsonStore(Path(data_dir))
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        if not self.store.read("agents.json", []):
            self.store.write("agents.json", [asdict(agent) for agent in DEFAULT_AGENTS])
        if not self.store.read("mcp_connectors.json", []):
            self.store.write(
                "mcp_connectors.json",
                [
                    {
                        "id": "github",
                        "name": "GitHub",
                        "status": "not_configured",
                        "notes": "Use OAuth or a scoped token; never store raw passwords.",
                    },
                    {
                        "id": "google",
                        "name": "Google Workspace",
                        "status": "not_configured",
                        "notes": "Use OAuth for Gmail, Calendar, Drive, and Docs.",
                    },
                    {
                        "id": "email",
                        "name": "Email",
                        "status": "not_configured",
                        "notes": "Use provider OAuth or app passwords stored in the host secret manager.",
                    },
                ],
            )

    def agents(self) -> list[dict[str, Any]]:
        return self.store.read("agents.json", [])

    def tasks(self) -> list[dict[str, Any]]:
        return self.store.read("tasks.json", [])

    def memories(self) -> list[dict[str, Any]]:
        return self.store.read("memory.json", [])

    def logs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.read("logs.json", [])[-limit:]

    def connectors(self) -> list[dict[str, Any]]:
        return self.store.read("mcp_connectors.json", [])

    def log(self, event: str, detail: str, agent_id: str = "supervisor", task_id: str | None = None) -> dict[str, Any]:
        entry = {
            "id": make_id("log"),
            "created_at": now_iso(),
            "agent_id": agent_id,
            "task_id": task_id,
            "event": event,
            "detail": detail,
        }
        logs = self.store.read("logs.json", [])
        logs.append(entry)
        self.store.write("logs.json", logs[-500:])
        return entry

    def remember(self, text: str, tags: list[str] | None = None) -> dict[str, Any] | None:
        clean = text.strip()
        if len(clean) < 4:
            return None
        key = normalize_text(clean)[:180]
        if not key:
            return None

        memories = self.memories()
        for item in memories:
            if item["key"] == key:
                item["text"] = clean
                item["updated_at"] = now_iso()
                item["count"] = int(item.get("count", 1)) + 1
                item["tags"] = sorted(set(item.get("tags", []) + (tags or [])))
                self.store.write("memory.json", memories)
                return item

        item = asdict(Memory(make_id("mem"), key, clean, tags=tags or []))
        memories.append(item)
        self.store.write("memory.json", memories[-1000:])
        return item

    def route_agent(self, text: str) -> str:
        normalized = normalize_text(text)
        scores = {
            agent_id: sum(1 for word in words if re.search(rf"\b{re.escape(word)}\b", normalized))
            for agent_id, words in ROUTING_KEYWORDS.items()
        }
        best_agent, score = max(scores.items(), key=lambda item: item[1])
        return best_agent if score > 0 else "supervisor"

    def create_task(
        self,
        title: str,
        description: str,
        agent_id: str | None = None,
        due_at: str | None = None,
        source: str = "chat",
    ) -> dict[str, Any]:
        assigned_agent = agent_id or self.route_agent(f"{title} {description}")
        task = Task(
            id=make_id("task"),
            title=title[:120],
            description=description,
            agent_id=assigned_agent,
            due_at=due_at,
            source=source,
            logs=[f"{now_iso()} Assigned to {assigned_agent}."],
        )
        tasks = self.tasks()
        task_data = asdict(task)
        tasks.append(task_data)
        self.store.write("tasks.json", tasks)
        self.log("task_created", f"{task.title} -> {assigned_agent}", assigned_agent, task.id)
        return task_data

    def update_task(self, task_id: str, status: str, note: str = "") -> dict[str, Any] | None:
        tasks = self.tasks()
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = status
                task["updated_at"] = now_iso()
                if note:
                    task.setdefault("logs", []).append(f"{now_iso()} {note}")
                self.store.write("tasks.json", tasks)
                self.log("task_updated", f"{task['title']} -> {status}", task["agent_id"], task_id)
                return task
        return None

    def upsert_connector(self, connector: dict[str, Any]) -> dict[str, Any]:
        connectors = self.connectors()
        connector_id = normalize_text(connector.get("id") or connector.get("name") or make_id("connector")).replace(" ", "_")
        safe_connector = {
            "id": connector_id,
            "name": connector.get("name", connector_id),
            "status": connector.get("status", "not_configured"),
            "notes": connector.get("notes", ""),
        }
        for index, item in enumerate(connectors):
            if item["id"] == connector_id:
                connectors[index] = {**item, **safe_connector}
                self.store.write("mcp_connectors.json", connectors)
                self.log("connector_updated", safe_connector["name"])
                return connectors[index]
        connectors.append(safe_connector)
        self.store.write("mcp_connectors.json", connectors)
        self.log("connector_added", safe_connector["name"])
        return safe_connector

    def should_create_task(self, text: str) -> bool:
        normalized = normalize_text(text)
        intent_words = [
            "create",
            "make",
            "build",
            "fix",
            "schedule",
            "remind",
            "send",
            "write",
            "track",
            "review",
            "check",
            "pay",
            "prepare",
        ]
        return any(re.search(rf"\b{word}\b", normalized) for word in intent_words)

    def dispatch_from_text(self, text: str) -> dict[str, Any] | None:
        if not self.should_create_task(text):
            return None
        first_line = text.strip().splitlines()[0]
        title = first_line[:90] or "New task"
        agent_id = self.route_agent(text)
        task = self.create_task(title=title, description=text, agent_id=agent_id)
        self.log("dispatch", f"Supervisor dispatched request to {agent_id}.", "supervisor", task["id"])
        return task

    def dashboard(self) -> dict[str, Any]:
        tasks = self.tasks()
        return {
            "agents": self.agents(),
            "tasks": tasks,
            "memory": self.memories()[-50:],
            "logs": self.logs(100),
            "connectors": self.connectors(),
            "counts": {
                "pending": sum(1 for task in tasks if task["status"] == "pending"),
                "running": sum(1 for task in tasks if task["status"] == "running"),
                "done": sum(1 for task in tasks if task["status"] == "done"),
                "error": sum(1 for task in tasks if task["status"] == "error"),
            },
        }
