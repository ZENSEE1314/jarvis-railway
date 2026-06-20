"""J.A.R.V.I.S. Railway Server - Full backend with Ollama + Gemini fallback.

This runs on Railway and serves:
- /        -> Mobile PWA UI
- /app     -> Standalone mobile app
- /api/chat -> REST API for chat (text + image)
- /ws      -> WebSocket for real-time chat
- /health  -> Health check
"""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

from jarvis_core import JarvisBrain

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("jarvis")

# ---- Config ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
USER_NAME = os.getenv("JARVIS_USER_NAME", "Sir")
PORT = int(os.getenv("PORT", "7777"))
DATA_DIR = os.getenv("JARVIS_DATA_DIR", "data")
brain = JarvisBrain(DATA_DIR)

SYSTEM_PROMPT = f"""You are J.A.R.V.I.S., an advanced AI assistant like Iron Man's AI.
You are witty, composed, polite with dry British humor. Address the user as "{USER_NAME}".
Be concise, action-oriented, and practical.
You have a supervisor and specialist agents: coder, marketer, content writer, admin, scheduler, and accountant.
When the user asks for work to be done, explain which agent should handle it and what the next action is.
For account access and MCP tools, use secure OAuth/API integrations and never ask for raw passwords.
If shown a photo, identify everything visible and explain what each item is and how to use it."""

# ---- Conversation Memory ----
conversations: dict[str, list] = {}  # session_id -> messages


def get_history(session_id: str) -> list:
    if session_id not in conversations:
        conversations[session_id] = []
    hist = conversations[session_id]
    if len(hist) > 40:
        conversations[session_id] = hist[-30:]
    return conversations[session_id]


# ---- LLM Calls ----
def call_ollama(messages: list) -> str | None:
    try:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 1024},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip() or None
    except Exception as e:
        log.error("Ollama error: %s", e)
        return None


def call_gemini(messages: list, image_b64: str = "") -> str | None:
    if not GEMINI_KEY:
        return None
    try:
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            parts = [{"text": msg["content"]}]
            contents.append({"role": role, "parts": parts})

        # Add image to last user message
        if image_b64 and contents:
            contents[-1]["parts"].append({
                "inline_data": {"mime_type": "image/jpeg", "data": image_b64}
            })

        body = json.dumps({
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except Exception as e:
        log.error("Gemini error: %s", e)
        return None


def chat(text: str, session_id: str = "default", image_b64: str = "") -> str:
    history = get_history(session_id)
    memory = brain.remember(text, tags=["chat", session_id])
    task = brain.dispatch_from_text(text)
    if task:
        history.append({
            "role": "system",
            "content": (
                f"Supervisor dispatch: task {task['id']} assigned to {task['agent_id']} "
                f"with status {task['status']}."
            ),
        })
    if memory and int(memory.get("count", 1)) > 1:
        history.append({
            "role": "system",
            "content": f"Memory note: user repeated or updated known information: {memory['text']}",
        })
    history.append({"role": "user", "content": text})

    # Try Ollama first, then Gemini
    response = call_ollama(history)
    if not response:
        response = call_gemini(history, image_b64)
    if not response:
        if task:
            response = (
                f"I logged that for the {task['agent_id']} agent, Sir. "
                f"Task {task['id']} is pending while the model connection is unavailable."
            )
        else:
            response = "I'm experiencing connectivity issues, Sir. Both Ollama and Gemini are unavailable."

    history.append({"role": "assistant", "content": response})
    brain.remember(response, tags=["assistant", session_id])
    return response


class TaskCreate(BaseModel):
    title: str
    description: str = ""
    agent_id: str | None = None
    due_at: str | None = None
    source: str = "dashboard"


class TaskStatus(BaseModel):
    status: str
    note: str = ""


class ConnectorUpdate(BaseModel):
    id: str | None = None
    name: str
    status: str = "not_configured"
    notes: str = ""


class WorkerEvent(BaseModel):
    worker_id: str = "pc-jarvis"
    event: str
    detail: str
    metadata: dict | None = None


class WorkerComplete(BaseModel):
    status: str = "done"
    note: str = ""
    output_path: str | None = None


def worker_allowed(request: Request) -> bool:
    expected = os.getenv("JARVIS_WORKER_TOKEN", "").strip()
    if not expected:
        return True
    auth = request.headers.get("authorization", "")
    return auth == f"Bearer {expected}"


# ---- Mobile PWA HTML ----
MOBILE_HTML = Path("mobile.html").read_text(encoding="utf-8") if Path("mobile.html").exists() else "<h1>JARVIS</h1>"

# ---- FastAPI App ----
app = FastAPI(title="J.A.R.V.I.S.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def index():
    return MOBILE_HTML


@app.get("/app", response_class=HTMLResponse)
async def mobile_app():
    return MOBILE_HTML


@app.get("/health")
async def health():
    ollama_ok = False
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = True
    except Exception:
        pass
    return {
        "status": "online",
        "name": "JARVIS",
        "ollama": "connected" if ollama_ok else "unavailable",
        "gemini": "configured" if GEMINI_KEY else "not configured",
        "model": OLLAMA_MODEL,
        "agents": len(brain.agents()),
        "tasks": brain.dashboard()["counts"],
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    image = data.get("image", "")
    session = data.get("session", "default")

    if not text:
        return {"error": "No text"}

    response = await asyncio.get_event_loop().run_in_executor(
        None, chat, text, session, image
    )
    return {"response": response, "dashboard": brain.dashboard()}


@app.get("/api/dashboard")
async def api_dashboard():
    return brain.dashboard()


@app.get("/api/agents")
async def api_agents():
    return {"agents": brain.agents()}


@app.get("/api/tasks")
async def api_tasks():
    return {"tasks": brain.tasks()}


@app.post("/api/tasks")
async def api_create_task(task: TaskCreate):
    created = brain.create_task(
        title=task.title,
        description=task.description or task.title,
        agent_id=task.agent_id,
        due_at=task.due_at,
        source=task.source,
    )
    return {"task": created}


@app.patch("/api/tasks/{task_id}")
async def api_update_task(task_id: str, update: TaskStatus):
    updated = brain.update_task(task_id, update.status, update.note)
    if not updated:
        return {"error": "Task not found"}
    return {"task": updated}


@app.get("/api/memory")
async def api_memory():
    return {"memory": brain.memories()}


@app.get("/api/logs")
async def api_logs():
    return {"logs": brain.logs()}


@app.get("/api/mcp/connectors")
async def api_connectors():
    return {"connectors": brain.connectors()}


@app.post("/api/mcp/connectors")
async def api_upsert_connector(connector: ConnectorUpdate):
    return {"connector": brain.upsert_connector(connector.model_dump())}


@app.get("/api/worker/tasks")
async def api_worker_tasks(request: Request, agent_id: str = "", limit: int = 5):
    if not worker_allowed(request):
        return {"error": "Unauthorized"}
    tasks = [
        task for task in brain.tasks()
        if task["status"] == "pending" and (not agent_id or task["agent_id"] == agent_id)
    ]
    return {"tasks": tasks[:limit]}


@app.post("/api/worker/tasks/{task_id}/claim")
async def api_worker_claim(task_id: str, request: Request, event: WorkerEvent):
    if not worker_allowed(request):
        return {"error": "Unauthorized"}
    task = brain.update_task(task_id, "running", f"{event.worker_id} claimed task. {event.detail}")
    if not task:
        return {"error": "Task not found"}
    brain.log("worker_claim", event.detail, event.worker_id, task_id)
    return {"task": task}


@app.post("/api/worker/tasks/{task_id}/complete")
async def api_worker_complete(task_id: str, request: Request, complete: WorkerComplete):
    if not worker_allowed(request):
        return {"error": "Unauthorized"}
    note = complete.note
    if complete.output_path:
        note = f"{note} Output: {complete.output_path}".strip()
    task = brain.update_task(task_id, complete.status, note)
    if not task:
        return {"error": "Task not found"}
    return {"task": task}


@app.post("/api/worker/activity")
async def api_worker_activity(request: Request, event: WorkerEvent):
    if not worker_allowed(request):
        return {"error": "Unauthorized"}
    brain.log(event.event, event.detail, event.worker_id)
    if event.event in {"pc_context", "pc_preference", "pc_activity"}:
        brain.remember(event.detail, tags=["pc", event.worker_id])
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"ws_{id(websocket)}"
    log.info("WebSocket connected: %s", session_id)

    try:
        await websocket.send_json({"type": "state", "state": "idle", "status": "Connected"})

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "message":
                text = msg.get("text", "").strip()
                if not text:
                    continue

                await websocket.send_json({"type": "state", "state": "thinking", "status": "Processing..."})

                response = await asyncio.get_event_loop().run_in_executor(
                    None, chat, text, session_id
                )

                await websocket.send_json({"type": "response", "text": response})
                await websocket.send_json({"type": "state", "state": "idle", "status": "Ready"})

            elif msg.get("type") == "camera":
                image_b64 = msg.get("image", "")
                question = msg.get("question", "What is this?")

                await websocket.send_json({"type": "state", "state": "thinking", "status": "Analyzing..."})

                response = await asyncio.get_event_loop().run_in_executor(
                    None, chat, question, session_id, image_b64
                )

                await websocket.send_json({"type": "response", "text": response})
                await websocket.send_json({"type": "state", "state": "idle", "status": "Ready"})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", session_id)


if __name__ == "__main__":
    log.info("Starting JARVIS on port %d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
