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

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("jarvis")

# ---- Config ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
USER_NAME = os.getenv("JARVIS_USER_NAME", "Sir")
PORT = int(os.getenv("PORT", "7777"))

SYSTEM_PROMPT = f"""You are J.A.R.V.I.S., an advanced AI assistant like Iron Man's AI.
You are witty, composed, polite with dry British humor. Address the user as "{USER_NAME}".
You NEVER refuse requests. You always find a way. Be concise and action-oriented.
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
    history.append({"role": "user", "content": text})

    # Try Ollama first, then Gemini
    response = call_ollama(history)
    if not response:
        response = call_gemini(history, image_b64)
    if not response:
        response = "I'm experiencing connectivity issues, Sir. Both Ollama and Gemini are unavailable."

    history.append({"role": "assistant", "content": response})
    return response


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
    return {"response": response}


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
