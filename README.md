# JARVIS Railway

Mobile-first J.A.R.V.I.S. server for Railway with:

- voice and typed chat input
- Ollama first, Gemini fallback
- persistent memory with duplicate merging
- supervisor dispatch to specialist agents
- task dashboard with pending, running, done, and error states
- logs for task creation, routing, and updates
- MCP connector metadata for future OAuth/API integrations

## Run locally

```bash
pip install -r requirements.txt
python server.py
```

Open `http://localhost:7777`.

## Ollama Codex Model

Default Ollama model:

```bash
kimi-k2.7-code:cloud
```

Launch Codex with the same Ollama cloud model:

```bash
ollama launch codex --model kimi-k2.7-code:cloud
```

JARVIS uses this model through `OLLAMA_MODEL`. Override it in `.env` or Railway variables when you want a different model.

## Environment

Copy `env.example` to `.env` or set the same values in Railway.

- `OLLAMA_URL`: Ollama server URL.
- `OLLAMA_MODEL`: model name for Ollama chat.
- `GEMINI_API_KEY`: optional fallback key.
- `JARVIS_USER_NAME`: how JARVIS addresses you.
- `JARVIS_DATA_DIR`: folder for JSON memory, tasks, logs, agents, and MCP connector records.
- `PORT`: server port. Railway sets this automatically.

## Agent Setup

The first run creates these agents in `JARVIS_DATA_DIR`:

- Supervisor: routes work and watches status.
- Coder: coding, bug fixes, deployment, checks.
- Marketer: social, outreach, campaigns, emails.
- Content Writer: articles, scripts, captions, copy.
- Admin: review and quality checks.
- Scheduler: dates, deadlines, reminders, meetings.
- Accountant: bills, invoices, budgets, money tracking.

## PC JARVIS Worker

Railway is the online brain/dashboard. The PC worker is the local hands that
can run on your Windows computer and report back to Railway.

Work is saved by default to:

```text
%USERPROFILE%\Documents\Jarvis Work
```

Run interactive PC JARVIS:

```powershell
.\run_pc_jarvis.ps1
```

Or double-click:

```text
run_pc_jarvis.bat
```

Build a Windows `.exe`:

```powershell
.\build_pc_jarvis_exe.ps1
```

Or double-click:

```text
build_pc_jarvis_exe.bat
```

The executable will be created at:

```text
dist\PC-JARVIS\PC-JARVIS.exe
```

The interactive launcher speaks replies with Windows text-to-speech. Type
`listen` for microphone input after installing optional voice packages:

```powershell
pip install SpeechRecognition pyaudio
```

Useful interactive commands:

```text
chat help me plan today
task fix the website bug
open https://github.com/ZENSEE1314/jarvis-railway
open notepad
note remember this idea for my business
listen
poll
work folder
status
```

Run the background task worker:

```powershell
.\run_pc_worker.ps1
```

This polls Railway for pending tasks, claims them, performs safe local actions
that it understands, writes a result file, and updates the dashboard.

Run optional PC awareness:

```powershell
.\run_pc_awareness.ps1
```

Awareness mode logs foreground window titles only, so JARVIS can learn context
like which app or project you are working in. It does not record keystrokes,
passwords, screenshots, or hidden screen content.

For private worker endpoints, set `JARVIS_WORKER_TOKEN` in Railway and set the
same local environment variable before running the PC worker.

## MCP And Accounts

The app stores connector status and notes, not raw passwords. Real account control should be added through secure MCP servers, OAuth, scoped API tokens, or Railway secrets. Add connectors through `/api/mcp/connectors`, then wire the corresponding MCP tool server behind the agent that needs it.
