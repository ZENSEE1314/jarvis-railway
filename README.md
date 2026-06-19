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

## MCP And Accounts

The app stores connector status and notes, not raw passwords. Real account control should be added through secure MCP servers, OAuth, scoped API tokens, or Railway secrets. Add connectors through `/api/mcp/connectors`, then wire the corresponding MCP tool server behind the agent that needs it.
