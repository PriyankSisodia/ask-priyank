# Deploy Ask Priyank (single-file app)

All behavior and step-by-step setup live in the **module docstring at the top of `app.py`**. Open that file first: it covers Git hygiene, Google Sheets (logging + `twin_context` tab), hiding `summary.txt` / PDF from GitHub, and Render free tier.

## Files in this folder

| File | Purpose |
|------|---------|
| `app.py` | Gradio chat, Gemini calls, Sheets read/write — **no other Python files required** |
| `requirements.txt` | Dependencies for `pip install` on Render or locally |
| `env.example` | Copy to `.env` for local secrets (do not commit), if present |
| `render.yaml` | Optional Render Blueprint |

## Quick Render settings

- **Root directory:** `_agentic_ai/ask priyank/deploy`
- **Build:** `pip install -r requirements.txt`
- **Start:** `python app.py`

## Local run

```bash
cd _agentic_ai/ask\ priyank/deploy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Create .env with GEMINI_API_KEY (or GOOGLE_API_KEY) and optional vars from app.py docstring
python app.py
```

## Logs and debugging

- **Local:** Run `python app.py` in a terminal. Every log line is printed there with timestamps (`ask_priyank` logger).
- **Render:** Dashboard → your Web Service → **Logs** (live tail). Same messages appear after deploy.
- **Jupyter:** If you use a notebook instead, watch the cell output where `launch()` runs.

Useful environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` to see context cache hits and more detail. |
| `LLM_HTTP_TIMEOUT` | `120` | Max seconds per Gemini HTTP call; avoids hanging forever on network issues. |
| `CHAT_HEARTBEAT_SEC` | `60` | While waiting on Gemini, a **WARNING** is logged every N seconds (e.g. “1 min elapsed”). Set to `0` to disable. |

On startup the app logs whether the Gemini key is set (length only, never the secret), Sheet/URL flags, and timeouts.

## Privacy

Emails and unknown questions may be appended to Google Sheets; disclose on your public page if required (e.g. GDPR). Never commit `GSPREAD_SERVICE_ACCOUNT_JSON`.
