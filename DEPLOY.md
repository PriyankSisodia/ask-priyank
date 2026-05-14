# Deploy Ask Priyank (single-file app)

All behavior and step-by-step setup live in the **module docstring at the top of `app.py`**. Open that file first: it covers Git hygiene, Google Sheets (logging + `twin_context` tab), hiding `summary.txt` / PDF from GitHub, and Render free tier.

## Files in this folder

| File | Purpose |
|------|---------|
| `app.py` | Gradio chat, Gemini calls, Sheets read/write — **no other Python files required** |
| `requirements.txt` | Dependencies for `pip install` on Render or locally |
| `env.example` | Copy to `.env` for local secrets (do not commit) |
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
cp env.example .env   # then edit .env
python app.py
```

Privacy: emails and unknown questions may be appended to Google Sheets; disclose on your public page if needed.
