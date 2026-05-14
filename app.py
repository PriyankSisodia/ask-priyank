"""
================================================================================
Ask Priyank — digital twin (single-file web app)
================================================================================

This file is enough to run the chatbot: no other Python modules from this repo
are required. Deploy from this folder with ``pip install -r requirements.txt``
and ``python app.py``.

--------------------------------------------------------------------------------
STEP A — Keep secrets off GitHub
--------------------------------------------------------------------------------

1. Add to your repo root ``.gitignore`` (if not already there)::

    _agentic_ai/ask priyank/summary.txt
    _agentic_ai/ask priyank/priyank_sisodia.pdf

2. If those files were ever committed, remove them from Git tracking (keeps
   local copies)::

    git rm --cached "_agentic_ai/ask priyank/summary.txt" 2>/dev/null || true
    git rm --cached "_agentic_ai/ask priyank/priyank_sisodia.pdf" 2>/dev/null || true

3. Store your real biography text **outside** the repo using one of the options
   in STEP C (recommended: Google Sheet tab ``twin_context`` or Render env vars).

--------------------------------------------------------------------------------
STEP B — Google Sheet for contacts + unknown questions (optional but recommended)
--------------------------------------------------------------------------------

1. Google Cloud Console → create project → enable **Google Sheets API**.
2. IAM → Service Accounts → create → Keys → JSON. Copy ``client_email``.
3. New Google Sheet → **Share** with that email as **Editor**.
4. Copy spreadsheet ID from the URL::
   https://docs.google.com/spreadsheets/d/1NEphTLimwNlo_28Bx2T78oP26NkiPzgDMoQy6ebSA6k/edit

5. In Render (or ``.env`` locally), set **one line** JSON::

    GOOGLE_SHEET_ID=<id>
    GSPREAD_SERVICE_ACCOUNT_JSON={"type":"service_account",...}

   The app appends to tabs ``unknown_questions`` and ``user_details`` (created
   automatically). For **hidden twin text**, add a tab named ``twin_context``:
   row1 headers ``key`` | ``value`` then rows::

    summary | <your summary paragraph(s)>
    resume  | <text from CV/LinkedIn/PDF — paste plain text>

--------------------------------------------------------------------------------
STEP C — Where the model reads “summary” + “resume” (sources merge)
--------------------------------------------------------------------------------
!pip install -r requirements.txt
Values are combined in order; each step only fills fields still empty:

1. **Sheet tab ``twin_context``** (needs ``GOOGLE_SHEET_ID`` + JSON key) — edit in
   Sheets; re-read every ``CONTEXT_CACHE_SECONDS`` (default 120).
2. **``CONTEXT_URL``** — JSON ``{"summary":"...","resume":"..."}`` or plain text
   with a line ``---RESUME---`` between summary and resume.
3. **``TWIN_SUMMARY``** and **``TWIN_RESUME_TEXT``** environment variables.
4. **Local dev only:** ``ALLOW_LOCAL_CONTEXT_FILES=1`` reads ``../summary.txt`` and
   optional ``../priyank_sisodia.pdf``. Do **not** enable in production without those files on the server.

--------------------------------------------------------------------------------
STEP D — Deploy on Render (free tier)
--------------------------------------------------------------------------------

1. Push the repo to GitHub (without secrets or private context files).
2. Render → **New** → **Web Service** → connect repo.
3. Settings::

    Root Directory:  _agentic_ai/ask priyank/deploy
    Build Command:   pip install -r requirements.txt
    Start Command:   python app.py

4. Environment variables (minimum)::

    GEMINI_API_KEY=<key>          # or GOOGLE_API_KEY
    CHAT_MODEL=gemini-2.5-flash-lite   # optional
    TWIN_DISPLAY_NAME=Priyank Sisodia  # optional
    LLM_HTTP_TIMEOUT=120          # optional: HTTP timeout per Gemini request (seconds)
    CHAT_HEARTBEAT_SEC=60         # optional: log WARNING every N sec while waiting (0=off)
    LOG_LEVEL=INFO                # optional: DEBUG for more verbose logs

   Plus one context source from STEP C, and optionally Sheet vars from STEP B.

5. **Create Web Service** and open the URL. Cold starts on free tier are normal.

--------------------------------------------------------------------------------
Privacy
--------------------------------------------------------------------------------

The model may log emails and unanswered questions to Google Sheets. Say so on
your public page if required (e.g. GDPR). Never commit ``GSPREAD_SERVICE_ACCOUNT_JSON``.

================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# -----------------------------------------------------------------------------
# Optional: load .env from this directory (local dev)
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

import gradio as gr
from openai import OpenAI

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | ask_priyank | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ask_priyank")

_DEPLOY_DIR = Path(__file__).resolve().parent
_PARENT = _DEPLOY_DIR.parent

# --- Public config (all overridable via environment) --------------------------------
PORT = int(os.environ.get("PORT", "7860"))
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-2.5-flash-lite")
# HTTP timeout for each Gemini request (seconds). Stops “hanging forever” on network stalls.
LLM_HTTP_TIMEOUT = float(os.environ.get("LLM_HTTP_TIMEOUT", "120"))
# While the model is thinking, log a WARNING every N seconds (0 = disabled).
CHAT_HEARTBEAT_SEC = float(os.environ.get("CHAT_HEARTBEAT_SEC", "60"))
NAME = os.environ.get("TWIN_DISPLAY_NAME", "Priyank Sisodia")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
CONTEXT_URL = os.environ.get("CONTEXT_URL", "").strip()
CONTEXT_CACHE_SECONDS = int(os.environ.get("CONTEXT_CACHE_SECONDS", "120"))
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "").strip()
print(f"GOOGLE_SHEET_ID: {GOOGLE_SHEET_ID}")
GSPREAD_JSON = os.environ.get("GSPREAD_SERVICE_ACCOUNT_JSON", "").strip()
ALLOW_LOCAL_FILES = os.environ.get("ALLOW_LOCAL_CONTEXT_FILES", "").lower() in (
    "1",
    "true",
    "yes",
)


class _LongWaitHeartbeat:
    """Log every ``interval_sec`` while a blocking HTTP call runs (watch the terminal)."""

    def __init__(self, interval_sec: float, label: str) -> None:
        self.interval_sec = max(0.0, float(interval_sec))
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_sec <= 0:
            return

        def _run() -> None:
            n = 0
            while not self._stop.wait(self.interval_sec):
                n += 1
                log.warning("%s — still waiting (%d min elapsed)", self.label, n)

        self._thread = threading.Thread(target=_run, daemon=True, name="llm-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def log_startup_diagnostics() -> None:
    """Run once at launch; helps confirm env and keys without printing secrets."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    log.info("=== Ask Priyank startup ===")
    log.info("PORT=%s CHAT_MODEL=%s", PORT, CHAT_MODEL)
    log.info(
        "Gemini API key: %s",
        "set (%d chars)" % len(key) if key else "MISSING — set GEMINI_API_KEY or GOOGLE_API_KEY",
    )
    log.info("LLM_HTTP_TIMEOUT=%s CHAT_HEARTBEAT_SEC=%s", LLM_HTTP_TIMEOUT, CHAT_HEARTBEAT_SEC)
    log.info(
        "Google Sheet: GOOGLE_SHEET_ID=%s GSPREAD_JSON=%s",
        "set" if GOOGLE_SHEET_ID else "unset",
        "set" if GSPREAD_JSON else "unset",
    )
    log.info("CONTEXT_URL=%s ALLOW_LOCAL_CONTEXT_FILES=%s", bool(CONTEXT_URL), ALLOW_LOCAL_FILES)
    log.info("Logs: watch this terminal (stdout). On Render: Dashboard → your service → Logs.")
    log.info("=== end startup ===")


_SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_gspread_client = None
_context_cache: tuple[str, str, float] = ("", "", 0.0)  # summary, resume, monotonic_ts


# =============================================================================
# 1) Twin context — never require repo files in production
# =============================================================================


def _fetch_url_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ask-priyank-twin/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_context_payload(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    if not raw:
        return "", ""
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return (str(data.get("summary", "")), str(data.get("resume", "")))
        except json.JSONDecodeError:
            pass
    if "---RESUME---" in raw:
        a, b = raw.split("---RESUME---", 1)
        return a.strip(), b.strip()
    return raw, ""


def _gspread_authorize():
    global _gspread_client
    if _gspread_client is not None:
        return _gspread_client
    if not GSPREAD_JSON or not GOOGLE_SHEET_ID:
        return None
    import gspread
    from google.oauth2.service_account import Credentials

    info = json.loads(GSPREAD_JSON)
    creds = Credentials.from_service_account_info(info, scopes=_SHEETS_SCOPES)
    _gspread_client = gspread.authorize(creds)
    return _gspread_client


def _load_context_from_sheet() -> tuple[str, str] | None:
    if not GOOGLE_SHEET_ID or not GSPREAD_JSON:
        return None
    log.info("Context: reading Google Sheet twin_context…")
    try:
        import gspread

        gc = _gspread_authorize()
        if gc is None:
            return None
        sh = gc.open_by_key(GOOGLE_SHEET_ID)
        try:
            ws = sh.worksheet("twin_context")
        except gspread.WorksheetNotFound:
            return None
        rows = ws.get_all_values()
        if not rows:
            return None
        summary, resume = "", ""
        for i, row in enumerate(rows):
            if i == 0 and row and str(row[0]).lower() in ("key", "k"):
                continue
            if len(row) >= 2:
                k = str(row[0]).strip().lower()
                v = str(row[1]).strip()
                if k == "summary":
                    summary = v
                elif k in ("resume", "cv", "linkedin", "pdf_text"):
                    resume = v
        if summary or resume:
            return summary, resume
    except Exception:
        log.exception("Could not read twin_context from Google Sheet")
    return None


def _load_context_from_env() -> tuple[str, str]:
    return (
        os.environ.get("TWIN_SUMMARY", "").strip(),
        os.environ.get("TWIN_RESUME_TEXT", "").strip(),
    )


def _load_context_local_files() -> tuple[str, str]:
    summary, resume = "", ""
    sp = _PARENT / "summary.txt"
    if sp.exists():
        summary = sp.read_text(encoding="utf-8", errors="replace")
    pdf = _PARENT / "priyank_sisodia.pdf"
    if pdf.exists():
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf))
            resume = "\n".join(
                (p.extract_text() or "") for p in reader.pages
            ).strip()
        except Exception:
            log.exception("PDF read failed for %s", pdf)
    return summary.strip(), resume.strip()


def load_twin_context() -> tuple[str, str]:
    """
    Return (summary_text, resume_text) for the system prompt.
    Sources are **merged** in order: Sheet → URL → env → (optional) local files;
    each step only fills fields that are still empty.
    Cached for CONTEXT_CACHE_SECONDS to avoid hitting Sheet/URL every message.
    """
    global _context_cache
    now = time.monotonic()
    prev_s, prev_r, t0 = _context_cache
    if (now - t0) < CONTEXT_CACHE_SECONDS and (prev_s or prev_r):
        log.debug("Context: using cache (age %.1fs)", now - t0)
        return prev_s, prev_r

    log.info("Context: loading (cache miss or expired)…")
    t_load = time.monotonic()
    summary, resume = "", ""

    sheet_pair = _load_context_from_sheet()
    if sheet_pair is not None:
        s, r = sheet_pair
        if s:
            summary = s
        if r:
            resume = r
        if s or r:
            log.info(
                "Context: Google Sheet twin_context (%d + %d chars)", len(summary), len(resume)
            )

    if (not summary or not resume) and CONTEXT_URL:
        try:
            raw = _fetch_url_text(CONTEXT_URL)
            us, ur = _parse_context_payload(raw)
            if not summary:
                summary = us
            if not resume:
                resume = ur
            if us or ur:
                log.info("Context: CONTEXT_URL filled gaps (%d + %d chars)", len(summary), len(resume))
        except (urllib.error.URLError, OSError) as e:
            log.warning("CONTEXT_URL fetch failed: %s", e)

    if not summary or not resume:
        es, er = _load_context_from_env()
        if not summary:
            summary = es
        if not resume:
            resume = er
        if es or er:
            log.info("Context: env TWIN_SUMMARY / TWIN_RESUME_TEXT (%d + %d chars)", len(summary), len(resume))

    if ALLOW_LOCAL_FILES and (not summary or not resume):
        ls, lr = _load_context_local_files()
        if not summary:
            summary = ls
        if not resume:
            resume = lr
        if ls or lr:
            log.warning("Context: local files (ALLOW_LOCAL_CONTEXT_FILES) (%d + %d chars)", len(summary), len(resume))

    if not summary and not resume:
        log.warning(
            "No twin context loaded. Set Sheet tab twin_context, CONTEXT_URL, "
            "or TWIN_SUMMARY / TWIN_RESUME_TEXT (see module docstring)."
        )

    _context_cache = (summary, resume, now)
    log.info(
        "Context: load finished in %.2fs (summary=%d chars, resume=%d chars)",
        time.monotonic() - t_load,
        len(summary),
        len(resume),
    )
    return summary, resume


# =============================================================================
# 2) Google Sheets — append unknown questions + user emails
# =============================================================================


def _sheets_append(worksheet_title: str, values: list[Any]) -> None:
    if not GOOGLE_SHEET_ID or not GSPREAD_JSON:
        return
    try:
        import gspread

        gc = _gspread_authorize()
        if gc is None:
            return
        sh = gc.open_by_key(GOOGLE_SHEET_ID)
        try:
            ws = sh.worksheet(worksheet_title)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_title, rows=2000, cols=8)
            if worksheet_title == "unknown_questions":
                ws.append_row(["timestamp_utc", "question"])
            elif worksheet_title == "user_details":
                ws.append_row(["timestamp_utc", "email", "name", "notes"])
            else:
                ws.append_row(["timestamp_utc", "data"])
        ws.append_row(["" if v is None else str(v) for v in values])
    except Exception:
        log.exception("Google Sheets append failed (%s)", worksheet_title)


def log_unknown_question(question: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    _sheets_append("unknown_questions", [ts, question])


def log_user_details(email: str, name: str = "", notes: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    _sheets_append("user_details", [ts, email, name or "", notes or ""])


# =============================================================================
# 3) Gemini via OpenAI-compatible API (no extra repo modules)
# =============================================================================


def _gemini_client() -> OpenAI:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for the chat model.")
    return OpenAI(
        base_url=GEMINI_BASE_URL,
        api_key=key,
        timeout=LLM_HTTP_TIMEOUT,
    )


def _normalize_tool_call(tc: Any) -> SimpleNamespace:
    if isinstance(tc, dict):
        tid = tc.get("id") or ""
        fn = tc.get("function")
        if isinstance(fn, dict):
            name, args = fn.get("name") or "", fn.get("arguments")
        elif fn is not None:
            name, args = getattr(fn, "name", "") or "", getattr(fn, "arguments", None)
        else:
            name, args = "", None
    else:
        tid = getattr(tc, "id", "") or ""
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn is not None else ""
        args = getattr(fn, "arguments", None) if fn is not None else None
    if args is None:
        args = "{}"
    elif not isinstance(args, str):
        args = str(args)
    return SimpleNamespace(id=tid, function=SimpleNamespace(name=name, arguments=args))


def _assistant_message_dict(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
        tcs = msg.get("tool_calls")
        if tcs:
            out["tool_calls"] = [dict(x) if isinstance(x, dict) else x for x in tcs]
        return out
    dump = getattr(msg, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    out2: dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", None)}
    tco = getattr(msg, "tool_calls", None)
    if tco:
        out2["tool_calls"] = []
        for tc in tco:
            fn = getattr(tc, "function", None)
            out2["tool_calls"].append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "") if fn else "",
                    },
                }
            )
    return out2


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[SimpleNamespace] | None
    assistant_message: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tool_round(self) -> bool:
        return bool(self.tool_calls)


def _parse_choice(choice: Any) -> tuple[str, list[SimpleNamespace] | None, dict[str, Any]]:
    msg = choice.message
    tool_calls_raw = getattr(msg, "tool_calls", None)
    if tool_calls_raw is None and isinstance(msg, dict):
        tool_calls_raw = msg.get("tool_calls")
    tool_calls = (
        [_normalize_tool_call(tc) for tc in tool_calls_raw] if tool_calls_raw else None
    )
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = getattr(msg, "content", None)
    text = "" if content is None else (content if isinstance(content, str) else str(content))
    assistant = _assistant_message_dict(msg)
    return text, tool_calls, assistant


def llm_chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
    client = _gemini_client()
    hb = _LongWaitHeartbeat(CHAT_HEARTBEAT_SEC, f"Gemini HTTP model={model!r}")
    hb.start()
    t0 = time.monotonic()
    try:
        log.info(
            "LLM: POST chat.completions (model=%s, messages=%d, tools=%d, timeout=%ss)",
            model,
            len(messages),
            len(tools),
            LLM_HTTP_TIMEOUT,
        )
        raw = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            timeout=LLM_HTTP_TIMEOUT,
        )
    except Exception:
        log.exception("LLM: request failed after %.2fs", time.monotonic() - t0)
        raise
    finally:
        hb.stop()

    elapsed = time.monotonic() - t0
    choice = raw.choices[0]
    fr = getattr(choice, "finish_reason", None)
    log.info("LLM: response in %.2fs finish_reason=%s", elapsed, fr)
    text, tool_calls, assistant = _parse_choice(choice)
    return LLMResponse(text=text or "", tool_calls=tool_calls, assistant_message=assistant)


# =============================================================================
# 4) Chat logic — system prompt + tool loop
# =============================================================================

record_user_details_json = {
    "name": "record_user_details",
    "description": "Record that the user wants to stay in touch and provided an email.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "User email"},
            "name": {"type": "string", "description": "User name if given"},
            "notes": {"type": "string", "description": "Extra context"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record a question you could not answer from the given context.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unanswered question"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}
TOOLS = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


def build_system_prompt() -> str:
    summary, resume = load_twin_context()
    return f"""You are acting as {NAME}. You are answering questions on {NAME}'s website, especially about career, projects, background, skills, and experience.
Represent {NAME} naturally — conversational, concise, human. Do not invent facts not supported by the context below.

If the answer is not clearly in the Summary or Resume/LinkedIn section, or you are not confident, call `record_unknown_question` with the user's question (verbatim or a clean paraphrase), then reply briefly that it was logged for future updates.

If the user wants to stay in touch, ask for their email and call `record_user_details` with their email (and name/notes if given).

## Summary:
{summary}

## Resume / profile text:
{resume}

Stay in character as {NAME}."""


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
    log_user_details(email=email, name=name or "", notes=notes or "")
    return {"recorded": "ok"}


def record_unknown_question(question: str):
    log_unknown_question(question)
    return {"recorded": "ok"}


def handle_tool_calls(tool_calls: list[SimpleNamespace]) -> list[dict[str, Any]]:
    results = []
    for tc in tool_calls:
        tool_name = tc.function.name
        arguments = json.loads(tc.function.arguments or "{}")
        log.info("Tool: executing %s", tool_name)
        fn = globals().get(tool_name)
        result = fn(**arguments) if callable(fn) else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tc.id})
    return results


def _scrub_assistant_message(msg: dict) -> dict:
    out = dict(msg)
    if out.get("role") == "assistant" and out.get("content") is None:
        out["content"] = ""
    return out


def chat(message: str, history: list, model: str = CHAT_MODEL):
    """Gradio ``type="messages"`` callback: ``history`` is list of dicts role/content."""
    t_chat = time.monotonic()
    log.info("Chat: new user message (%d chars), history_len=%d", len(message or ""), len(history))
    history_msgs = [{"role": h["role"], "content": h["content"]} for h in history]

    t0 = time.monotonic()
    system_content = build_system_prompt()
    log.info("Chat: system prompt built in %.2fs (%d chars)", time.monotonic() - t0, len(system_content))

    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": system_content}] + history_msgs + [{"role": "user", "content": message}]
    )
    round_n = 0
    while True:
        round_n += 1
        log.info("Chat: LLM round %d (messages=%d)", round_n, len(messages))
        response = llm_chat(model=model, messages=messages, tools=TOOLS)
        if response.is_tool_round:
            assert response.tool_calls is not None
            names = [tc.function.name for tc in response.tool_calls]
            log.info("Chat: tool round — %s", names)
            messages.append(_scrub_assistant_message(response.assistant_message))
            messages.extend(handle_tool_calls(response.tool_calls))
        else:
            text = (response.text or "").strip()
            log.info("Chat: done in %.2fs (final text %d chars)", time.monotonic() - t_chat, len(text))
            return text or "Thanks — noted. Anything else?"


# =============================================================================
# 5) Minimal Gradio UI
# =============================================================================


def launch_ui() -> None:
    log_startup_diagnostics()
    theme = gr.themes.Soft(
        font=gr.themes.GoogleFont("DM Sans"),
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_md,
    )
    title = f"Chat with {NAME}"
    desc = (
        "Career twin — answers from your private context (Sheet or env). "
        "Unanswered questions and contact emails may be saved to your Google Sheet if configured."
    )
    _ui_css = """
    .gradio-container { max-width: 900px !important; margin: 0 auto !important; }
    .prose { line-height: 1.55; }
    footer { opacity: 0.88; font-size: 0.82rem; }
    """
    demo = gr.ChatInterface(
        chat,
        type="messages",
        title=title,
        description=desc,
        theme=theme,
        css=_ui_css,
        chatbot=gr.Chatbot(height=520, show_copy_button=True, latex_delimiters=[]),
        examples=["What are you working on lately?", "What tech stack do you prefer?"],
        show_progress="full",
        fill_width=True,
    )
    log.info("Gradio UI starting on http://0.0.0.0:%s", PORT)
    demo.launch(server_name="0.0.0.0", server_port=PORT)


if __name__ == "__main__":
    launch_ui()
