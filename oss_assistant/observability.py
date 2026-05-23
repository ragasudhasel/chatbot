"""
Observability layer (Bonus)
Lightweight structured logging for every LLM call.
Logs to stdout (JSON Lines) and optionally to a local SQLite DB.

Drop-in wrapper — replace direct API calls with traced_call().

Usage:
    from observability import traced_call, get_stats

    reply, meta = traced_call(
        model="Qwen2.5-72B",
        messages=[{"role": "user", "content": "Hello"}],
        call_fn=lambda msgs: client.chat_completion(model=..., messages=msgs, max_tokens=512),
        extract_reply=lambda r: r.choices[0].message.content,
    )
"""

import json, time, uuid, sqlite3, os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

DB_PATH = os.environ.get("OBS_DB", "observability.db")

# ── Schema ────────────────────────────────────────────────────────────────────
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    id          TEXT PRIMARY KEY,
    ts          TEXT,
    model       TEXT,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    latency_ms  REAL,
    safety_blocked INTEGER DEFAULT 0,
    error       TEXT,
    user_msg    TEXT,
    assistant_msg TEXT
);
"""

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_INIT_SQL)
    conn.commit()
    return conn


@dataclass
class CallMeta:
    call_id: str
    model: str
    latency_ms: float
    prompt_chars: int
    reply_chars: int
    safety_blocked: bool
    error: str | None


def traced_call(
    model: str,
    messages: list[dict],
    call_fn,
    extract_reply,
    safety_blocked: bool = False,
) -> tuple[str, CallMeta]:
    """
    Wraps any LLM call with timing + logging.

    Args:
        model: Human-readable model name for logging.
        messages: The messages list passed to the API.
        call_fn: Callable that takes messages → raw API response.
        extract_reply: Callable that takes raw API response → str reply.
        safety_blocked: Set True if a guardrail already blocked the call.

    Returns:
        (reply_text, CallMeta)
    """
    call_id = str(uuid.uuid4())[:8]
    user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    if safety_blocked:
        meta = CallMeta(call_id, model, 0.0, len(user_msg), 0, True, None)
        _log(meta, user_msg, "[blocked by guardrail]")
        return "[blocked by guardrail]", meta

    t0 = time.perf_counter()
    error = None
    reply = ""
    try:
        raw   = call_fn(messages)
        reply = extract_reply(raw)
    except Exception as e:
        error = str(e)
        reply = f"⚠️ Error: {e}"

    latency = (time.perf_counter() - t0) * 1000
    meta = CallMeta(call_id, model, round(latency, 1), len(user_msg), len(reply), False, error)
    _log(meta, user_msg, reply)
    return reply, meta


def _log(meta: CallMeta, user_msg: str, reply: str):
    row = {
        "id":                meta.call_id,
        "ts":                datetime.now(timezone.utc).isoformat(),
        "model":             meta.model,
        "prompt_tokens":     meta.prompt_chars // 4,   # rough approximation
        "completion_tokens": meta.reply_chars  // 4,
        "latency_ms":        meta.latency_ms,
        "safety_blocked":    int(meta.safety_blocked),
        "error":             meta.error,
        "user_msg":          user_msg[:500],
        "assistant_msg":     reply[:500],
    }
    # JSON Lines to stdout
    print(json.dumps(row))
    # SQLite
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)",
            list(row.values()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_stats() -> dict:
    """Return aggregate stats from the SQLite log."""
    conn = _get_db()
    cur  = conn.cursor()
    stats = {}
    for model, in cur.execute("SELECT DISTINCT model FROM calls").fetchall():
        cur.execute(
            "SELECT COUNT(*), AVG(latency_ms), SUM(safety_blocked), SUM(error IS NOT NULL) FROM calls WHERE model=?",
            (model,),
        )
        cnt, avg_lat, blocked, errors = cur.fetchone()
        stats[model] = {
            "total_calls":      cnt,
            "avg_latency_ms":   round(avg_lat or 0, 1),
            "safety_blocks":    blocked or 0,
            "errors":           errors or 0,
        }
    conn.close()
    return stats


if __name__ == "__main__":
    print("Current stats:", json.dumps(get_stats(), indent=2))
