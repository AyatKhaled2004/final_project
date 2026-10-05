"""SQLite memory: chat log per session + a few persistent preferences."""
import sqlite3
import time

import config


def _conn() -> sqlite3.Connection:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts REAL)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS prefs (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def add_message(session_id: str, role: str, content: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def recent_messages(session_id: str, limit: int = 12) -> list[tuple[str, str]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows))


def get_pref(key: str, default: str = "") -> str:
    with _conn() as c:
        row = c.execute("SELECT value FROM prefs WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_pref(key: str, value: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO prefs (key, value) VALUES (?, ?)", (key, value))
