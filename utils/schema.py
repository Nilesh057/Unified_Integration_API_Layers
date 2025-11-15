import sqlite3

def ensure_schema(db_path: str):
    """Ensure all tables exist in the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Messages table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Summaries table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            message_id INTEGER,
            summary_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (message_id) REFERENCES messages (id)
        )
    """)

    # Tasks table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT DEFAULT 'created',
            created_at TEXT NOT NULL
        )
    """)

    # Feedback table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,  -- 'message', 'task', etc.
            feedback_value REAL NOT NULL,  -- e.g., 1 for thumbs up, -1 for down
            created_at TEXT NOT NULL
        )
    """)

    # Decisions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            top_agent TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # Embeddings table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            text TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # RL Logs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rl_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()