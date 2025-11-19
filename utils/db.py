import sqlite3
import os
from contextlib import contextmanager

def init_db(db_path: str):
    """Initialize the database if it doesn't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        conn.close()

@contextmanager
def get_db(db_path: str):
    """Get a database connection with row factory as context manager."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()