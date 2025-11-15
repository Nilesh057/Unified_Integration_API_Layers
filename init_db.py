#!/usr/bin/env python3
"""
Database initialization script for assistant_core.db
Creates the unified SQLite database with all required tables.
"""

import os
from utils.db import init_db
from utils.schema import ensure_schema

def main():
    # Define paths
    data_dir = "data"
    db_path = os.path.join(data_dir, "assistant_core.db")

    print(f"Initializing database at {db_path}...")

    # Initialize the database (creates file if not exists)
    init_db(db_path)

    # Ensure all tables exist
    ensure_schema(db_path)

    print("Database initialized successfully with all tables:")
    print("- messages")
    print("- summaries")
    print("- tasks")
    print("- feedback")
    print("- decisions")
    print("- embeddings")
    print("- rl_logs")

if __name__ == "__main__":
    main()