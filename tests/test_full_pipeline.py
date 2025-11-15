import pytest
from fastapi.testclient import TestClient
from app.main import app
import sqlite3
import os
import pandas as pd

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "assistant_core.db")
CSV_PATH = os.path.join(DATA_DIR, "decision_log.csv")

client = TestClient(app)

def test_full_pipeline():
    # Simulate a message from WhatsApp
    payload = {
        "payload": {
            "source": "whatsapp",
            "content": "Hello, process this message",
            "rl_reward": 0.8,
            "user_feedback": 1.0,
            "action_success": 0.9,
            "cognitive_score": 0.7,
            "confidences": {
                "summarizer": 0.9,
                "cognitive": 0.8,
                "rl_agent": 0.7,
                "embedcore": 0.6,
                "actionsense": 0.5
            }
        }
    }

    # Call decision_hub
    response = client.post("/api/decision_hub", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "decision" in data["data"]
    trace_id = data["trace_id"]

    # Check DB for message and decision
    conn = sqlite3.connect(DB_PATH)
    messages = pd.read_sql("SELECT * FROM messages WHERE trace_id = ?", conn, params=(trace_id,))
    assert not messages.empty

    decisions = pd.read_sql("SELECT * FROM decisions WHERE trace_id = ?", conn, params=(trace_id,))
    assert not decisions.empty
    conn.close()

    # Check CSV for logs
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        recent_logs = df[df['timestamp'].str.contains(trace_id[:10])]  # rough check
        assert not recent_logs.empty

    # Simulate agent_action
    action_payload = {
        "payload": {
            "action": "respond",
            "reward": 0.8,
            "confidence": 0.9
        },
        "trace_id": trace_id
    }
    response = client.post("/api/agent_action", json=action_payload)
    assert response.status_code == 200

    # Check RL logs
    conn = sqlite3.connect(DB_PATH)
    rl_logs = pd.read_sql("SELECT * FROM rl_logs WHERE trace_id = ?", conn, params=(trace_id,))
    assert not rl_logs.empty
    conn.close()

    # Simulate embed
    embed_payload = {
        "payload": {
            "text": "test embedding"
        },
        "trace_id": trace_id
    }
    response = client.post("/api/embed", json=embed_payload)
    assert response.status_code == 200

    # Check embeddings
    conn = sqlite3.connect(DB_PATH)
    embeddings = pd.read_sql("SELECT * FROM embeddings WHERE trace_id = ?", conn, params=(trace_id,))
    assert not embeddings.empty
    conn.close()

    # Simulate respond
    respond_payload = {
        "payload": {
            "content": "response content"
        },
        "trace_id": trace_id
    }
    response = client.post("/api/respond", json=respond_payload)
    assert response.status_code == 200

    # Check tasks
    conn = sqlite3.connect(DB_PATH)
    tasks = pd.read_sql("SELECT * FROM tasks WHERE trace_id = ?", conn, params=(trace_id,))
    assert not tasks.empty
    conn.close()

if __name__ == "__main__":
    test_full_pipeline()
    print("All tests passed!")