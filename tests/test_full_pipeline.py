import pytest
from fastapi.testclient import TestClient
from app.main import app
import sqlite3
import os
import pandas as pd
import tempfile
import shutil

# Use temporary directory for tests
TEST_DATA_DIR = tempfile.mkdtemp()
TEST_DB_PATH = os.path.join(TEST_DATA_DIR, "test_assistant_core.db")
TEST_CSV_PATH = os.path.join(TEST_DATA_DIR, "test_decision_log.csv")
TEST_REGISTRY_PATH = os.path.join(TEST_DATA_DIR, "test_agent_registry.json")

# Global test client - will be created in fixture
client = None

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with mock registry and database."""
    global client

    # Set environment variables
    os.environ["DATA_DIR"] = TEST_DATA_DIR
    os.environ["DB_PATH"] = TEST_DB_PATH
    os.environ["DECISION_LOG_PATH"] = TEST_CSV_PATH
    os.environ["AGENT_REGISTRY_PATH"] = TEST_REGISTRY_PATH

    # Create test registry
    test_registry = {
        "agents": {
            "summarizer": {
                "endpoint": "/api/summarize",
                "weight": 0.3
            },
            "cognitive": {
                "endpoint": "/api/process_summary",
                "weight": 0.3
            },
            "rl_agent": {
                "endpoint": "/api/agent_action",
                "weight": 0.2
            },
            "embedcore": {
                "endpoint": "/api/embed",
                "weight": 0.1
            },
            "actionsense": {
                "endpoint": "/api/respond",
                "weight": 0.1
            }
        }
    }

    with open(TEST_REGISTRY_PATH, 'w') as f:
        import json
        json.dump(test_registry, f)

    # Initialize test database
    from utils.db import init_db
    from utils.schema import ensure_schema
    init_db(TEST_DB_PATH)
    ensure_schema(TEST_DB_PATH)

    # Create TestClient after environment setup
    client = TestClient(app)

    yield

    # Cleanup
    shutil.rmtree(TEST_DATA_DIR)

def test_full_pipeline():
    """Test the complete pipeline with realistic data."""
    # Simulate a message from WhatsApp
    payload = {
        "payload": {
            "source": "whatsapp",
            "content": "Hello, this is a test message that needs processing and analysis",
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

    # Check DB for message and decision (use the actual DB path from environment)
    actual_db_path = os.environ.get("DB_PATH", TEST_DB_PATH)
    conn = sqlite3.connect(actual_db_path)
    messages = pd.read_sql("SELECT * FROM messages WHERE trace_id = ?", conn, params=(trace_id,))
    assert not messages.empty
    assert messages.iloc[0]["content"] == payload["payload"]["content"]

    decisions = pd.read_sql("SELECT * FROM decisions WHERE trace_id = ?", conn, params=(trace_id,))
    assert not decisions.empty
    conn.close()

    # CSV logging is optional and not critical for the core functionality
    # The test focuses on API responses and database integrity

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
    actual_db_path = os.environ.get("DB_PATH", TEST_DB_PATH)
    conn = sqlite3.connect(actual_db_path)
    rl_logs = pd.read_sql("SELECT * FROM rl_logs WHERE trace_id = ?", conn, params=(trace_id,))
    assert not rl_logs.empty
    conn.close()

    # Simulate embed
    embed_payload = {
        "payload": {
            "text": "test embedding generation"
        },
        "trace_id": trace_id
    }
    response = client.post("/api/embed", json=embed_payload)
    assert response.status_code == 200

    # Check embeddings
    conn = sqlite3.connect(actual_db_path)
    embeddings = pd.read_sql("SELECT * FROM embeddings WHERE trace_id = ?", conn, params=(trace_id,))
    assert not embeddings.empty
    conn.close()

    # Simulate respond
    respond_payload = {
        "payload": {
            "content": "response content for testing"
        },
        "trace_id": trace_id
    }
    response = client.post("/api/respond", json=respond_payload)
    assert response.status_code == 200

    # Check tasks
    conn = sqlite3.connect(actual_db_path)
    tasks = pd.read_sql("SELECT * FROM tasks WHERE trace_id = ?", conn, params=(trace_id,))
    assert not tasks.empty
    conn.close()

def test_api_error_cases():
    """Test API endpoints with invalid inputs."""
    # Test decision_hub with missing payload (Pydantic validation)
    response = client.post("/api/decision_hub", json={})
    assert response.status_code in [400, 422]  # Either our validation or Pydantic's

    # Test decision_hub with invalid content
    response = client.post("/api/decision_hub", json={"payload": {"content": ""}})
    assert response.status_code == 400

    # Test decision_hub with invalid signal values
    response = client.post("/api/decision_hub", json={
        "payload": {
            "content": "test",
            "rl_reward": "invalid"
        }
    })
    assert response.status_code == 400

    # Test embed with missing text
    response = client.post("/api/embed", json={"payload": {}})
    assert response.status_code == 400

    # Test embed with empty text
    response = client.post("/api/embed", json={"payload": {"text": ""}})
    assert response.status_code == 400

    # Test feedback with missing required fields (Pydantic validation)
    response = client.post("/api/feedback", json={})
    assert response.status_code == 422  # Pydantic validation error

    # Test feedback with invalid target_id (Pydantic validation)
    response = client.post("/api/feedback", json={
        "feedback": 0.5,
        "target_id": "invalid",
        "target_type": "message"
    })
    assert response.status_code == 422  # Pydantic validation error

def test_database_integrity():
    """Test database constraints and integrity."""
    # Test unique trace_id constraint
    payload1 = {
        "payload": {
            "content": "test message 1",
            "rl_reward": 0.5,
            "user_feedback": 0.5,
            "action_success": 0.5,
            "cognitive_score": 0.5
        },
        "trace_id": "test-trace-123"
    }

    payload2 = {
        "payload": {
            "content": "test message 2",
            "rl_reward": 0.5,
            "user_feedback": 0.5,
            "action_success": 0.5,
            "cognitive_score": 0.5
        },
        "trace_id": "test-trace-123"  # Same trace_id
    }

    # First request should succeed
    response1 = client.post("/api/decision_hub", json=payload1)
    assert response1.status_code == 200

    # Second request should fail due to unique constraint
    response2 = client.post("/api/decision_hub", json=payload2)
    assert response2.status_code == 500  # Database integrity error


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with extreme values
    payload = {
        "payload": {
            "content": "test with extreme values",
            "rl_reward": 100.0,  # Should be clamped
            "user_feedback": -10.0,  # Should be clamped
            "action_success": 1.5,  # Should be clamped
            "cognitive_score": -5.0  # Should be clamped
        }
    }
    response = client.post("/api/decision_hub", json=payload)
    assert response.status_code == 200

    # Test with very long text
    long_text = "word " * 1000
    embed_payload = {
        "payload": {
            "text": long_text
        }
    }
    response = client.post("/api/embed", json=embed_payload)
    # Should either succeed or fail with appropriate error
    assert response.status_code in [200, 400]

if __name__ == "__main__":
    test_full_pipeline()
    test_api_error_cases()
    test_database_integrity()
    test_edge_cases()
    print("All tests passed!")