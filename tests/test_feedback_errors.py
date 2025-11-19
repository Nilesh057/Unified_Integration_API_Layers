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


class TestFeedbackValidation:
    """Test feedback endpoint validation and error handling."""

    def test_valid_feedback(self):
        """Test successful feedback submission."""
        # First create a message to reference
        payload = {
            "payload": {
                "content": "test message for feedback",
                "rl_reward": 0.5,
                "user_feedback": 0.5,
                "action_success": 0.5,
                "cognitive_score": 0.5
            }
        }

        response = client.post("/api/decision_hub", json=payload)
        assert response.status_code == 200
        trace_id = response.json()["trace_id"]

        # Get message ID
        conn = sqlite3.connect(TEST_DB_PATH)
        message = pd.read_sql("SELECT id FROM messages WHERE trace_id = ?", conn, params=(trace_id,))
        message_id = int(message.iloc[0]["id"])
        conn.close()

        # Test valid feedback
        feedback_payload = {
            "feedback": 0.8,
            "target_id": message_id,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["data"]["feedback_recorded"] == 0.8
        assert data["data"]["target_type"] == "message"
        assert data["data"]["target_id"] == message_id

    def test_feedback_value_out_of_range(self):
        """Test feedback values outside allowed range."""
        # Test value too low
        feedback_payload = {
            "feedback": -2.0,  # Below -1.0
            "target_id": 1,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 422  # Pydantic validation error

        # Test value too high
        feedback_payload = {
            "feedback": 2.0,  # Above 1.0
            "target_id": 1,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_invalid_target_type(self):
        """Test invalid target_type values."""
        feedback_payload = {
            "feedback": 0.5,
            "target_id": 1,
            "target_type": "invalid_type"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_missing_target_id(self):
        """Test missing target_id field."""
        feedback_payload = {
            "feedback": 0.5,
            "target_type": "message"
            # missing target_id
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_invalid_target_id_type(self):
        """Test non-integer target_id."""
        feedback_payload = {
            "feedback": 0.5,
            "target_id": "not_an_integer",
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_zero_target_id(self):
        """Test zero target_id (should be > 0)."""
        feedback_payload = {
            "feedback": 0.5,
            "target_id": 0,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_negative_target_id(self):
        """Test negative target_id."""
        feedback_payload = {
            "feedback": 0.5,
            "target_id": -1,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_nonexistent_target_message(self):
        """Test feedback for non-existent message."""
        feedback_payload = {
            "feedback": 0.5,
            "target_id": 99999,  # Non-existent ID
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 404

        data = response.json()
        assert data["status"] == "error"
        assert "not found" in data["message"]

    def test_nonexistent_target_task(self):
        """Test feedback for non-existent task."""
        feedback_payload = {
            "feedback": 0.5,
            "target_id": 99999,  # Non-existent ID
            "target_type": "task"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 404

    def test_nonexistent_target_decision(self):
        """Test feedback for non-existent decision."""
        feedback_payload = {
            "feedback": 0.5,
            "target_id": 99999,  # Non-existent ID
            "target_type": "decision"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 404

    def test_boundary_feedback_values(self):
        """Test boundary feedback values."""
        # First create a message to reference
        payload = {
            "payload": {
                "content": "test message for boundary feedback",
                "rl_reward": 0.5,
                "user_feedback": 0.5,
                "action_success": 0.5,
                "cognitive_score": 0.5
            }
        }

        response = client.post("/api/decision_hub", json=payload)
        assert response.status_code == 200
        trace_id = response.json()["trace_id"]

        # Get message ID
        conn = sqlite3.connect(TEST_DB_PATH)
        message = pd.read_sql("SELECT id FROM messages WHERE trace_id = ?", conn, params=(trace_id,))
        message_id = int(message.iloc[0]["id"])
        conn.close()

        # Test minimum value
        feedback_payload = {
            "feedback": -1.0,
            "target_id": message_id,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 200

        # Test maximum value
        feedback_payload = {
            "feedback": 1.0,
            "target_id": message_id,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 200

    def test_feedback_persistence(self):
        """Test that feedback is properly persisted in database."""
        # First create a message to reference
        payload = {
            "payload": {
                "content": "test message for persistence check",
                "rl_reward": 0.5,
                "user_feedback": 0.5,
                "action_success": 0.5,
                "cognitive_score": 0.5
            }
        }

        response = client.post("/api/decision_hub", json=payload)
        assert response.status_code == 200
        trace_id = response.json()["trace_id"]

        # Get message ID
        conn = sqlite3.connect(TEST_DB_PATH)
        message = pd.read_sql("SELECT id FROM messages WHERE trace_id = ?", conn, params=(trace_id,))
        message_id = int(message.iloc[0]["id"])
        conn.close()

        # Submit feedback
        feedback_payload = {
            "feedback": 0.7,
            "target_id": message_id,
            "target_type": "message"
        }
        response = client.post("/api/feedback", json=feedback_payload)
        assert response.status_code == 200

        # Verify feedback was stored
        conn = sqlite3.connect(TEST_DB_PATH)
        feedback = pd.read_sql("SELECT * FROM feedback WHERE target_id = ? AND target_type = ?",
                              conn, params=(message_id, "message"))
        conn.close()

        assert not feedback.empty
        assert feedback.iloc[0]["feedback_value"] == 0.7
        assert feedback.iloc[0]["target_type"] == "message"
        assert feedback.iloc[0]["target_id"] == message_id