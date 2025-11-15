#!/usr/bin/env python3
"""
Deployment Test Script
Tests the Unified Cognitive Intelligence API endpoints with sample data.
Run this after deploying to verify functionality.
"""

import requests
import json
import uuid

# Change this to your deployed API URL
API_BASE = "http://localhost:8000"  # For local testing, change to deployed URL

def test_endpoint(endpoint, payload, description):
    url = f"{API_BASE}{endpoint}"
    print(f"\nTesting {description}: {url}")
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return data
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    trace_id = str(uuid.uuid4())
    print(f"Using trace_id: {trace_id}")

    # Test Health
    test_endpoint("/api/health", {}, "Health Check")

    # Test Decision Hub
    decision_payload = {
        "payload": {
            "source": "whatsapp",
            "content": "Test message for deployment",
            "rl_reward": 0.8,
            "user_feedback": 1.0,
            "action_success": 0.9,
            "cognitive_score": 0.7
        },
        "trace_id": trace_id
    }
    decision_response = test_endpoint("/api/decision_hub", decision_payload, "Decision Hub")

    # Test Agent Action
    action_payload = {
        "payload": {
            "action": "test_action",
            "reward": 0.8,
            "confidence": 0.9
        },
        "trace_id": trace_id
    }
    test_endpoint("/api/agent_action", action_payload, "Agent Action")

    # Test Embed
    embed_payload = {
        "payload": {
            "text": "This is a test for embedding generation."
        },
        "trace_id": trace_id
    }
    test_endpoint("/api/embed", embed_payload, "Embed")

    # Test Respond
    respond_payload = {
        "payload": {
            "content": "Test response content"
        },
        "trace_id": trace_id
    }
    test_endpoint("/api/respond", respond_payload, "Respond")

    print("\nDeployment test completed. Check logs and database for persistence.")

if __name__ == "__main__":
    main()