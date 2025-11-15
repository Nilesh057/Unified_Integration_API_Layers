import json
import os

def load_agent_registry(path: str) -> dict:
    """Load agent registry from JSON file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}