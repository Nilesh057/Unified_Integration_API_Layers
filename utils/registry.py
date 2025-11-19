import json
import os
from typing import Dict, Any

def validate_agent_registry(registry: Dict[str, Any]) -> bool:
    """Validate the structure of agent registry."""
    if not isinstance(registry, dict) or "agents" not in registry:
        return False

    agents = registry.get("agents", {})
    if not isinstance(agents, dict):
        return False

    required_keys = ["endpoint", "weight"]
    for agent_name, agent_config in agents.items():
        if not isinstance(agent_config, dict):
            return False
        if not all(key in agent_config for key in required_keys):
            return False
        # Validate weight is a number between 0 and 1
        weight = agent_config.get("weight")
        if not isinstance(weight, (int, float)) or not (0.0 <= weight <= 1.0):
            return False

    return True

def load_agent_registry(path: str) -> Dict[str, Any]:
    """Load and validate agent registry from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent registry file not found: {path}")

    try:
        with open(path, 'r') as f:
            registry = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in agent registry file: {e}")

    if not validate_agent_registry(registry):
        raise ValueError("Invalid agent registry structure")

    return registry