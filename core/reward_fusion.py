from __future__ import annotations
import json
import os
import csv
from datetime import datetime
from typing import Dict, Any

DEFAULT_AGENT_CONF = {
    "summarizer": 0.3,
    "cognitive": 0.3,
    "rl_agent": 0.2,
    "embedcore": 0.1,
    "actionsense": 0.1,
}


def safe_normalize(x: float, min_v: float = -1.0, max_v: float = 1.0) -> float:
    """Safely normalize value to 0..1 range with robust edge case handling."""
    # Clamp input to range
    x = max(min(x, max_v), min_v)

    # Handle zero range case - return equal weights (0.5)
    if max_v == min_v:
        return 0.5

    # Standard min-max normalization
    return (x - min_v) / (max_v - min_v)

# Keep backward compatibility
def normalize(x: float, min_v: float = -1.0, max_v: float = 1.0) -> float:
    """Legacy normalize function - use safe_normalize instead."""
    return safe_normalize(x, min_v, max_v)


def fuse_rewards(
    rl_reward: float,
    user_feedback: float,
    action_success: float,
    cognitive_score: float,
    registry: Dict[str, Any],
    dynamic_confidences: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    agents = registry.get("agents", {}) if registry else {}

    # Normalize all signals to 0..1
    signals = {
        "rl_agent": normalize(rl_reward),
        "actionsense": normalize(action_success),
        "cognitive": normalize(cognitive_score),
        # map user feedback -1..1 to 0..1
        "summarizer": normalize(user_feedback),
        "embedcore": 0.5,  # embeddings neutral contribution
    }

    decision_trace = []

    # Determine final weight per agent = registry weight * confidence
    final_weights = {}
    for name, cfg in agents.items():
        base_w = float(cfg.get("weight", DEFAULT_AGENT_CONF.get(name, 0.1)))
        conf = 0.5
        if dynamic_confidences and name in dynamic_confidences:
            conf = max(0.0, float(dynamic_confidences[name]))  # Ensure non-negative confidence
        final_w = base_w * conf
        final_weights[name] = final_w

    # Normalize weights to ensure they sum to 1.0 after clamping
    total_weight = sum(final_weights.values())
    if total_weight > 0:
        # Explicit normalization step to ensure weights sum to 1.0
        final_weights = {name: w / total_weight for name, w in final_weights.items()}

        # Final validation: ensure weights sum to exactly 1.0 (handle floating point precision)
        weight_sum = sum(final_weights.values())
        if abs(weight_sum - 1.0) > 1e-10:  # Allow for small floating point errors
            # Re-normalize to guarantee sum = 1.0
            final_weights = {name: w / weight_sum for name, w in final_weights.items()}
    else:
        # Fallback: equal weights if all weights are zero
        num_agents = len(final_weights)
        if num_agents > 0:
            equal_weight = 1.0 / num_agents
            final_weights = {name: equal_weight for name in final_weights.keys()}

    # Compute weighted sum using the already normalized weights
    weighted_score = sum((final_weights.get(a, 0.0) * signals.get(a, 0.0)) for a in final_weights)

    # Build decision trace with normalized weights
    for name, cfg in agents.items():
        base_w = float(cfg.get("weight", DEFAULT_AGENT_CONF.get(name, 0.1)))
        conf = 0.5
        if dynamic_confidences and name in dynamic_confidences:
            conf = max(0.0, float(dynamic_confidences[name]))
        normalized_w = final_weights.get(name, 0.0)
        decision_trace.append({
            "agent": name,
            "base_weight": base_w,
            "confidence": conf,
            "final_weight": normalized_w,  # Already normalized above
            "signal": signals.get(name, 0.0),
            "contribution": normalized_w * signals.get(name, 0.0),
        })

    # Confidence as weight concentration (1 - normalized entropy)
    import math
    eps = 1e-12

    if len(final_weights) > 0:
        probs = [w for w in final_weights.values() if w > 0]
        if probs:
            # Handle single agent case (perfect confidence)
            if len(probs) == 1:
                final_confidence = 1.0
            else:
                # Use consistent epsilon to prevent log(0)
                entropy = -sum(p * math.log(max(p, eps)) for p in probs)
                max_entropy = math.log(len(probs))
                final_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            final_confidence = 0.0
    else:
        final_confidence = 0.0

    # Determine top agent by contribution
    top_agent = max(decision_trace, key=lambda x: x["contribution"]) if decision_trace else {"agent": "none"}

    decision = "proceed" if weighted_score >= 0.5 else "defer"

    # Log to CSV is now handled by the caller (main.py) to use configurable path

    return {
        "final_score": round(weighted_score, 4),
        "final_confidence": round(final_confidence, 4),
        "top_agent": top_agent.get("agent"),
        "decision": decision,
        "decision_trace": decision_trace,
    }
