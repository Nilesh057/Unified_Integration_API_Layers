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


def normalize(x: float, min_v: float = -1.0, max_v: float = 1.0) -> float:
    if max_v == min_v:
        return 0.0
    x = max(min(x, max_v), min_v)
    return (x - min_v) / (max_v - min_v)  # 0..1


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
            conf = float(dynamic_confidences[name])
        final_w = max(base_w * conf, 0.0)
        final_weights[name] = final_w
        decision_trace.append({
            "agent": name,
            "base_weight": base_w,
            "confidence": conf,
            "final_weight": final_w,
            "signal": signals.get(name, 0.0),
            "contribution": final_w * signals.get(name, 0.0),
        })

    # Compute weighted sum
    denom = sum(final_weights.values()) or 1e-6
    weighted_score = sum((final_weights.get(a, 0.0) * signals.get(a, 0.0)) for a in final_weights) / denom

    # Confidence as weight concentration (1 - entropy-like)
    import math
    probs = [w / denom for w in final_weights.values()] if denom > 0 else [1.0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 0 else 1.0
    final_confidence = 1.0 - (entropy / (max_entropy + 1e-12)) if max_entropy > 0 else 0.0

    # Determine top agent by contribution
    top_agent = max(decision_trace, key=lambda x: x["contribution"]) if decision_trace else {"agent": "none"}

    decision = "proceed" if weighted_score >= 0.5 else "defer"

    # Log to CSV
    timestamp = datetime.now().isoformat()
    final_score = round(weighted_score, 4)
    file_path = 'data/decision_log.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'agent_name', 'input_signal', 'reward', 'confidence', 'final_score'])
        for entry in decision_trace:
            writer.writerow([
                timestamp,
                entry['agent'],
                entry['signal'],
                entry['contribution'],
                entry['confidence'],
                final_score
            ])

    return {
        "final_score": final_score,
        "final_confidence": round(final_confidence, 4),
        "top_agent": top_agent.get("agent"),
        "decision": decision,
        "decision_trace": decision_trace,
    }
