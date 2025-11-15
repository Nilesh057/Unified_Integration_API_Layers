import os
import sqlite3
import uuid
import json
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.db import get_db, init_db
from utils.schema import ensure_schema
from utils.logging_utils import log_decision_csv
from utils.registry import load_agent_registry
from core.reward_fusion import fuse_rewards
from utils.nlp import summarize_text, compute_cognitive_score

app = FastAPI(title="Unified Cognitive Intelligence API", version="0.1.0")

REGISTRY_PATH = os.getenv("AGENT_REGISTRY_PATH", "config/agent_registry.json")
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "assistant_core.db"))
DECISION_LOG_PATH = os.path.join(DATA_DIR, "decision_log.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("config", exist_ok=True)

init_db(DB_PATH)
ensure_schema(DB_PATH)

class Envelope(BaseModel):
    payload: Dict[str, Any]
    trace_id: str | None = None


def response_ok(data: Dict[str, Any], trace_id: str | None = None) -> JSONResponse:
    return JSONResponse(
        content={
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id or str(uuid.uuid4()),
            "data": data,
        }
    )


@app.post("/api/decision_hub")
async def decision_hub(req: Envelope):
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    # Persist message
    with get_db(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO messages(trace_id, source, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (trace_id, payload.get("source", "unknown"), payload.get("content", ""), ts),
        )
        conn.commit()

    # Load registry and compute dynamic routing weights
    registry = load_agent_registry(REGISTRY_PATH)
    confidences = payload.get("confidences", {})  # optional dynamic confidences per agent

    # Build input signals for fusion (default zeros if missing)
    rl_reward = payload.get("rl_reward", 0.0)
    user_feedback = payload.get("user_feedback", 0.0)  # -1 to +1
    action_success = payload.get("action_success", 0.0)  # 0/1 or probability
    cognitive_score = payload.get("cognitive_score", 0.0)

    fusion_result = fuse_rewards(
        rl_reward=rl_reward,
        user_feedback=user_feedback,
        action_success=action_success,
        cognitive_score=cognitive_score,
        registry=registry,
        dynamic_confidences=confidences,
    )

    # Persist decision
    with get_db(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO decisions(trace_id, decision, score, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trace_id, fusion_result["decision"], fusion_result["final_score"], fusion_result["final_confidence"], ts),
        )
        conn.commit()

    # Log decision trace to CSV for dashboard
    log_decision_csv(
        DECISION_LOG_PATH,
        timestamp=ts,
        agent_name=fusion_result["top_agent"],
        input_signal=json.dumps({
            "rl_reward": rl_reward,
            "user_feedback": user_feedback,
            "action_success": action_success,
            "cognitive_score": cognitive_score,
        }),
        reward=fusion_result["final_score"],
        confidence=fusion_result["final_confidence"],
        final_score=fusion_result["final_score"],
        decision_trace=json.dumps(fusion_result["decision_trace"]),
    )

    return response_ok({
        "decision": fusion_result["decision"],
        "top_agent": fusion_result["top_agent"],
        "final_score": fusion_result["final_score"],
        "final_confidence": fusion_result["final_confidence"],
        "decision_trace": fusion_result["decision_trace"],
    }, trace_id=trace_id)


@app.post("/api/agent_action")
async def agent_action(req: Envelope):
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    # For now we simulate RL agent action output
    action = payload.get("action", "noop")
    reward = float(payload.get("reward", 0.0))
    confidence = float(payload.get("confidence", 0.5))

    with get_db(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO rl_logs(trace_id, action, reward, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trace_id, action, reward, confidence, ts),
        )
        conn.commit()

    return response_ok({
        "action": action,
        "reward": reward,
        "confidence": confidence,
    }, trace_id=trace_id)


@app.post("/api/embed")
async def embed(req: Envelope):
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    text = payload.get("text", "")
    # Simple deterministic fake embedding for now
    vec = [float((sum(bytearray(text.encode())) % 100) / 100.0)] * 8

    with get_db(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO embeddings(trace_id, text, vector_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (trace_id, text, json.dumps(vec), ts),
        )
        conn.commit()

    return response_ok({
        "embedding": vec,
        "dim": len(vec),
    }, trace_id=trace_id)


@app.post("/api/summarize")
async def summarize(req: Envelope):
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    text = payload.get("text", "")
    # Improved extractive summary
    summary = summarize_text(text)

    with get_db(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO summaries(trace_id, summary_text, created_at)
            VALUES (?, ?, ?)
            """,
            (trace_id, summary, ts),
        )
        conn.commit()

    return response_ok({
        "summary": summary,
    }, trace_id=trace_id)


@app.post("/api/process_summary")
async def process_summary(req: Envelope):
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    summary = payload.get("summary", "")
    # Heuristic cognitive scoring
    cognitive_score = compute_cognitive_score(summary)

    return response_ok({
        "cognitive_score": cognitive_score,
        "processed": f"Processed: {summary}",
    }, trace_id=trace_id)


@app.post("/api/feedback")
async def feedback(req: Envelope):
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    # Normalize feedback to [-1, 1]
    raw_fb = payload.get("feedback", 0.0)
    try:
        feedback_value = float(raw_fb)
    except (TypeError, ValueError):
        feedback_value = 0.0
    feedback_value = max(-1.0, min(1.0, feedback_value))
    target_id = payload.get("target_id", "")
    target_type = payload.get("target_type", "message")

    with get_db(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback(trace_id, target_id, target_type, feedback_value, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trace_id, target_id, target_type, feedback_value, ts),
        )
        conn.commit()

    return response_ok({
        "feedback_recorded": feedback_value,
    }, trace_id=trace_id)


@app.post("/api/respond")
async def respond(req: Envelope):
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    content = payload.get("content", "")

    with get_db(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks(trace_id, task_type, content, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trace_id, "respond", content, "created", ts),
        )
        conn.commit()

    return response_ok({
        "response": f"ACK: {content[:64]}",
    }, trace_id=trace_id)


@app.get("/api/health")
async def health():
    return response_ok({"service": "assistant-core", "db_path": DB_PATH})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
