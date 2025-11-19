import os
import sqlite3
import uuid
import json
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.schemas import (
    Envelope, FeedbackPayload, DecisionHubPayload, EmbedPayload,
    SummarizePayload, ProcessSummaryPayload, RespondPayload, AgentActionPayload,
    APIResponse, ErrorResponse, DecisionResponse, FeedbackResponse,
    EmbedResponse, SummarizeResponse, ProcessSummaryResponse,
    RespondResponse, AgentActionResponse, HealthResponse
)

from utils.db import get_db, init_db
from utils.schema import ensure_schema
from utils.logging_utils import log_decision_csv
from utils.registry import load_agent_registry
from core.reward_fusion import fuse_rewards
from utils.nlp import summarize_text, compute_cognitive_score

app = FastAPI(title="Unified Cognitive Intelligence API", version="0.1.0")

# Configuration - read at runtime to allow test overrides
def get_config():
    return {
        "REGISTRY_PATH": os.getenv("AGENT_REGISTRY_PATH", "config/agent_registry.json"),
        "DATA_DIR": os.getenv("DATA_DIR", "data"),
        "DB_PATH": os.getenv("DB_PATH", os.path.join(os.getenv("DATA_DIR", "data"), "assistant_core.db")),
        "DECISION_LOG_PATH": os.getenv("DECISION_LOG_PATH", os.path.join(os.getenv("DATA_DIR", "data"), "decision_log.csv")),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "MAX_TEXT_LENGTH": int(os.getenv("MAX_TEXT_LENGTH", "10000")),
        "EMBEDDING_DIM": int(os.getenv("EMBEDDING_DIM", "8")),
    }

# Config will be read at runtime in each endpoint

def validate_target_exists(target_type: str, target_id: int, db_path: str) -> tuple[bool, str]:
    """
    Validate that a target entity exists in the database.

    Args:
        target_type: Type of target ('message', 'task', or 'decision')
        target_id: ID of the target entity
        db_path: Path to the database file

    Returns:
        Tuple of (exists: bool, error_message: str)
    """
    table_mapping = {
        "message": "messages",
        "task": "tasks",
        "decision": "decisions"
    }

    table_name = table_mapping.get(target_type)
    if not table_name:
        return False, f"Invalid target_type: {target_type}"

    try:
        with get_db(db_path) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT id FROM {table_name} WHERE id = ?", (target_id,))

            if not cur.fetchone():
                return False, f"Target {target_type} with id {target_id} not found"

            return True, ""
    except Exception as e:
        return False, f"Database validation error: {str(e)}"

# Initialize directories and database at module level with default config
default_config = get_config()
os.makedirs(default_config["DATA_DIR"], exist_ok=True)
os.makedirs("config", exist_ok=True)

try:
    init_db(default_config["DB_PATH"])
    ensure_schema(default_config["DB_PATH"])
except Exception as e:
    print(f"Failed to initialize database: {e}")
    raise



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
    try:
        config = get_config()
        trace_id = req.trace_id or str(uuid.uuid4())
        payload = req.payload
        ts = datetime.utcnow().isoformat()

        # Validate required payload fields
        if not payload:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Payload is required", "trace_id": trace_id}
            )

        content = payload.get("content", "")
        if not content or not isinstance(content, str):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Valid content string is required", "trace_id": trace_id}
            )

        # Validate signal values
        rl_reward = payload.get("rl_reward", 0.0)
        user_feedback = payload.get("user_feedback", 0.0)
        action_success = payload.get("action_success", 0.0)
        cognitive_score = payload.get("cognitive_score", 0.0)

        for name, value in [("rl_reward", rl_reward), ("user_feedback", user_feedback),
                           ("action_success", action_success), ("cognitive_score", cognitive_score)]:
            if not isinstance(value, (int, float)):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"{name} must be a number", "trace_id": trace_id}
                )

        # Validate confidences if provided
        confidences = payload.get("confidences", {})
        if confidences and not isinstance(confidences, dict):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "confidences must be a dictionary", "trace_id": trace_id}
            )

        # Persist message
        try:
            with get_db(config["DB_PATH"]) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO messages(trace_id, source, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (trace_id, payload.get("source", "unknown"), content, ts),
                )
                conn.commit()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Database error: {str(e)}", "trace_id": trace_id}
            )

        # Load registry and compute dynamic routing weights
        try:
            registry = load_agent_registry(config["REGISTRY_PATH"])
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Registry loading error: {str(e)}", "trace_id": trace_id}
            )

        try:
            fusion_result = fuse_rewards(
                rl_reward=rl_reward,
                user_feedback=user_feedback,
                action_success=action_success,
                cognitive_score=cognitive_score,
                registry=registry,
                dynamic_confidences=confidences,
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Fusion error: {str(e)}", "trace_id": trace_id}
            )

        # Persist decision
        try:
            with get_db(config["DB_PATH"]) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO decisions(trace_id, decision, score, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (trace_id, fusion_result["decision"], fusion_result["final_score"], fusion_result["final_confidence"], ts),
                )
                conn.commit()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Database error: {str(e)}", "trace_id": trace_id}
            )

        # Log decision trace to CSV for dashboard
        try:
            log_decision_csv(
                config["DECISION_LOG_PATH"],
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
        except Exception as e:
            # Log error but don't fail the request
            print(f"Warning: Failed to log decision to CSV: {e}")

        return response_ok({
            "decision": fusion_result["decision"],
            "top_agent": fusion_result["top_agent"],
            "final_score": fusion_result["final_score"],
            "final_confidence": fusion_result["final_confidence"],
            "decision_trace": fusion_result["decision_trace"],
        }, trace_id=trace_id)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unexpected error: {str(e)}", "trace_id": req.trace_id}
        )


@app.post("/api/agent_action")
async def agent_action(req: Envelope):
    config = get_config()
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    # For now we simulate RL agent action output
    action = payload.get("action", "noop")
    reward = float(payload.get("reward", 0.0))
    confidence = float(payload.get("confidence", 0.5))

    with get_db(config["DB_PATH"]) as conn:
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
    try:
        config = get_config()
        trace_id = req.trace_id or str(uuid.uuid4())
        payload = req.payload
        ts = datetime.utcnow().isoformat()

        # Validate payload
        if not payload:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Payload is required", "trace_id": trace_id}
            )

        text = payload.get("text", "")
        if not isinstance(text, str):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "text must be a string", "trace_id": trace_id}
            )

        if len(text) > config["MAX_TEXT_LENGTH"]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Text too long (max {config['MAX_TEXT_LENGTH']} characters)", "trace_id": trace_id}
            )

        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Text cannot be empty", "trace_id": trace_id}
            )

        # Generate embedding (simple deterministic fake embedding)
        try:
            vec = [float((sum(bytearray(text.encode())) % 100) / 100.0)] * config["EMBEDDING_DIM"]
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Embedding generation error: {str(e)}", "trace_id": trace_id}
            )

        # Persist to database
        try:
            with get_db(config["DB_PATH"]) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO embeddings(trace_id, text, vector_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (trace_id, text, json.dumps(vec), ts),
                )
                conn.commit()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Database error: {str(e)}", "trace_id": trace_id}
            )

        return response_ok({
            "embedding": vec,
            "dim": len(vec),
        }, trace_id=trace_id)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unexpected error: {str(e)}", "trace_id": req.trace_id}
        )


@app.post("/api/summarize")
async def summarize(req: Envelope):
    try:
        config = get_config()
        trace_id = req.trace_id or str(uuid.uuid4())
        payload = req.payload
        ts = datetime.utcnow().isoformat()

        # Validate payload
        if not payload:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Payload is required", "trace_id": trace_id}
            )

        text = payload.get("text", "")
        if not isinstance(text, str):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "text must be a string", "trace_id": trace_id}
            )

        if len(text) > config["MAX_TEXT_LENGTH"]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Text too long (max {config['MAX_TEXT_LENGTH']} characters)", "trace_id": trace_id}
            )

        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Text cannot be empty", "trace_id": trace_id}
            )

        # Generate summary
        try:
            summary = summarize_text(text)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Summarization error: {str(e)}", "trace_id": trace_id}
            )

        # Persist to database
        try:
            with get_db(config["DB_PATH"]) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO summaries(trace_id, summary_text, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (trace_id, summary, ts),
                )
                conn.commit()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Database error: {str(e)}", "trace_id": trace_id}
            )

        return response_ok({
            "summary": summary,
        }, trace_id=trace_id)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unexpected error: {str(e)}", "trace_id": req.trace_id}
        )


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
async def feedback(feedback_data: FeedbackPayload, trace_id: str | None = None):
    try:
        config = get_config()
        trace_id = trace_id or str(uuid.uuid4())
        ts = datetime.utcnow().isoformat()

        # Validate target exists using the dedicated validation function
        target_exists, error_msg = validate_target_exists(
            feedback_data.target_type,
            feedback_data.target_id,
            config["DB_PATH"]
        )

        if not target_exists:
            status_code = 404 if "not found" in error_msg else 400
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": error_msg, "trace_id": trace_id}
            )

        # Persist feedback
        try:
            with get_db(config["DB_PATH"]) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO feedback(trace_id, target_id, target_type, feedback_value, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (trace_id, feedback_data.target_id, feedback_data.target_type, feedback_data.feedback, ts),
                )
                conn.commit()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Database error: {str(e)}", "trace_id": trace_id}
            )

        return response_ok({
            "feedback_recorded": feedback_data.feedback,
            "target_type": feedback_data.target_type,
            "target_id": feedback_data.target_id
        }, trace_id=trace_id)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unexpected error: {str(e)}", "trace_id": trace_id}
        )


@app.post("/api/respond")
async def respond(req: Envelope):
    config = get_config()
    trace_id = req.trace_id or str(uuid.uuid4())
    payload = req.payload
    ts = datetime.utcnow().isoformat()

    content = payload.get("content", "")

    with get_db(config["DB_PATH"]) as conn:
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
    config = get_config()
    return response_ok({"service": "assistant-core", "db_path": config["DB_PATH"]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
