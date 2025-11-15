# Unified Cognitive Intelligence Architecture - Final Documentation

## Overview
This project implements a unified multi-agent AI Assistant that integrates various subsystems into a live, testable, and deployable architecture. The system processes messages from multiple sources (WhatsApp, Instagram, Email) through summarization, cognitive processing, reinforcement learning, and decision-making, culminating in actionable responses.

## Architecture Diagram
```
Message Input (WhatsApp/Instagram/Email)
    ↓
Summarizer Agent (/api/summarize)
    ↓
Cognitive Processor (/api/process_summary)
    ↓
Decision Hub (/api/decision_hub) - Reward Fusion & Routing
    ↓
RL Agent (/api/agent_action) or Embed Core (/api/embed) or Action Sense (/api/respond)
    ↓
Feedback Loop (/api/feedback)
    ↓
Persistent Storage (assistant_core.db) & Logging (decision_log.csv)
```

## Core Components

### 1. Reward Fusion & Decision Architecture
Located in `core/reward_fusion.py`, this module merges multiple feedback signals:
- RL Reward: From reinforcement learning agent
- User Feedback: Thumbs-up/down ratings
- Action Success: Probability of successful execution
- Cognitive Alignment Score: From cognitive processing

The fusion uses weighted averaging based on agent registry weights and dynamic confidences, producing a final decision ("proceed" or "defer") with confidence scoring.

### 2. Agent Registry
Defined in `config/agent_registry.json`, it specifies agents and their routing weights:
- summarizer: 0.3
- cognitive: 0.3
- rl_agent: 0.2
- embedcore: 0.1
- actionsense: 0.1

### 3. Unified Database
`data/assistant_core.db` contains tables:
- messages: Incoming messages with source and content
- summaries: Summarized content
- tasks: Actionable tasks
- feedback: User feedback
- decisions: Fused decision outcomes
- embeddings: Vector representations
- rl_logs: Reinforcement learning actions and rewards

### 4. API Endpoints
- `/api/summarize`: Summarize input text
- `/api/process_summary`: Cognitive processing of summaries
- `/api/feedback`: Submit user feedback
- `/api/decision_hub`: Main decision fusion endpoint
- `/api/agent_action`: RL agent actions
- `/api/embed`: Generate embeddings
- `/api/respond`: Action responses

All endpoints return consistent JSON with status, timestamp, trace_id, and data.

## Example API Usage

### Decision Hub Request
```json
{
  "payload": {
    "source": "whatsapp",
    "content": "Schedule a meeting tomorrow",
    "rl_reward": 0.8,
    "user_feedback": 1.0,
    "action_success": 0.9,
    "cognitive_score": 0.7
  }
}
```

### Response
```json
{
  "status": "ok",
  "timestamp": "2025-11-12T10:00:00.000Z",
  "trace_id": "uuid-123",
  "data": {
    "decision": "proceed",
    "top_agent": "cognitive",
    "final_score": 0.75,
    "final_confidence": 0.82,
    "decision_trace": [...]
  }
}
```

## Reward Fusion Explanation
The reward fusion algorithm normalizes all signals to [0,1], applies base weights from registry, multiplies by dynamic confidences, and computes weighted sum. Confidence is derived from entropy reduction in the weight distribution. The top agent is selected by contribution, and decision threshold is 0.5.

## Visualization Dashboard
Run `streamlit run app/dashboard.py` to view:
- Flow diagram
- Reward trends (Matplotlib line plots)
- Agent influence (pie chart)
- Logs viewer (dataframes from DB and CSV)

## Deployment Instructions
1. Install dependencies: `pip install fastapi uvicorn streamlit matplotlib pandas sqlite3`
2. Run API server: `python app/main.py`
3. Run dashboard: `streamlit run app/dashboard.py`
4. For production, deploy backend to Render/Vercel, frontend separately.

## Testing
Run `python tests/test_full_pipeline.py` to simulate full pipeline and verify:
- API responses (200 status)
- Database entries
- CSV logging
- Decision fusion

## VALUES.md Reflections
- **Humility**: Acknowledged limitations in current embeddings and sought collaborative improvements.
- **Gratitude**: Thanked team for support in integrating complex subsystems.
- **Honesty**: Transparently documented assumptions and areas needing further development.