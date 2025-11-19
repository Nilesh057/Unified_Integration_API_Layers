"""
Pydantic models for request/response validation in the Unified Cognitive Intelligence API.

This module defines all the data models used for API request/response validation,
ensuring type safety and data integrity throughout the application.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from datetime import datetime


class Envelope(BaseModel):
    """Generic envelope for API requests containing payload and optional trace_id."""
    payload: Dict[str, Any]
    trace_id: Optional[str] = None


class FeedbackPayload(BaseModel):
    """Request model for feedback submission with strict validation."""
    feedback: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Feedback value between -1.0 (negative) and 1.0 (positive)"
    )
    target_id: int = Field(
        ...,
        gt=0,
        description="ID of the target entity (message, task, or decision)"
    )
    target_type: str = Field(
        ...,
        pattern="^(message|task|decision)$",
        description="Type of target entity"
    )

    @validator('target_type')
    def validate_target_type(cls, v):
        """Ensure target_type is one of the allowed values."""
        allowed = {'message', 'task', 'decision'}
        if v not in allowed:
            raise ValueError(f'target_type must be one of {allowed}')
        return v


class DecisionHubPayload(BaseModel):
    """Request model for decision hub with comprehensive validation."""
    source: str = Field(..., min_length=1, description="Source of the message")
    content: str = Field(..., min_length=1, description="Message content")
    rl_reward: float = Field(..., description="RL reward signal")
    user_feedback: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="User feedback between -1.0 and 1.0"
    )
    action_success: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Action success probability between 0.0 and 1.0"
    )
    cognitive_score: float = Field(..., description="Cognitive processing score")
    confidences: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional dynamic confidence scores per agent"
    )


class EmbedPayload(BaseModel):
    """Request model for text embedding."""
    text: str = Field(..., min_length=1, description="Text to embed")


class SummarizePayload(BaseModel):
    """Request model for text summarization."""
    text: str = Field(..., min_length=1, description="Text to summarize")


class ProcessSummaryPayload(BaseModel):
    """Request model for summary processing."""
    summary: str = Field(..., min_length=1, description="Summary text to process")


class RespondPayload(BaseModel):
    """Request model for response generation."""
    content: str = Field(..., min_length=1, description="Content for response")


class AgentActionPayload(BaseModel):
    """Request model for agent actions."""
    action: str = Field(..., min_length=1, description="Action to perform")
    reward: float = Field(..., description="Reward for the action")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the action between 0.0 and 1.0"
    )


# Response Models

class APIResponse(BaseModel):
    """Base response model for all API endpoints."""
    status: str = Field(..., description="Response status ('ok' or 'error')")
    timestamp: str = Field(..., description="ISO timestamp of response")
    trace_id: str = Field(..., description="Request trace ID")
    data: Dict[str, Any] = Field(..., description="Response data")


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = Field(default="error", description="Always 'error'")
    timestamp: str = Field(..., description="ISO timestamp of error")
    trace_id: Optional[str] = Field(None, description="Request trace ID if available")
    message: str = Field(..., description="Error message")


class DecisionResponse(BaseModel):
    """Response model for decision hub."""
    decision: str = Field(..., description="Decision outcome ('proceed' or 'defer')")
    top_agent: str = Field(..., description="Agent with highest contribution")
    final_score: float = Field(..., description="Final decision score")
    final_confidence: float = Field(..., description="Confidence in the decision")
    decision_trace: list = Field(..., description="Detailed decision trace")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_recorded: float = Field(..., description="The feedback value that was recorded")
    target_type: str = Field(..., description="Type of target entity")
    target_id: int = Field(..., description="ID of target entity")


class EmbedResponse(BaseModel):
    """Response model for text embedding."""
    embedding: list = Field(..., description="Embedding vector")
    dim: int = Field(..., description="Embedding dimension")


class SummarizeResponse(BaseModel):
    """Response model for text summarization."""
    summary: str = Field(..., description="Generated summary")


class ProcessSummaryResponse(BaseModel):
    """Response model for summary processing."""
    cognitive_score: float = Field(..., description="Computed cognitive score")
    processed: str = Field(..., description="Processed summary text")


class RespondResponse(BaseModel):
    """Response model for response generation."""
    response: str = Field(..., description="Generated response")


class AgentActionResponse(BaseModel):
    """Response model for agent actions."""
    action: str = Field(..., description="Action performed")
    reward: float = Field(..., description="Reward received")
    confidence: float = Field(..., description="Confidence level")


class HealthResponse(BaseModel):
    """Response model for health check."""
    service: str = Field(..., description="Service name")
    db_path: str = Field(..., description="Database path")