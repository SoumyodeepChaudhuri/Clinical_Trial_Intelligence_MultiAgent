##############################################################################
# api/schemas.py
#
# PURPOSE:
#   Defines all the REQUEST and RESPONSE shapes for every API endpoint.
#
# WHAT IS A SCHEMA?
#   A schema is a contract. It says:
#   "When you call this endpoint, send data in THIS shape.
#    When I respond, I will always respond in THAT shape."
#
#   Without schemas, APIs are chaotic — callers never know what to send
#   or what they will get back. With Pydantic schemas, everything is:
#   - TYPED: every field has a declared type
#   - VALIDATED: Pydantic rejects wrong types immediately with clear errors
#   - DOCUMENTED: FastAPI reads these schemas and auto-generates Swagger docs
#   - SERIALISABLE: Pydantic converts to/from JSON automatically
#
# HOW FASTAPI USES THESE:
#   Every API endpoint declares its request and response schema.
#   FastAPI automatically:
#   - Validates incoming request data against the request schema
#   - Serialises outgoing response data using the response schema
#   - Generates Swagger documentation at /docs showing all schemas
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. api/main.py imports and uses these schemas.
##############################################################################


from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
# BaseModel is the Pydantic base class for all schemas.
# Field lets us add default values, descriptions, and constraints.
# These descriptions appear in the auto-generated Swagger docs.


##############################################################################
# REQUEST SCHEMAS — what the API RECEIVES from callers
##############################################################################

class AnalysisRequest(BaseModel):
    """
    Request body for POST /api/v1/analyze

    This is what a caller sends to trigger a full MOSAIC analysis run.
    All fields have defaults so the simplest possible call is just:
    POST /api/v1/analyze {}
    """

    task: str = Field(
        default="Find completed clinical trials with research integrity issues",
        description="The analysis task in plain English. "
                    "Agents read this and decide what to investigate.",
        example="Find completed trials where sponsor never posted results",
    )

    nct_ids: list[str] = Field(
        default=[],
        description="Specific NCT IDs to analyse. "
                    "Empty list means analyse broadly across all studies.",
        example=["NCT04788680", "NCT02208921"],
    )

    max_studies: int = Field(
        default=10,
        ge=1,
        # ge=1 means "greater than or equal to 1".
        # Pydantic rejects max_studies=0 or negative numbers.
        le=100,
        # le=100 means "less than or equal to 100".
        # Prevents callers from accidentally triggering massive runs.
        description="Maximum studies to analyse per agent. Default 10.",
    )


class ReviewDecisionRequest(BaseModel):
    """
    Request body for PATCH /api/v1/review/{queue_id}

    Submitted by a human analyst after reviewing a queued signal.
    """

    decision: str = Field(
        ...,
        # ... means REQUIRED — no default value.
        description="The reviewer's decision: approve, reject, or edit",
        example="reject",
    )

    reviewer: str = Field(
        default="analyst",
        description="Name or ID of the human reviewer.",
        example="chirantan@tarkaupskilling.com",
    )

    rejection_reason: str = Field(
        default="",
        description="Why this signal was rejected. "
                    "IMPORTANT: This gets written to procedural memory "
                    "and permanently changes how the agent reasons. "
                    "Be specific and clear.",
        example="This trial was terminated early due to COVID — "
                "terminated trials are exempt from result posting requirements.",
    )

    edit_summary: str = Field(
        default="",
        description="Corrected signal summary if decision is 'edit'. "
                    "Replaces the agent's original summary.",
    )


##############################################################################
# RESPONSE SCHEMAS — what the API SENDS BACK to callers
##############################################################################

class SignalResponse(BaseModel):
    """
    One signal in an API response.
    Matches the structure of a row in the signals table.
    """

    signal_id:   str
    nct_id:      str
    agent:       str
    signal_type: str
    summary:     str
    confidence:  float
    status:      str
    created_at:  str

    class Config:
        # Config tells Pydantic how to handle the model.
        from_attributes = True
        # from_attributes=True allows creating this model from
        # ORM objects or database rows — not just plain dicts.
        # asyncpg returns Record objects — this makes them
        # automatically compatible with our Pydantic schema.


class AnalysisResponse(BaseModel):
    """
    Response from POST /api/v1/analyze

    Contains the complete results of one analysis run.
    """

    run_id:                   str
    task:                     str
    final_brief:              str
    # The GPT-4o compiled intelligence brief — the main deliverable.
    # Plain English narrative ready for an analyst to read and act on.

    total_signals:            int
    signals_requiring_review: int
    agents_activated:         list[str]
    duration_seconds:         float
    # How long the full analysis run took.
    # Useful for benchmarking and optimisation.


class ReviewQueueItem(BaseModel):
    """
    One item in the human review queue.
    Returned by GET /api/v1/review/queue
    """

    review_id:   str
    signal_id:   str
    agent:       str
    signal_type: str
    summary:     str
    confidence:  float
    nct_id:      str
    decision:    str
    # "pending" for all items in the queue — changes after review.


class ReviewQueueResponse(BaseModel):
    """
    Full response from GET /api/v1/review/queue
    Contains the queue plus summary statistics.
    """

    queue:          list[ReviewQueueItem]
    total_pending:  int
    total_approved: int
    total_rejected: int


class ReviewDecisionResponse(BaseModel):
    """
    Response from PATCH /api/v1/review/{queue_id}
    Confirms the decision was recorded.
    """

    success:          bool
    decision:         str
    signal_id:        str
    queue_id:         str
    memory_updated:   bool
    # True if this was a rejection — meaning procedural memory
    # was updated with the rejection reason (the learning loop).
    message:          str


class EpisodeResponse(BaseModel):
    """
    One episode from episodic memory.
    Returned by GET /api/v1/memory/episodes
    """

    episode_id:  str
    agent_name:  str
    nct_id:      str | None
    content:     str
    outcome:     str | None
    similarity:  float | None
    # similarity is only present when the episode was found
    # via semantic search — not when listing recent episodes.
    created_at:  str


class ProcedureResponse(BaseModel):
    """
    One reasoning rule from procedural memory.
    Returned by GET /api/v1/memory/procedures/{agent_name}
    """

    procedure_id: str
    agent_name:   str
    rule_text:    str
    rule_type:    str
    # "default" = built-in rule
    # "learned"  = added from human HITL feedback
    source:       str
    created_at:   str


class SponsorProfileResponse(BaseModel):
    """
    Full sponsor credibility profile.
    Returned by GET /api/v1/sponsors/{sponsor_name}
    """

    sponsor:           str
    credibility_score: float
    total_studies:     int
    results_posted:    int
    results_missing:   int
    broken_promises:   int
    avg_delay_days:    float
    last_updated:      str


class HealthResponse(BaseModel):
    """
    System health check response.
    Returned by GET /api/v1/health
    """

    status:      str
    # "healthy" if all systems operational, "degraded" if issues.

    app:         str
    version:     str
    database:    str
    # "connected" or "disconnected"

    details:     dict[str, Any]
    # Contains: signals_in_db, pending_reviews, episodes_count etc.