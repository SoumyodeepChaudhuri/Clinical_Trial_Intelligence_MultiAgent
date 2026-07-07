##############################################################################
# api/routers/analysis.py
#
# PURPOSE:
#   Handles POST /api/v1/analyze — the most important endpoint.
#   This is the endpoint that triggers a full MOSAIC analysis run.
#
# WHAT HAPPENS WHEN THIS ENDPOINT IS CALLED:
#   1. Receives the analysis task and parameters
#   2. Invokes the compiled LangGraph graph with the task
#   3. The graph runs all 6 specialist agents in parallel
#   4. Signals are processed through the HITL gate
#   5. Supervisor compiles the final intelligence brief
#   6. Returns the complete analysis result to the caller
#
# THIS IS THE ENTRY POINT FOR EVERYTHING:
#   All the infrastructure we built — ingestion, processing,
#   memory, agents, tools — is exercised by this one endpoint.
##############################################################################


import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import AnalysisRequest, AnalysisResponse
from api.dependencies import get_hitl_gate, get_graph
from graph.hitl import HITLGate
from config.logging_config import setup_logging

logger = setup_logging(__name__)

router = APIRouter()
# APIRouter is FastAPI's way of grouping related endpoints.
# This router handles all analysis-related endpoints.
# It is mounted at /api/v1 prefix in main.py.


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    # response_model tells FastAPI to validate and serialise
    # the response using AnalysisResponse schema.
    # FastAPI will reject any response that does not match.

    summary="Trigger a full MOSAIC analysis run",
    description="Runs all 6 specialist agents in parallel and returns "
                "a compiled intelligence brief with all signals found.",
    tags=["Analysis"],
    # tags group endpoints in the Swagger UI at /docs.
)
async def run_analysis(
    request: AnalysisRequest,
    # FastAPI automatically reads and validates the request body
    # against AnalysisRequest schema. Invalid requests are rejected
    # with a 422 error before this function even runs.

    hitl_gate: HITLGate = Depends(get_hitl_gate),
    # Depends(get_hitl_gate) tells FastAPI to call get_hitl_gate()
    # and inject the result as "hitl_gate" parameter.
    # We use HITLGate to process each signal after agents finish.

    graph = Depends(get_graph),
    # The compiled LangGraph graph — ready to invoke.
):
    """
    Triggers a complete MOSAIC analysis run.

    Runs all 6 specialist agents in parallel:
    - Missing Results Agent
    - Broken Promises Agent
    - Track Record Agent
    - Pattern Finder Agent
    - Side Effect Checker
    - Timeline Analyst

    Returns a compiled intelligence brief with all signals found.
    """

    run_id     = str(uuid.uuid4())
    start_time = time.time()
    # Record start time — we report duration in the response.

    logger.info(
        f"Analysis run started | "
        f"run_id={run_id} | "
        f"task='{request.task[:80]}'"
    )

    try:
        # ── INVOKE THE LANGGRAPH GRAPH ──────────────────────────────────
        initial_state = {
            "task":        request.task,
            "nct_ids":     request.nct_ids,
            "max_studies": request.max_studies,
            "messages":    [],
            "signals":     [],
            "agents_activated": [],
            "final_brief": "",
            "run_complete": False,
            "run_id":      run_id,
            "error_log":   [],
        }
        # Build the initial state dict that gets passed to the first node.
        # Every field in MosaicState must have an initial value here.
        # LangGraph agents read from and write to this state as they run.

        result = await graph.ainvoke(initial_state)
        # ainvoke() is the async version of invoke().
        # This runs the entire graph — all agents — and waits for completion.
        # The graph runs supervisor_route → all 6 specialists in parallel
        # → supervisor_compile → returns final state.
        # This is the line where all the intelligence happens.

        # ── PROCESS SIGNALS THROUGH HITL ───────────────────────────────
        signals          = result.get("signals", [])
        processed_signals = []

        for signal in signals:
            # Route each signal through the HITL gate.
            # High confidence → saved directly to signals table.
            # Low confidence → sent to review queue.
            hitl_result = await hitl_gate.process_signal(signal)
            processed_signals.append({
                **signal,
                # ** unpacks the signal dict into the new dict.
                # This copies all existing signal fields.
                "hitl_action": hitl_result.get("action"),
                # Add the HITL routing decision to the signal.
            })

            logger.info(
                f"Signal saved directly | "
                f"agent={signal.get('agent')} | "
                f"confidence={signal.get('confidence', 0):.2f}"
            )

        duration = round(time.time() - start_time, 2)
        # Calculate total run duration in seconds.
        # round(..., 2) gives 2 decimal places: 15.24 seconds.

        signals_requiring_review = sum(
            1 for s in processed_signals
            if s.get("hitl_action") == "sent_to_review"
        )
        # Count how many signals went to the review queue.
        # sum() with a generator expression counts matching items.
        # 1 for each item where hitl_action == "sent_to_review".

        logger.info(
            f"Analysis run complete | "
            f"run_id={run_id} | "
            f"signals={len(signals)} | "
            f"duration={duration}s"
        )

        return AnalysisResponse(
            run_id=run_id,
            task=request.task,
            final_brief=result.get("final_brief", "No brief generated."),
            total_signals=len(signals),
            signals_requiring_review=signals_requiring_review,
            agents_activated=result.get("agents_activated", []),
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error(f"Analysis run failed | run_id={run_id} | error={e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis run failed: {str(e)}",
            # HTTPException tells FastAPI to return an error response.
            # status_code=500 = Internal Server Error.
            # detail contains the error message the caller sees.
        )