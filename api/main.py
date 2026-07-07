##############################################################################
# api/main.py
#
# PURPOSE:
#   The FastAPI application entry point.
#   This is the file that starts the entire API server.
#
# WHAT THIS FILE DOES:
#   1. Creates the FastAPI app with metadata
#   2. Defines the lifespan — what runs at startup and shutdown
#   3. Mounts all 4 routers at their URL prefixes
#   4. Adds CORS middleware for browser access
#   5. Defines the health check endpoint
#
# HOW TO RUN LOCALLY:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
#   Then open http://localhost:8000/docs for Swagger UI
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   Yes — this is the API entry point.
#   Run it with uvicorn as shown above.
##############################################################################


import asyncio
from contextlib import asynccontextmanager
# asynccontextmanager turns an async generator function into
# an async context manager — used for FastAPI's lifespan pattern.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# CORSMiddleware allows browsers to call our API from different origins.
# Without this, a browser at localhost:3000 cannot call our API at
# localhost:8000 — CORS security blocks it.

from api.routers import analysis, signals, review, memory
from api.dependencies import get_hitl_gate, get_procedural_store
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "api.main"


##############################################################################
# LIFESPAN — STARTUP AND SHUTDOWN LOGIC
#
# FastAPI's lifespan replaces the older @app.on_event("startup") pattern.
# Everything BEFORE yield runs at startup.
# Everything AFTER yield runs at shutdown.
# Using @asynccontextmanager makes this an async context manager.
##############################################################################

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.

    STARTUP (before yield):
    - Initialises default agent procedures in Cloud SQL
    - Logs that the API is ready

    SHUTDOWN (after yield):
    - Closes all database connection pools cleanly
    """

    # ── STARTUP ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MOSAIC API starting up...")
    logger.info("=" * 60)

    try:
        # Initialise default procedures for all agents.
        # This inserts default reasoning rules if they do not exist yet.
        # Safe to call on every startup — uses ON CONFLICT DO NOTHING.
        procedural_store = get_procedural_store()
        await procedural_store.initialise_defaults()
        logger.info("Default agent procedures initialised")

    except Exception as e:
        # Log but do not crash — API should start even if this fails.
        # Agents will still work, just without default procedures.
        logger.error(f"Failed to initialise procedures | error={e}")

    logger.info("MOSAIC API ready | version=0.1.0")
    logger.info("=" * 60)

    yield
    # Everything above runs at startup.
    # Everything below runs at shutdown.

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("MOSAIC API shutting down...")

    try:
        # Close the HITL gate connection pool.
        hitl = get_hitl_gate()
        await hitl.close()
        logger.info("HITLGate pool closed")
    except Exception as e:
        logger.error(f"Error closing HITLGate | error={e}")

    logger.info("MOSAIC API shutdown complete")


##############################################################################
# CREATE THE FASTAPI APP
##############################################################################

app = FastAPI(
    title="MOSAIC — Clinical Trial Intelligence API",
    description=(
        "Multi-Agent Operating System for AI Cognition. "
        "Detects research integrity signals across clinical trials "
        "using 6 specialist AI agents running in parallel on GCP."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # Attach our startup/shutdown logic to the app.

    docs_url="/docs",
    # Swagger UI available at http://localhost:8000/docs
    # Auto-generated from all our Pydantic schemas and endpoint definitions.

    redoc_url="/redoc",
    # Alternative ReDoc documentation at http://localhost:8000/redoc
)


##############################################################################
# CORS MIDDLEWARE
##############################################################################

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.api_env == "development" else [],
    # Development: allow ALL origins (convenient for testing)
    # Production:  allow NO origins via CORS (Cloud Run handles auth)
    # settings.api_env comes from .env: "development" or "production"

    allow_credentials=True,
    allow_methods=["*"],
    # Allow all HTTP methods: GET, POST, PATCH, DELETE etc.

    allow_headers=["*"],
    # Allow all headers including Authorization.
)


##############################################################################
# MOUNT ROUTERS
##############################################################################

app.include_router(
    analysis.router,
    prefix="/api/v1",
    # All endpoints in analysis.router are prefixed with /api/v1
    # So /analyze becomes /api/v1/analyze
)

app.include_router(
    signals.router,
    prefix="/api/v1",
    # /signals   → /api/v1/signals
    # /signals/{id} → /api/v1/signals/{id}
)

app.include_router(
    review.router,
    prefix="/api/v1",
    # /review/queue → /api/v1/review/queue
    # /review/{id}  → /api/v1/review/{id}
)

app.include_router(
    memory.router,
    prefix="/api/v1",
    # /memory/episodes → /api/v1/memory/episodes
    # /memory/procedures/{agent} → /api/v1/memory/procedures/{agent}
    # /sponsors → /api/v1/sponsors
    # /sponsors/{name} → /api/v1/sponsors/{name}
)


##############################################################################
# HEALTH CHECK ENDPOINT
##############################################################################

@app.get(
    "/api/v1/health",
    summary="System health check",
    description="Returns system status including database connectivity "
                "and queue depth. Called by Cloud Run to verify the "
                "container is healthy before routing traffic.",
    tags=["System"],
)
async def health_check():
    """
    Returns the health status of the MOSAIC system.

    Cloud Run calls this endpoint automatically to check if the
    container is healthy. If this returns non-200, Cloud Run marks
    the instance as unhealthy and stops routing traffic to it.
    """

    import asyncpg
    db_status   = "connected"
    signals_count = 0
    pending_count = 0
    episodes_count = 0

    try:
        pool = await asyncpg.create_pool(
            host=settings.db_host, port=settings.db_port,
            database=settings.db_name, user=settings.db_user,
            password=settings.db_password, min_size=1, max_size=2,
        )
        async with pool.acquire() as conn:
            signals_count  = await conn.fetchval("SELECT COUNT(*) FROM signals") or 0
            pending_count  = await conn.fetchval(
                "SELECT COUNT(*) FROM hitl_reviews WHERE decision = 'pending'"
            ) or 0
            episodes_count = await conn.fetchval("SELECT COUNT(*) FROM episodes") or 0
        await pool.close()

    except Exception as e:
        db_status = f"disconnected: {str(e)}"
        logger.error(f"Health check DB error | error={e}")

    return {
        "status":   "healthy" if db_status == "connected" else "degraded",
        "app":      "MOSAIC",
        "version":  "0.1.0",
        "database": db_status,
        "details": {
            "signals_in_db":   signals_count,
            "pending_reviews": pending_count,
            "episodes_count":  episodes_count,
            "queue_depth":     pending_count,
        },
    }


##############################################################################
# ROOT ENDPOINT
##############################################################################

@app.get("/", tags=["System"])
async def root():
    """Root endpoint — confirms the API is running."""
    return {
        "message": "MOSAIC Clinical Trial Intelligence API",
        "version": "0.1.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }