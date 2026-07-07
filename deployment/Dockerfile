##############################################################################
# deployment/Dockerfile
#
# PURPOSE:
#   Packages the entire MOSAIC application into a Docker container
#   that can run on Google Cloud Run.
#
# WHAT IS DOCKER?
#   Docker is a tool that packages your application AND everything it
#   needs to run (Python, libraries, OS tools) into one self-contained
#   unit called a container.
#
#   Think of it like a shipping container. A shipping container can be
#   put on any ship, any truck, any train — and the contents inside
#   always arrive in exactly the same condition.
#   A Docker container can run on any machine — your Mac, a GCP server,
#   AWS, Azure — and the application always runs exactly the same way.
#
# WHY --platform linux/amd64?
#   This is the most important line in this Dockerfile for Mac users.
#   Apple Silicon Macs (M1, M2, M3) use ARM64 architecture.
#   Google Cloud Run runs on AMD64 (Intel/AMD) architecture.
#   If you build a Docker image on your Mac without specifying --platform,
#   it builds for ARM64 — which Cloud Run cannot run.
#   --platform linux/amd64 forces Docker to build for Cloud Run's architecture
#   even though you are building on a Mac.
#   We discovered this during our original build — without it, Cloud Run
#   returned "exec format error" when trying to start the container.
#
# MULTI-STAGE BUILD — WHY WE USE IT:
#   We use two stages: "builder" and the final image.
#   Stage 1 (builder): installs all Python packages
#   Stage 2 (final):   copies only what is needed to run the app
#
#   WHY TWO STAGES?
#   Installing Python packages requires build tools (compilers, headers)
#   that are large and not needed at runtime.
#   With a single stage, all those build tools end up in the final image
#   — making it unnecessarily large and slower to deploy.
#   With two stages, build tools stay in stage 1 and never reach
#   the final image. Result: smaller, faster, more secure container.
#
# HOW TO BUILD THIS IMAGE:
#   From the project root (mosaic/ directory):
#   docker buildx build --platform linux/amd64 -t mosaic-api -f deployment/Dockerfile .
#
# HOW TO RUN LOCALLY FOR TESTING:
#   docker run -p 8000:8000 --env-file .env mosaic-api
#   Then test at http://localhost:8000/docs
##############################################################################


# ── STAGE 1: BUILDER ─────────────────────────────────────────────────────
# This stage installs all Python dependencies.
# It is called "builder" — we reference this name in Stage 2 to
# copy the installed packages across.

FROM --platform=linux/amd64 python:3.12-slim AS builder
# FROM sets the BASE IMAGE — the starting point for our container.
# python:3.12-slim is an official Python image.
# "slim" means it is a minimal image — only what Python needs to run.
# No GUI, no extra tools, no unnecessary packages.
# This keeps the base image small.
#
# AS builder names this stage "builder" so Stage 2 can reference it.

WORKDIR /app
# WORKDIR sets the working directory inside the container.
# All subsequent commands run from /app.
# If /app does not exist, Docker creates it automatically.
# Think of it as "cd /app" that persists for all future commands.

# Install system dependencies needed to compile some Python packages.
# Some Python packages (like asyncpg, psycopg2) include C extensions
# that need to be compiled from source. This requires build tools.
RUN apt-get update && apt-get install -y \
    gcc \
    # gcc is the GNU C Compiler — needed to compile C extensions in Python packages.
    libpq-dev \
    # libpq-dev contains PostgreSQL header files needed to compile
    # asyncpg and psycopg2 — our PostgreSQL drivers.
    curl \
    # curl is used in the health check command below.
    && apt-get clean \
    # apt-get clean removes cached package lists — reduces image size.
    && rm -rf /var/lib/apt/lists/*
    # Remove the package list cache — another size reduction.
    # These cleanup steps are best practice — every MB saved means
    # faster container startup times on Cloud Run.

COPY requirements.txt .
# COPY copies files from your local machine INTO the container.
# requirements.txt . means: copy requirements.txt to /app/requirements.txt
# We copy requirements BEFORE copying the source code.
# WHY? Docker caches each layer. If requirements.txt has not changed,
# Docker reuses the cached layer with installed packages — much faster rebuilds.
# If we copied all source code first, any code change would invalidate
# the package install cache and force a full reinstall every build.

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# pip install --no-cache-dir installs packages WITHOUT saving the
# download cache. Inside Docker, the cache would just waste space.
# --upgrade pip ensures we have the latest pip before installing.
# && chains commands — if the first fails, the second does not run.


# ── STAGE 2: FINAL PRODUCTION IMAGE ──────────────────────────────────────
# This stage creates the actual image that runs on Cloud Run.
# It copies the installed packages from Stage 1 but NOT the build tools.

FROM --platform=linux/amd64 python:3.12-slim
# Start fresh from the same slim base image.
# This image does NOT have gcc, libpq-dev, or other build tools.
# Much smaller and more secure than carrying build tools into production.

WORKDIR /app

# Copy installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# --from=builder means: copy FROM the stage named "builder".
# /usr/local/lib/python3.12/site-packages is where pip installs packages.
# We copy the entire installed packages directory — all our dependencies
# arrive in the final image without needing to reinstall them.

COPY --from=builder /usr/local/bin /usr/local/bin
# Copy Python executables (uvicorn, pip, etc.) from the builder stage.
# Without this, uvicorn would not be found when we try to start the server.

# Copy the application source code into the container.
COPY config/       ./config/
COPY ingestion/    ./ingestion/
COPY processing/   ./processing/
COPY memory/       ./memory/
COPY agents/       ./agents/
COPY tools/        ./tools/
COPY graph/        ./graph/
COPY api/          ./api/
COPY pyproject.toml .
# We copy each directory explicitly rather than COPY . .
# (which would copy everything including .env, .git, __pycache__ etc.)
# Explicit copies = only what the application actually needs.
# .env is intentionally NOT copied — secrets come from Secret Manager.

RUN pip install -e . --no-cache-dir
# pip install -e . installs our project in "editable" mode.
# This registers our package so Python can find our modules:
# from config.settings import settings  ← works because of this
# from agents.supervisor import ...      ← works because of this
# Without this, Python would throw ModuleNotFoundError for all imports.

# ── RUNTIME CONFIGURATION ──────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
# PYTHONUNBUFFERED=1 tells Python not to buffer stdout/stderr output.
# Without this, log lines might get buffered and appear all at once
# instead of streaming in real time.
# Cloud Run captures stdout for logs — we need real-time streaming
# so we can see log lines in the GCP Console as they happen.

ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from writing .pyc compiled bytecode files.
# In a container, these files just waste space — the container
# starts fresh each time anyway.

EXPOSE 8000
# EXPOSE documents that the container listens on port 8000.
# This is metadata only — it does not actually open the port.
# Cloud Run reads this to know which port to route traffic to.
# Our settings.py has API_PORT=8000 which must match this.

# ── HEALTH CHECK ───────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
# HEALTHCHECK tells Docker how to verify the container is healthy.
# --interval=30s   → check every 30 seconds
# --timeout=10s    → if no response in 10s, consider it failed
# --start-period=40s → wait 40s after startup before first check
#                      (gives the app time to initialise)
# --retries=3      → mark unhealthy only after 3 consecutive failures
# CMD curl ...     → the actual check: call our health endpoint
#                    curl -f returns exit code 1 on HTTP error (non-200)
#                    || exit 1 → if curl fails, return exit code 1
#                    exit code 1 = unhealthy, 0 = healthy

# ── START COMMAND ──────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD is the command that runs when the container starts.
# uvicorn is the ASGI server that runs our FastAPI application.
# api.main:app means: in the file api/main.py, find the object named app.
# --host 0.0.0.0 → listen on ALL network interfaces (required for Cloud Run)
# --port 8000    → listen on port 8000 (matches EXPOSE above)
#
# WHY NOT --reload?
# --reload watches for file changes and restarts automatically.
# That is useful in development but harmful in production —
# we never want our production server randomly restarting.