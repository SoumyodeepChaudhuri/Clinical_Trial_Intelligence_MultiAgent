# Multi-Agent Clinical Trial Intelligence System

> A production-grade multi-agent AI system that detects research integrity signals across clinical trials using 6 specialist agents running in parallel on Google Cloud Platform.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.6-orange.svg)](https://langchain-ai.github.io/langgraph)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-blue.svg)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

- [Project Description](#-project-description)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📌 Project Description

**MOSAIC** (*Multi-Agent Operating System for AI Cognition*) is a production-grade multi-agent AI system that ingests clinical trial data from **ClinicalTrials.gov** and **PubMed**, reasons across studies simultaneously, detects research integrity signals, and exposes a FastAPI REST layer — deployed on Google Cloud Run.

**The problem it solves:**  
About 30% of completed clinical trials never publish their results. Outcome switching, timeline delays, and safety discrepancies are buried across 400,000+ government records that no single human can read simultaneously. MOSAIC reads all of them at once and surfaces what humans miss.

**What makes it different:**
- **6 specialist agents** run in parallel — not sequentially
- **Three types of memory** — episodic, procedural, semantic
- **Learns from human feedback** — rejections update agent reasoning permanently
- **Production deployed** on GCP for under $55/month

---

## 🎬 Demo

```bash
# Run a live analysis against production
curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"task": "Find completed clinical trials where results were never posted", "max_studies": 3}' \
  [https://mosaic-api-569957100480.us-central1.run.app/api/v1/analyze](https://mosaic-api-569957100480.us-central1.run.app/api/v1/analyze) | python3 -m json.tool

```

**Sample response:**

```json
{
  "run_id": "a5712262-7b94-4bfc-9d46-6861000681fe",
  "task": "Find completed clinical trials where results were never posted",
  "final_brief": "Three completed clinical trials identified with results overdue by several years...",
  "total_signals": 3,
  "signals_requiring_review": 0,
  "agents_activated": ["missing_results_agent", "track_record_agent"],
  "duration_seconds": 15.24
}

```

---

## ✨ Features

### 6 Specialist Agents Running in Parallel

| Agent | What it detects |
| --- | --- |
| **Missing Results** | Completed trials with no results posted (legal violation after 12 months) |
| **Broken Promises** | Outcome switching — goals changed mid-study |
| **Track Record** | Sponsor credibility scores built over time |
| **Pattern Finder** | Cross-study patterns invisible to single-study readers |
| **Side Effect Checker** | Safety gaps between official filings and published papers |
| **Timeline Analyst** | Silent delays past completion date with no explanation |

### Three-Layer Memory System

* **Episodic** — Agents remember what they found in past sessions.
* **Procedural** — Agents learn from human corrections permanently.
* **Semantic** — Sponsor knowledge base grows with every analysis run.

### Human-in-the-Loop Gate

* Low-confidence signals are routed to a human review queue.
* Rejections are written back to procedural memory, permanently updating how agents reason.
* A single correction adapts agent behavior in all future sessions.

### Production Infrastructure

* **Serverless on Cloud Run** — Scales to zero, costs nothing when idle.
* **PostgreSQL + pgvector** — Semantic search over 1536-dimensional embeddings.
* **FastAPI REST API** — 9 fully interactive REST endpoints with automated OpenAPI docs.

---

## 🛠 Tech Stack

| Layer | Technology |
| --- | --- |
| **Language** | Python 3.12 (Async) |
| **Agent Framework** | LangGraph 0.2.6 |
| **LLM** | GPT-4o (Reasoning & Graph Orchestration) |
| **Embeddings** | OpenAI text-embedding-3-small (1536 dims) |
| **API Framework** | FastAPI + Uvicorn |
| **Database** | PostgreSQL 15 + pgvector (Cloud SQL) |
| **Vector Search** | pgvector HNSW Index |
| **Storage** | Google Cloud Storage |
| **Deployment** | Google Cloud Run |
| **Secrets** | GCP Secret Manager |
| **HTTP Clients** | requests (ClinicalTrials.gov) + httpx (PubMed) |
| **Validation & Retries** | Pydantic v2 + Tenacity |

---

## 🏗 Architecture

```text
               +----------------------------------+
               |       Clinical Data Sources      |
               | (ClinicalTrials.gov / PubMed)    |
               +----------------+-----------------+
                                |
                                v
               +----------------------------------+
               |         Ingestion Layer          |
               | (CT Client / PubMed Client / GCS)|
               +----------------+-----------------+
                                |
                                v
               +----------------------------------+
               |         Processing Layer         |
               | (Chunker / OpenAI Embedder / DB) |
               +----------------+-----------------+
                                |
                                v
               +----------------------------------+
               |      Memory Layer (LangMem)      |
               | (Episodic / Procedural / Semantic)|
               +----------------+-----------------+
                                |
                                v
               +----------------------------------+
               |     LangGraph Orchestrator      |
               |          (Supervisor)            |
               +----------------+-----------------+
                                |
      +-----------------+-------+-------+-----------------+
      |                 |               |                 |
      v                 v               v                 v
+-----------+    +-------------+    +-------+    +------------------+
|  Missing  |    |   Broken    |    | Track |    |  Pattern / Side  |
|  Results  |    |  Promises   |    | Record|    |  Effect / Timeline|
+-----+-----+    +------+------+    +---+---+    +--------+---------+
      |                 |               |                 |
      +-----------------+-------+-------+-----------------+
                                |
                                v
               +----------------------------------+
               |        Human-in-the-Loop         |
               |        (Verification Gate)       |
               +----------------+-----------------+
                                |
                                v
               +----------------------------------+
               |   FastAPI Layer (Cloud Run)      |
               +----------------------------------+

```

---

## 📁 Project Structure

```text
mosaic/
├── config/
│   ├── settings.py           # Pydantic BaseSettings, configuration & env vars
│   └── logging_config.py     # Centralized logging setup
├── ingestion/
│   ├── clinical_trials_client.py
│   ├── pubmed_client.py
│   ├── document_parser.py
│   ├── gcs_store.py
│   └── run_ingestion.py       # Data ingestion pipeline entrypoint
├── processing/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── run_processing.py      # Embedding & vector index creation pipeline
├── memory/
│   ├── episodic_store.py
│   ├── procedural_store.py
│   └── semantic_store.py
├── agents/
│   ├── supervisor.py
│   ├── broken_promises_agent.py
│   ├── missing_results_agent.py
│   ├── track_record_agent.py
│   ├── pattern_finder_agent.py
│   ├── side_effect_agent.py
│   └── timeline_agent.py
├── tools/
│   ├── search_tools.py        # Database & memory query tools
│   ├── clinical_tools.py      # ClinicalTrials.gov API tools
│   └── pubmed_tools.py        # Live PubMed query tools
├── graph/
│   ├── state.py               # MosaicState TypedDict
│   ├── graph_builder.py       # Wires multi-agent graph with LangGraph
│   └── hitl.py                # Human-in-the-Loop review & procedural memory loop
├── api/
│   ├── main.py                # FastAPI application entrypoint
│   ├── schemas.py             # Request & response Pydantic schemas
│   ├── dependencies.py        # Dependency injection & singletons
│   └── routers/
│       ├── analysis.py        # POST /api/v1/analyze
│       ├── signals.py         # GET /api/v1/signals
│       ├── review.py          # GET/PATCH /api/v1/review
│       └── memory.py          # GET /api/v1/memory & /sponsors
├── deployment/
│   ├── Dockerfile             # Multi-stage production container build
│   ├── .dockerignore
│   └── gcp/
│       ├── cloudsql_init.sql  # Database schemas & pgvector extensions
│       └── deploy.sh          # GCP Cloud Run deployment script
├── .env.example               # Template environment configuration file
├── pyproject.toml
├── requirements.txt
└── README.md

```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.12+
* Docker Desktop (for container deployment)
* Google Cloud account with active billing
* OpenAI API Key

### 1. Clone the repository

```bash
git clone [https://github.com/SoumyodeepChaudhuri/Trial_Intelligence_MultiAgent.git](https://github.com/SoumyodeepChaudhuri/Trial_Intelligence_MultiAgent.git)
cd Trial_Intelligence_MultiAgent

```

### 2. Set up virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .

```

### 4. Set up GCP Infrastructure

```bash
# Authenticate
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Enable required GCP APIs
gcloud services enable sqladmin.googleapis.com storage.googleapis.com run.googleapis.com secretmanager.googleapis.com --project=YOUR_PROJECT_ID

# Create Google Cloud Storage bucket
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_BUCKET_NAME

# Create Cloud SQL Instance
gcloud sql instances create clinical-trial-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --zone=us-central1-f \
  --project=YOUR_PROJECT_ID

# Create database and user
gcloud sql databases create clinical_trial_db --instance=clinical-trial-db
gcloud sql users create mosaic_user --instance=clinical-trial-db --password=YOUR_PASSWORD

```

### 5. Initialize Database Schema

```bash
# Authorize your local IP to access Cloud SQL
curl -4 ifconfig.me

gcloud sql instances patch clinical-trial-db \
  --authorized-networks=YOUR_IP/32 \
  --project=YOUR_PROJECT_ID

# Connect to database and execute schema script
psql "host=YOUR_SQL_IP port=5432 dbname=clinical_trial_db user=mosaic_user" -f deployment/gcp/cloudsql_init.sql

```

### 6. Environment Configuration

```bash
cp .env.example .env
# Open .env and fill in required environment credentials

```

### 7. Run Ingestion & Processing Pipeline

```bash
# 1. Fetch studies from APIs -> GCS
python3 ingestion/run_ingestion.py

# 2. Chunk documents, compute embeddings, & store in Cloud SQL
python3 processing/run_processing.py

```

### 8. Run API Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

```

Access Swagger API Documentation at: `http://localhost:8000/docs`

---

## 🔐 Environment Variables

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o

# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=your-bucket-name

# PostgreSQL / Cloud SQL Credentials
DB_HOST=34.XXX.XXX.XXX
DB_PORT=5432
DB_NAME=clinical_trial_db
DB_USER=mosaic_user
DB_PASSWORD=your-password

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development

```

---

## 📡 Usage

### Run Multi-Agent Analysis

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"task": "Find completed trials with missing results", "max_studies": 5}' \
  http://localhost:8000/api/v1/analyze | python3 -m json.tool

```

### Query Pending Review Queue

```bash
curl -s http://localhost:8000/api/v1/review/queue | python3 -m json.tool

```

### Submit Human Review Decision

```bash
curl -s -X PATCH \
  -H "Content-Type: application/json" \
  -d '{"decision": "reject", "reviewer": "analyst@company.com", "rejection_reason": "Trial was terminated early — exempt from posting requirement"}' \
  http://localhost:8000/api/v1/review/QUEUE_ID_HERE | python3 -m json.tool

```

### Search Agent Procedural Rules

```bash
curl -s http://localhost:8000/api/v1/memory/procedures/missing_results_agent | python3 -m json.tool

```

---

## 📚 API Documentation

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/v1/analyze` | Trigger multi-agent analysis graph |
| `GET` | `/api/v1/signals` | List all detected research integrity signals |
| `GET` | `/api/v1/signals/{id}` | Fetch detailed signal record by ID |
| `GET` | `/api/v1/review/queue` | List items awaiting human review |
| `PATCH` | `/api/v1/review/{id}` | Approve, reject, or edit flagged signals |
| `GET` | `/api/v1/memory/episodes` | Semantic search over historical session memory |
| `GET` | `/api/v1/memory/procedures/{agent}` | Get updated reasoning rules for a specific agent |
| `GET` | `/api/v1/sponsors` | List sponsor profiles |
| `GET` | `/api/v1/sponsors/{name}` | Fetch credibility metrics for a specific sponsor |
| `GET` | `/api/v1/health` | System health check endpoint |

---

## ☁️ Deployment

### Google Cloud Run Deployment

```bash
# Make deployment script executable
chmod +x deployment/gcp/deploy.sh

# Run deployment pipeline
./deployment/gcp/deploy.sh

```

Or manually containerize and deploy via Docker & gcloud CLI:

```bash
# Build & push image (AMD64 platform required for Cloud Run)
docker buildx build \
  --platform linux/amd64 \
  --tag gcr.io/YOUR_PROJECT_ID/mosaic-api:latest \
  --file deployment/Dockerfile \
  --push .

# Deploy container image to Google Cloud Run
gcloud run deploy mosaic-api \
  --image=gcr.io/YOUR_PROJECT_ID/mosaic-api:latest \
  --platform=managed \
  --region=us-central1 \
  --memory=2Gi \
  --cpu=2 \
  --port=8000 \
  --project=YOUR_PROJECT_ID

```

---

## 🗺 Roadmap

* [ ] Add evaluation layer using LangSmith evals
* [ ] Build a Streamlit analytics dashboard for signal review
* [ ] Ingest FDA Adverse Event Reporting System (FAERS) database as a third data source
* [ ] Configure scheduled periodic runs via GCP Cloud Scheduler
* [ ] Integrate automated email alerts for high-confidence risk signals
* [ ] Multi-sponsor comparative analysis dashboards
* [ ] Support international trial registries (EudraCT, ISRCTN)

---

## 📄 License

Distributed under the MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.
