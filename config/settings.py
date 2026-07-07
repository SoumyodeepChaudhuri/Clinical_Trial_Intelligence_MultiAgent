##############################################################################
# config/settings.py
#
# PURPOSE:
#   Single source of truth for all configuration in MOSAIC.
#   Every environment variable the system needs is defined here,
#   typed, validated, and accessible from one single object.
#
# WHAT PROBLEM DOES THIS SOLVE:
#   Without centralised settings, developers scatter os.getenv() calls
#   across 30 different files. If an env var name changes, you have to
#   hunt it down everywhere. And if a required variable is missing,
#   you only find out when that specific line of code runs —
#   which might be 10 minutes into a complex agent run.
#
#   With Pydantic settings:
#   - All env vars are defined in ONE place
#   - All are TYPED — DB_PORT is validated, not a raw string
#   - All required vars are validated at STARTUP
#   - If OPENAI_API_KEY is missing, you get a clear error immediately
#     when the app starts — not when the first LLM call fires
#   This is called "fail fast" — better to crash at startup with a
#   clear message than to fail silently in the middle of a run.
#
# HOW TO USE IN ANY FILE:
#   from config.settings import settings
#   print(settings.openai_api_key)
#   print(settings.gcp_project_id)
#   print(settings.database_url)   ← computed property, built from parts
##############################################################################


from pydantic_settings import BaseSettings, SettingsConfigDict
# BaseSettings : it knows how to read values from environment variables
#                and .env files automatically.
# SettingsConfigDict : how we configure BaseSettings behaviour —
#                      where to find the .env file, encoding, case rules etc.

from pydantic import Field
# Field : adds metadata to each setting.
#   ...     = the field is REQUIRED — must exist in .env or app crashes at startup
#   default = a fallback value used if the env var is not set in .env


class Settings(BaseSettings):
    """
    All MOSAIC configuration defined in one place only.

    Pydantic reads these values from environment variables automatically.
    Field names here map directly to .env variable names.
    case_sensitive=False means:
      openai_api_key here → OPENAI_API_KEY in .env
      db_host here        → DB_HOST in .env
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        # Tell Pydantic where the .env file is.
        # Path is relative to where the app is started from —
        # always run from the mosaic/ root directory.

        env_file_encoding="utf-8",
        # Standard text encoding for .env files.
        # Ensures special characters in passwords are read correctly.

        case_sensitive=False,
        # OPENAI_API_KEY in .env matches openai_api_key here.
        # .env files are conventionally UPPERCASE.
        # Python attributes are conventionally lowercase.
        # case_sensitive=False bridges that gap automatically.

        extra="ignore",
        # If .env has variables we have not defined here, ignore them.
        # Without this, Pydantic raises an error for unknown fields.
        # "ignore" means: only read what we defined, skip everything else.
    )

    # ── OPENAI ────────────────────────────────────────────────

    openai_api_key: str = Field(
        ...,
        # ... means REQUIRED.
        # If OPENAI_API_KEY is not in .env, Pydantic raises
        # ValidationError immediately at startup with a clear message.
        description="Open AI API KEY for LLM and embeddings"
    )

    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        # text-embedding-3-large produces 3072-dimensional vectors.
        # Higher quality embeddings than the small model.
        # Use this when embedding quality matters more than cost.
        description="OpenAI model used to generate the vector embeddings"
    )

    openai_chat_model: str = Field(
        default="gpt-4o",
        # GPT-4o is the best reasoning model for complex signal detection.
        # All six specialist agents use this for their LLM reasoning calls.
        description="OpenAI model used for agent reasoning"
    )

    # ── LANGSMITH ─────────────────────────────────────────────

    langsmith_api_key: str = Field(
        default = "",
        # Required — LangSmith tracing must be enabled.
        # Without it we cannot see what agents are reasoning about.
        description="Langsmith API key - Optional"
    )

    langsmith_project: str = Field(
        default="clinical_trial_intelligence",
        # The LangSmith project name — all traces from this system
        # appear under this project in the LangSmith dashboard.
        description="Langsmith project name"
    )

    langsmith_tracing_v2: bool = Field(
        default=False,
        # True = LangSmith tracing is active.
        # Every LLM call, every tool call, every agent run is traced.
        # Set to False in .env to disable tracing (saves LangSmith quota).
        description="Enable langsmith tracing for all the agent runs"
    )

    # ── GCP ───────────────────────────────────────────────────

    gcp_project_id: str = Field(
        ...,
        # Required — every GCP API call needs the project ID.
        # Example: "mosaictest001"
        description="GCP Project ID"
    )

    gcp_region: str = Field(
        default="us-central1",
        # us-central1 is the cheapest and most available GCP region.
        # All resources — Cloud Run, Cloud SQL, GCS — are in this region.
        # Keeping everything in the same region avoids inter-region data fees.
        description="GCP region for all the cloud resources"
    )

    gcs_bucket_name: str = Field(
        ...,
        # Required — this is where all raw and processed documents are stored.
        # Example: "mosaic-data-bucket-001"
        description="Google cloud storage bucket name"
    )

    # ── DATABASE ──────────────────────────────────────────────

    db_host: str = Field(
        ...,
        # Required — the Cloud SQL host.
        # Local development: the public IP of your Cloud SQL instance.
        #   Example: "35.232.74.203"
        # Cloud Run production: the Unix socket path injected by GCP.
        #   Example: "/cloudsql/mosaictest001:us-central1:mosaic-db"
        # The vector_store.py connection code detects which format this is
        # and connects using TCP or Unix socket accordingly.
        description="Cloud SQL host IP (local) or socket path (Cloud Run)"
    )

    db_port: str = Field(
        default=5432,
        # PostgreSQL standard port.
        # Kept as str to avoid type conflicts when building connection strings.
        # asyncpg accepts both int and str for port.
        description="PostgreSQL port"
    )

    db_name: str = Field(
        default="clinical_trial_db",
        # The name of the database inside the Cloud SQL instance.
        # Created with: gcloud sql databases create clinical_trial_db
        description="PostgreSQL database name"
    )

    db_user: str = Field(
        ...,
        # Required — the database user created for this application.
        # Created with: gcloud sql users create <user> --instance=<instance>
        description="PostgreSQL database user"
    )

    db_password: str = Field(
        ...,
        # Required — the database password.
        # Never hardcode this anywhere in the codebase.
        # Always read from .env locally, from Secret Manager on Cloud Run.
        description="PostgreSQL database password"
    )

    # ── CLINICALTRIALS.GOV ────────────────────────────────────

    clinical_trials_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2",
        # The base URL for the ClinicalTrials.gov API version 2.
        # All endpoint calls in clinical_trials_client.py are built
        # from this base URL.
        description="ClinicalTrials.gov API V2 base url"
    )

    clinical_trials_page_size: int = Field(
        default=100,
        # How many studies to request per API page.
        # ClinicalTrials.gov maximum is 1000 but 100 is safer
        # for rate limits and memory — the client paginates automatically.
        description="Number of studies to fetch per API page"
    )

    # ── PUBMED ────────────────────────────────────────────────

    pubmed_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        # The base URL for PubMed's eUtils API.
        # Two endpoints we use:
        #   /esearch.fcgi → search for paper IDs matching a query
        #   /efetch.fcgi  → fetch full paper details by those IDs
        description="Pubmed eutils API base url"
    )

    # ── API SERVER ────────────────────────────────────────────

    api_host: str = Field(
        default="0.0.0.0",
        # 0.0.0.0 means: listen on ALL network interfaces.
        # Required for Cloud Run — the container must be reachable
        # from outside, not just from localhost (127.0.0.1).
        description="FAST API Host address"
    )

    api_port: int = Field(
        default=8000,
        # The port FastAPI listens on.
        # Local: http://localhost:8000
        # Cloud Run: port 8000 exposed via EXPOSE in Dockerfile
        description="FAST API port"
    )

    api_env: str = Field(
        default="development",
        # "development" or "production"
        # Controls logging verbosity and error detail level.
        description="Environment name : development or production"
    )

    # ── COMPUTED PROPERTIES ───────────────────────────────────

    @property
    def database_url(self) -> str:
        """
        Builds the full async PostgreSQL connection string from parts.

        We use asyncpg as the async PostgreSQL driver.
        asyncpg requires the connection string to start with:
        postgresql+asyncpg://

        Returns:
            str: Full connection URL ready for asyncpg.create_pool()
            Example: postgresql+asyncpg://mosaic_user:password@35.232.74.203:5432/clinical_trial_db
        """
        return (
            f"postgresql+asyncpg://"
            f"{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}"
            f"/{self.db_name}"
        )

    @property
    def is_production(self) -> bool:
        """
        Returns True if the app is running in production.
        """
        return self.api_env.lower() == "production"


# ── SINGLETON INSTANCE ────────────────────────────────────────
settings = Settings()
# Create ONE instance when this module is first imported.
# Every other file imports this same instance:
#   from config.settings import settings
#
# .env is read exactly ONCE at startup.
# All fields are validated ONCE at startup.
# If anything required is missing → clear error immediately.

# from config.settings import settings