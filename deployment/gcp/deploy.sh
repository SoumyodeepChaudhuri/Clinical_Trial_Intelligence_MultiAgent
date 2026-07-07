#!/bin/bash
##############################################################################
# deployment/gcp/deploy.sh
#
# PURPOSE:
#   Complete deployment script for MOSAIC to Google Cloud Run.
#   Run this once to deploy — or re-run to deploy a new version.
#
# WHAT THIS SCRIPT DOES IN ORDER:
#   1. Sets all the configuration variables
#   2. Enables required GCP APIs (if not already enabled)
#   3. Creates a Service Account for Cloud Run to use
#   4. Grants the Service Account the right permissions
#   5. Stores all secrets in GCP Secret Manager
#   6. Builds the Docker image for AMD64 (required for Cloud Run)
#   7. Pushes the image to Google Container Registry
#   8. Deploys to Cloud Run with all environment variables
#   9. Tests the deployment with a health check
#
# HOW TO RUN:
#   chmod +x deployment/gcp/deploy.sh   (make it executable, first time only)
#   ./deployment/gcp/deploy.sh
#
# PREREQUISITES:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Docker installed and running
#   - .env file filled in with all real values
#   - Cloud SQL instance running (activation-policy=ALWAYS)
##############################################################################

set -e
# set -e means: EXIT IMMEDIATELY if any command fails.
# Without this, the script continues even after errors —
# leading to confusing failures much later in the script.
# With set -e, you see the exact command that failed.

set -o pipefail
# pipefail makes pipes (command1 | command2) fail if ANY command
# in the pipe fails — not just the last one.
# Without this: "false | true" succeeds (true is last, succeeds).
# With this:    "false | true" fails (false failed somewhere in pipe).

echo "============================================================"
echo "MOSAIC — Google Cloud Run Deployment"
echo "============================================================"


##############################################################################
# CONFIGURATION
# Set all variables here. Change these to match your GCP project.
##############################################################################

PROJECT_ID="mosaic-clinical-trials"
# Your GCP project ID — must match what you created in the console.

REGION="us-central1"
# GCP region — must match where your Cloud SQL instance is.
# Keeping everything in the same region avoids inter-region data fees.

SERVICE_NAME="mosaic-api"
# The name of your Cloud Run service.
# This appears in the GCP Console under Cloud Run.

IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
# The full Docker image path in Google Container Registry.
# Format: gcr.io/PROJECT_ID/IMAGE_NAME
# gcr.io is Google Container Registry — GCP's Docker image storage.

CLOUD_SQL_INSTANCE="${PROJECT_ID}:${REGION}:clinical-trial-db"
# The Cloud SQL connection name in the format:
# PROJECT_ID:REGION:INSTANCE_NAME
# Cloud Run uses this to connect via Unix socket — more secure
# than TCP with an IP address.

SERVICE_ACCOUNT_NAME="mosaic-sa"
# The service account that Cloud Run will run as.
# Service accounts are identities for applications (not humans).
# This account needs permissions to access Cloud SQL, GCS, and secrets.

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
# The full email address of the service account.
# GCP service accounts always follow this naming pattern.

echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE_NAME}"
echo ""


##############################################################################
# STEP 1: ENABLE REQUIRED GCP APIS
# These APIs must be enabled before we can use GCP services.
# Enabling them is idempotent — safe to run even if already enabled.
##############################################################################

echo "Step 1: Enabling required GCP APIs..."

gcloud services enable \
    run.googleapis.com \
    # Cloud Run — the serverless container platform we deploy to.
    cloudbuild.googleapis.com \
    # Cloud Build — used by GCP for container operations.
    containerregistry.googleapis.com \
    # Container Registry — where we store our Docker image.
    sqladmin.googleapis.com \
    # Cloud SQL Admin — manages our PostgreSQL instance.
    storage.googleapis.com \
    # Cloud Storage — where our study files are stored.
    secretmanager.googleapis.com \
    # Secret Manager — stores our API keys and passwords securely.
    --project=${PROJECT_ID}

echo "✓ APIs enabled"


##############################################################################
# STEP 2: CREATE SERVICE ACCOUNT
# Cloud Run needs an identity to make authenticated GCP API calls.
# Without a service account, Cloud Run cannot access Cloud SQL,
# GCS, or Secret Manager — permission denied on everything.
##############################################################################

echo ""
echo "Step 2: Creating service account..."

# Create the service account (|| true means: if it already exists, continue)
gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} \
    --display-name="MOSAIC API Service Account" \
    --project=${PROJECT_ID} \
    || true
# || true prevents set -e from stopping the script if the service account
# already exists (which happens on re-deployments).

echo "✓ Service account created: ${SERVICE_ACCOUNT_EMAIL}"


##############################################################################
# STEP 3: GRANT PERMISSIONS TO SERVICE ACCOUNT
# The service account needs specific permissions for each GCP service.
# We follow the principle of least privilege — grant only what is needed.
##############################################################################

echo ""
echo "Step 3: Granting service account permissions..."

# Permission to connect to Cloud SQL
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/cloudsql.client"
# cloudsql.client = can connect to Cloud SQL databases.
# Without this, Cloud Run gets "permission denied" when trying
# to connect to our PostgreSQL database.

# Permission to read/write Google Cloud Storage
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectAdmin"
# storage.objectAdmin = can read, write, and delete GCS objects.
# Our GCSStore class needs this to save and load study files.

# Permission to read secrets from Secret Manager
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"
# secretAccessor = can read secret values (but not list or create).
# Cloud Run reads our API keys from Secret Manager at startup.

echo "✓ Permissions granted"


##############################################################################
# STEP 4: STORE SECRETS IN SECRET MANAGER
# API keys and passwords should NEVER be in environment variables
# or Dockerfiles — they would be visible in logs and container metadata.
# Secret Manager stores them encrypted and only exposes them to
# authorised service accounts at runtime.
##############################################################################

echo ""
echo "Step 4: Storing secrets in Secret Manager..."

# Load values from local .env file
source .env 2>/dev/null || true
# source .env reads the .env file and sets all variables as shell variables.
# 2>/dev/null suppresses errors if .env does not exist.
# || true prevents failure if .env is missing.

# Function to create or update a secret
create_or_update_secret() {
    local SECRET_NAME=$1
    # $1 is the first argument passed to this function.
    local SECRET_VALUE=$2
    # $2 is the second argument — the actual secret value.

    if gcloud secrets describe ${SECRET_NAME} --project=${PROJECT_ID} &>/dev/null; then
        # gcloud secrets describe checks if the secret already exists.
        # &>/dev/null redirects all output to null — we only care about exit code.
        # Exit code 0 = secret exists. Non-zero = does not exist.

        # Secret exists — add a new version with the updated value
        echo "${SECRET_VALUE}" | gcloud secrets versions add ${SECRET_NAME} \
            --data-file=- \
            --project=${PROJECT_ID}
        # echo "${SECRET_VALUE}" pipes the value to gcloud.
        # --data-file=- means read from stdin (the piped value).
        # This creates a new version of the secret — the latest version
        # is what Cloud Run will read.
    else
        # Secret does not exist — create it with the initial value
        echo "${SECRET_VALUE}" | gcloud secrets create ${SECRET_NAME} \
            --data-file=- \
            --replication-policy=automatic \
            --project=${PROJECT_ID}
        # --replication-policy=automatic lets GCP decide where to store the secret.
        # This is the simplest option and works for most use cases.
    fi
}

# Store each secret
create_or_update_secret "openai-api-key"     "${OPENAI_API_KEY}"
create_or_update_secret "langsmith-api-key"  "${LANGSMITH_API_KEY}"
create_or_update_secret "db-password"        "${DB_PASSWORD}"

echo "✓ Secrets stored in Secret Manager"


##############################################################################
# STEP 5: BUILD THE DOCKER IMAGE
# Builds for linux/amd64 — REQUIRED for Cloud Run.
# Apple Silicon Macs are ARM64. Cloud Run is AMD64.
# Without --platform linux/amd64, Cloud Run cannot run the image.
##############################################################################

echo ""
echo "Step 5: Building Docker image for linux/amd64..."

docker buildx build \
    --platform linux/amd64 \
    # Forces AMD64 architecture — critical for Cloud Run on Apple Silicon.

    --tag ${IMAGE_NAME}:latest \
    # Tag the image with :latest — the version we are about to deploy.

    --file deployment/Dockerfile \
    # Path to our Dockerfile relative to the build context.

    --push \
    # --push builds AND pushes to the registry in one command.
    # Without --push, the image is built locally but not uploaded.
    # Cloud Run needs the image in the registry — local is not enough.

    .
    # The build context — the . means "use the current directory".
    # Docker sends all files in this directory to the build daemon.
    # .dockerignore controls which files are excluded.

echo "✓ Image built and pushed: ${IMAGE_NAME}:latest"


##############################################################################
# STEP 6: DEPLOY TO CLOUD RUN
# This is the main deployment command. It creates (or updates) the
# Cloud Run service with all our configuration.
##############################################################################

echo ""
echo "Step 6: Deploying to Cloud Run..."

gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    # Which Docker image to run — the one we just built and pushed.

    --platform=managed \
    # managed = Cloud Run (fully managed, serverless).
    # Alternative is "gke" (Google Kubernetes Engine) which requires
    # managing your own cluster. We use managed — simpler and cheaper.

    --region=${REGION} \
    # Which GCP region to deploy to.
    # Must match the region of your Cloud SQL instance.

    --service-account=${SERVICE_ACCOUNT_EMAIL} \
    # Which service account Cloud Run runs as.
    # This account has the permissions we granted in Step 3.

    --add-cloudsql-instances=${CLOUD_SQL_INSTANCE} \
    # Tells Cloud Run to open a secure connection to our Cloud SQL instance.
    # Cloud Run connects via a Unix socket at:
    # /cloudsql/PROJECT:REGION:INSTANCE
    # More secure than TCP — no IP address exposed, no firewall rules needed.

    --set-env-vars="\
GCP_PROJECT_ID=${PROJECT_ID},\
GCP_REGION=${REGION},\
GCS_BUCKET_NAME=${GCS_BUCKET_NAME},\
DB_HOST=/cloudsql/${CLOUD_SQL_INSTANCE},\
DB_PORT=5432,\
DB_NAME=${DB_NAME},\
DB_USER=${DB_USER},\
OPENAI_EMBEDDING_MODEL=${OPENAI_EMBEDDING_MODEL},\
OPENAI_CHAT_MODEL=${OPENAI_CHAT_MODEL},\
LANGSMITH_PROJECT=${LANGSMITH_PROJECT},\
LANGSMITH_TRACING_V2=${LANGSMITH_TRACING_V2},\
API_HOST=0.0.0.0,\
API_PORT=8000,\
API_ENV=production" \
    # Set non-sensitive environment variables directly.
    # These are not secret — they are just configuration.
    # DB_HOST uses the Unix socket path (not an IP) for Cloud Run.

    --set-secrets="\
OPENAI_API_KEY=openai-api-key:latest,\
LANGSMITH_API_KEY=langsmith-api-key:latest,\
DB_PASSWORD=db-password:latest" \
    # Inject secrets from Secret Manager as environment variables.
    # Format: ENV_VAR_NAME=SECRET_NAME:VERSION
    # :latest means use the most recent version of the secret.
    # The secret VALUES are never visible in logs or gcloud commands.

    --min-instances=0 \
    # Minimum running instances.
    # 0 = scales to zero when no requests come in.
    # This means ZERO cost when the API is idle.
    # Cloud Run starts a new instance automatically when a request arrives.

    --max-instances=3 \
    # Maximum simultaneous instances.
    # 3 is enough for our workload — each instance handles one request at a time.
    # Increasing this allows more parallel analysis runs.

    --memory=2Gi \
    # Memory per instance.
    # 2GB is required for:
    # - LangGraph graph in memory
    # - 6 simultaneous agent LLM calls
    # - pgvector embedding operations
    # Less than 2GB causes out-of-memory crashes mid-analysis.

    --cpu=2 \
    # CPU cores per instance.
    # 2 CPUs allows parallel processing within each instance.
    # The 6 agents run concurrently — more CPUs = faster execution.

    --timeout=300 \
    # Request timeout in seconds.
    # 300 seconds = 5 minutes maximum per request.
    # A full analysis run with 6 agents typically takes 15-60 seconds.
    # We set 300 as a generous safety buffer.

    --concurrency=1 \
    # How many requests one instance handles simultaneously.
    # 1 = one analysis run per instance at a time.
    # This prevents memory pressure from concurrent heavy workloads.
    # Cloud Run scales OUT (more instances) not UP (more concurrent) for us.

    --no-allow-unauthenticated \
    # Require authentication for all requests.
    # Without this, the API is publicly accessible — anyone on the internet
    # could trigger expensive analysis runs on our behalf.
    # Callers must include a Google identity token in the Authorization header.
    # Use --allow-unauthenticated for a public demo (personal GCP account).

    --project=${PROJECT_ID}

echo "✓ Deployed to Cloud Run"


##############################################################################
# STEP 7: GET THE SERVICE URL
##############################################################################

echo ""
echo "Step 7: Getting service URL..."

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")
# gcloud run services describe fetches the service details.
# --format="value(status.url)" extracts just the URL.
# $(...) captures the command output as a variable.

echo "✓ Service URL: ${SERVICE_URL}"


##############################################################################
# STEP 8: TEST THE DEPLOYMENT
##############################################################################

echo ""
echo "Step 8: Testing deployment with health check..."

sleep 10
# Wait 10 seconds for Cloud Run to fully initialise the container.
# The health check would fail immediately after deployment —
# the container needs a few seconds to start uvicorn and initialise
# the database connections.

HEALTH_RESPONSE=$(curl -s \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    "${SERVICE_URL}/api/v1/health")
# curl -s = silent mode (no progress bars)
# -H "Authorization: Bearer TOKEN" adds the auth header.
# $(gcloud auth print-identity-token) generates a fresh identity token
# for our authenticated account — Cloud Run accepts this token.

echo "Health check response:"
echo ${HEALTH_RESPONSE} | python3 -m json.tool
# python3 -m json.tool pretty-prints JSON with indentation.
# The health response should show:
# {
#   "status": "healthy",
#   "database": "connected",
#   ...
# }


##############################################################################
# DEPLOYMENT COMPLETE
##############################################################################

echo ""
echo "============================================================"
echo "MOSAIC deployment complete"
echo "============================================================"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test your deployment:"
echo "curl -s \\"
echo "  -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"task\": \"Find completed trials with missing results\"}' \\"
echo "  ${SERVICE_URL}/api/v1/analyze | python3 -m json.tool"
echo ""
echo "View logs in GCP Console:"
echo "https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/logs?project=${PROJECT_ID}"
echo ""
echo "Stop Cloud SQL when done to save cost:"
echo "gcloud sql instances patch clinical-trial-db --activation-policy=NEVER --project=${PROJECT_ID}"
echo "============================================================"