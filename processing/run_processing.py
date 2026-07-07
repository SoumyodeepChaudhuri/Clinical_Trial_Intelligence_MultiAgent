##############################################################################
# processing/run_processing.py
#
# PURPOSE:
#   This is the entry point for the PROCESSING pipeline.
#   It is the second major pipeline you run after run_ingestion.py.
#
# WHAT IT DOES IN ORDER:
#   1. Reads all cleaned study files from Google Cloud Storage
#   2. Saves each study's metadata into the Cloud SQL studies table
#      (required first because chunks have a foreign key to studies)
#   3. Splits each study's text into overlapping chunks (chunker.py)
#   4. Converts each chunk into 1536 numbers via OpenAI (embedder.py)
#   5. Saves chunks + embeddings into Cloud SQL chunks table (vector_store.py)
#
# WHY THIS ORDER MATTERS:
#   chunks table has: nct_id REFERENCES studies(nct_id)
#   This is a foreign key constraint — it means we CANNOT save a chunk
#   for a study that does not yet exist in the studies table.
#   So we ALWAYS save study metadata to Cloud SQL BEFORE saving chunks.
#   Getting this order wrong causes ForeignKeyViolationError on every insert.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   YES — this is the second file you run directly after run_ingestion.py
#
#   Make sure your .venv is active and .env is filled in, then:
#   python3 processing/run_processing.py
#
# WHAT OUTPUT TO EXPECT:
#   2024-03-15 14:32:11 | INFO | Starting MOSAIC processing pipeline
#   2024-03-15 14:32:11 | INFO | Found 150 studies in GCS
#   2024-03-15 14:32:15 | INFO | Loaded 150 studies successfully
#   2024-03-15 14:32:15 | INFO | Inserted 150 studies into database
#   2024-03-15 14:32:16 | INFO | Chunked all studies | total_chunks=312
#   2024-03-15 14:32:16 | INFO | Starting embedding | total_chunks=312
#   2024-03-15 14:32:18 | INFO | Embedding batch 1/7 | chunks_in_batch=50
#   2024-03-15 14:32:20 | INFO | Batch embedded successfully | chunks=50 | embedding_dims=1536
#   ... repeats for each batch ...
#   2024-03-15 14:32:45 | INFO | Chunks saved | saved=312 | skipped=0
#   2024-03-15 14:32:45 | INFO | Processing complete
#   2024-03-15 14:32:45 | INFO | Studies processed : 150
#   2024-03-15 14:32:45 | INFO | Chunks created    : 312
#   2024-03-15 14:32:45 | INFO | Chunks embedded   : 312
#   2024-03-15 14:32:45 | INFO | Chunks stored     : 312
##############################################################################

import asyncio 
from ingestion.document_parser import ParsedStudy
from ingestion.gcs_store import GCSStore
from processing.chunker import Chunker
from processing.embedder import Embedder
from processing.vector_store import VectorStore

from config.logging_config import setup_logging
logger = setup_logging(__name__)


async def run_processing():
    """
    Runs the complete processing pipeline from start to finish.

    Reads cleaned studies from GCS, chunks them, embeds them via
    OpenAI, and stores everything in Cloud SQL for agent search.
    """
    gcs_store = GCSStore()
    chunker = Chunker()
    embedder = Embedder()
    logger.info("Loading parsed studies from GCS...")
    nct_ids = await gcs_store.list_processed_studies()
    # Ask GCS for a list of every NCT ID we saved during ingestion.
    # Returns something like: ["NCT04788680", "NCT02208921", ...]
    logger.info(f"Found {len(nct_ids)} studies in GCS")
    
    studies : list[ParsedStudy] = []
    for nct_id in nct_ids:
        study = await gcs_store.load_parsed_study(nct_id)
        # Load the full ParsedStudy object for each NCT ID.
        # This reads the processed/studies/NCT*.json file from GCS
        # and rebuilds it as a Pydantic ParsedStudy object.
        if study:
            studies.append(study)
        logger.info(f"Loaded {len(studies)} studies successfully")
        
    # ── OPEN THE DATABASE CONNECTION ───────────────────────────
    async with VectorStore() as vector_store:
        # "async with" opens the connection pool when we enter
        # and GUARANTEES it closes when we exit — even if an
        # error occurs halfway through. This is the correct way
        # to use VectorStore — never open it manually.

        # ── STEP 2: SAVE STUDY METADATA TO CLOUD SQL ──────────
        # IMPORTANT: We MUST do this BEFORE saving chunks.
        # The chunks table has a foreign key to the studies table.
        # If we try to save chunks first, every insert will fail
        # with ForeignKeyViolationError because the studies do
        # not exist in the database yet.
        logger.info("Saving study metadata to Cloud SQL...")

        studies_saved = 0
        for study in studies:
            success = await vector_store.save_study(
                study_data={
                    "nct_id":             study.nct_id,
                    "title":              study.title,
                    "sponsor":            study.sponsor,
                    "phase":              study.phase,
                    "status":             study.status,
                    "conditions":         study.conditions,
                    "interventions":      study.interventions,
                    "primary_outcome":    study.primary_outcome,
                    "secondary_outcomes": study.secondary_outcomes,
                    "start_date":         study.start_date,
                    "completion_date":    study.completion_date,
                    "results_posted":     study.results_posted,
                    "enrollment":         study.enrollment,
                    "gcs_path":           f"processed/studies/{study.nct_id}.json",
                }
                # We pass a plain dictionary — save_study() accepts
                # dict[str, Any] not a ParsedStudy object directly.
                # We convert it here so vector_store stays decoupled
                # from the ingestion layer's data models.
            )
            if success:
                studies_saved += 1

        logger.info(
            f"Studies saved to Cloud SQL | "
            f"saved={studies_saved} | "
            f"total={len(studies)}"
        )

        # ── STEP 3: CHUNK ALL STUDIES ──────────────────────────
        logger.info("Chunking studies...")

        all_chunks = chunker.chunk_studies(studies)
        # chunk_studies() processes the entire list at once and
        # returns one flat list of TextChunk objects.
        # Each study typically produces 2-4 chunks depending on
        # how much text its fields contain.

        logger.info(f"Total chunks created: {len(all_chunks)}")

        # ── STEP 4: EMBED ALL CHUNKS ───────────────────────────
        logger.info("Embedding chunks via OpenAI...")

        embedded_chunks = await embedder.embed_chunks(all_chunks)
        # This is the step that costs OpenAI API credit.
        # embed_chunks() sends chunks in batches of 50 to OpenAI.
        # Each chunk comes back with 1536 numbers attached.
        # Total cost for 300 chunks ≈ $0.001 — negligible.

        logger.info(f"Total chunks embedded: {len(embedded_chunks)}")

        # ── STEP 5: SAVE CHUNKS TO CLOUD SQL ──────────────────
        logger.info("Saving embedded chunks to Cloud SQL...")

        chunks_stored = await vector_store.save_embedded_chunks(embedded_chunks)
        # save_embedded_chunks() inserts every EmbeddedChunk into the
        # chunks table — text, metadata, AND the 1536-number vector.
        # Uses ON CONFLICT DO NOTHING so re-runs do not duplicate data.

        # ── FINAL SUMMARY ──────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Processing complete")
        logger.info(f"Studies processed : {len(studies)}")
        logger.info(f"Chunks created    : {len(all_chunks)}")
        logger.info(f"Chunks embedded   : {len(embedded_chunks)}")
        logger.info(f"Chunks stored     : {chunks_stored}")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_processing())
    # asyncio.run() starts the async event loop and runs
    # run_processing() inside it — same pattern as run_ingestion.py.
    # This is the ignition key for the entire processing pipeline.