##############################################################################
# processing/vector_store.py
#
# PURPOSE:
#   This file saves EmbeddedChunks into Cloud SQL's chunks table
#   and provides semantic search — finding chunks by MEANING
#   rather than by keyword.
#
# THIS IS THE HEART OF THE ENTIRE SYSTEM.
#   Every agent query flows through this file.
#   When the Missing Results agent asks:
#   "find studies where sponsor never posted results"
#   — this file is what finds the answer.
#
# HOW SEMANTIC SEARCH WORKS HERE — STEP BY STEP:
#   1. The agent's question gets converted to 1536 numbers (embedder.py)
#   2. This file sends those 1536 numbers to Cloud SQL
#   3. pgvector compares them against every chunk's 1536 numbers
#   4. The chunks whose numbers are CLOSEST get returned
#   5. "Closest" is measured using cosine similarity — the <=> operator
#
# WHAT IS COSINE SIMILARITY?
#   Imagine two arrows pointing in different directions.
#   Cosine similarity measures the ANGLE between those arrows.
#   A small angle = similar meaning = small cosine distance.
#   A large angle = different meaning = large cosine distance.
#   The <=> operator in pgvector does this calculation for us.
#
# WHY asyncpg AND NOT SQLAlchemy?
#   SQLAlchemy is great for standard queries but pgvector's
#   VECTOR type is not natively supported by SQLAlchemy's ORM.
#   asyncpg is a raw async PostgreSQL driver — it gives us
#   complete control over the SQL we write, which means we
#   can use pgvector's custom operators (<=> for cosine distance)
#   without any compatibility issues.
#
# THE CODEC — THE MOST IMPORTANT TECHNICAL DETAIL:
#   asyncpg does not know what a VECTOR type is by default.
#   PostgreSQL knows, but asyncpg needs to be taught how to
#   convert between Python lists and PostgreSQL VECTOR columns.
#   We register a custom codec — a translator — that handles this.
#   Without it, every insert and select would crash with a type error.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. This defines a class used by run_processing.py and the agents.
#
# HOW OTHER FILES USE THIS:
#   from processing.vector_store import VectorStore
#
#   vs = VectorStore()
#   await vs.init()
#   await vs.save_embedded_chunks(list_of_embedded_chunks)
#   results = await vs.search(query_embedding, top_k=5)
##############################################################################


import asyncio
import asyncpg
# asyncpg is the fastest async PostgreSQL driver for Python.
# It talks directly to PostgreSQL without any ORM in between.
# This gives us full control over our SQL queries — important
# because pgvector uses custom operators that ORMs do not support.

import json
# We use json to serialize the embedding list into a format
# that PostgreSQL can accept as a VECTOR type.

from typing import Any
from processing.embedder import EmbeddedChunk
# EmbeddedChunk is our input — chunks with embeddings attached.

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "processing.vector_store"


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

POOL_MIN_SIZE = 2
# Minimum number of database connections to keep open at all times.
# Think of a connection pool like a team of workers —
# we always keep at least 2 ready to handle requests immediately.

POOL_MAX_SIZE = 10
# Maximum number of connections allowed at the same time.
# More connections = more parallelism but more DB memory usage.
# 10 is a safe ceiling for our db-f1-micro Cloud SQL instance.

TOP_K_DEFAULT = 5
# Default number of similar chunks to return per search query.
# When an agent searches, it gets back the 5 most relevant chunks
# by default. Agents can override this if they need more.


# ─────────────────────────────────────────────────────────────
# THE VECTOR STORE CLASS
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Saves EmbeddedChunks to Cloud SQL and enables semantic search
    over them using pgvector's cosine similarity operator.

    LIFECYCLE — always follow this order:
        vs = VectorStore()   # create
        await vs.init()      # connect to database
        # ... use it ...
        await vs.close()     # disconnect cleanly

    OR use it as an async context manager:
        async with VectorStore() as vs:
            await vs.save_embedded_chunks(chunks)
            results = await vs.search(query_embedding)
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        # The connection pool — our team of database workers.
        # Starts as None — created when init() is called.
        # We never create connections one by one — always use
        # the pool so connections are reused efficiently.

    # ── ASYNC CONTEXT MANAGER SUPPORT ─────────────────────────

    async def __aenter__(self) -> "VectorStore":
        """Allows using VectorStore with 'async with' pattern."""
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection pool when exiting 'async with'."""
        await self.close()

    # ── INITIALISE THE CONNECTION POOL ────────────────────────

    async def init(self) -> None:
        """
        Creates the asyncpg connection pool and registers the
        pgvector codec so Python can read and write VECTOR columns.

        MUST be called before any other method.
        This is where we actually connect to Cloud SQL.
        """

        logger.info(
            f"Connecting to Cloud SQL | "
            f"host={settings.db_host} | "
            f"database={settings.db_name}"
        )

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            # The Cloud SQL IP address from .env.
            # Local: "34.133.55.17" (public IP)
            # Cloud Run: "/cloudsql/project:region:instance" (Unix socket)

            port=int(settings.db_port),
            # Port 5432 — standard PostgreSQL port.
            # We cast to int because settings.db_port is stored as str.

            database=settings.db_name,
            # "clinical_trial_db" — the database we created.

            user=settings.db_user,
            # "mosaic_user" — the user we created.

            password=settings.db_password,
            # The password from .env — never hardcoded.

            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            # Keep between 2 and 10 connections in the pool.

            init=self._init_connection,
            # This is the KEY parameter — explained in detail below.
            # init= means: "run this function on EVERY new connection
            # the pool creates." We use it to register our pgvector codec.
        )

        logger.info("Connection pool created successfully")

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """
        Runs automatically on every new database connection.

        This is where we register the pgvector codec —
        the translator that teaches asyncpg how to convert
        between Python lists and PostgreSQL VECTOR columns.

        WITHOUT THIS:
            Saving a chunk → TypeError: cannot convert list to VECTOR
            Reading a chunk → asyncpg.exceptions.UndefinedTypeError

        WITH THIS:
            Python [0.023, -0.041, 0.891, ...] ↔ PostgreSQL VECTOR(1536)
            The conversion happens automatically, invisibly.

        Args:
            conn: A fresh asyncpg connection, just created by the pool.
        """

        await conn.set_type_codec(
            "vector",
            # The PostgreSQL type name we are teaching asyncpg about.
            # This matches the VECTOR column type in our chunks table.

            encoder=lambda v: json.dumps(v),
            # ENCODER: Python → PostgreSQL
            # When we SAVE a chunk, asyncpg needs to convert our
            # Python list [0.023, -0.041, ...] into something
            # PostgreSQL understands.
            # json.dumps([0.023, -0.041]) → "[0.023, -0.041]"
            # PostgreSQL's pgvector accepts this JSON string format.

            decoder=lambda v: json.loads(v),
            # DECODER: PostgreSQL → Python
            # When we READ a chunk back, pgvector returns the
            # vector as a string "[0.023, -0.041, ...]"
            # json.loads converts it back to a Python list.

            schema="public",
            # Which PostgreSQL schema contains the vector type.
            # "public" is the default schema — where our tables live.

            format="text",
            # Use text format for the conversion.
            # "text" means the encoder/decoder work with strings.
            # The alternative is "binary" (raw bytes) — text is
            # simpler and perfectly fast for our use case.
        )

    # ── CLOSE THE CONNECTION POOL ──────────────────────────────

    async def close(self) -> None:
        """
        Gracefully closes all database connections in the pool.
        Always call this when you are done with the VectorStore.
        Leaving connections open wastes Cloud SQL resources.
        """

        if self._pool:
            await self._pool.close()
            logger.info("Connection pool closed")

    # ── SAVE EMBEDDED CHUNKS TO CLOUD SQL ─────────────────────

    async def save_embedded_chunks(
        self,
        chunks: list[EmbeddedChunk],
        # The list of EmbeddedChunk objects from embedder.py.
        # Each one has text + metadata + 1536-number embedding.

    ) -> int:
        """
        Saves a list of EmbeddedChunks into the chunks table.

        Uses INSERT ... ON CONFLICT DO NOTHING so it is safe
        to run multiple times — duplicate chunks are silently
        skipped instead of causing an error.

        Args:
            chunks: List of EmbeddedChunk objects to save.

        Returns:
            Number of chunks successfully saved.
        """

        if not chunks:
            logger.warning("save_embedded_chunks called with empty list")
            return 0

        saved_count = 0

        async with self._pool.acquire() as conn:
            # acquire() checks out one connection from the pool.
            # When this block exits, the connection is returned
            # to the pool automatically — not closed, just returned.
            # This is efficient — the next save call reuses it.

            for chunk in chunks:
                try:
                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (nct_id, chunk_text, embedding, chunk_index, source)
                        VALUES
                            ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                        """,
                        # $1, $2, $3, $4, $5 are parameter placeholders.
                        # asyncpg fills them in from the arguments below.
                        # This is called a parameterised query —
                        # it prevents SQL injection attacks because
                        # the values are never inserted directly into
                        # the SQL string, they are passed separately.
                        #
                        # ON CONFLICT DO NOTHING means:
                        # if a chunk with this chunk_id already exists,
                        # skip it silently instead of raising an error.
                        # This makes the entire processing step idempotent
                        # — safe to run again without duplicating data.

                        chunk.nct_id,        # $1
                        chunk.chunk_text,    # $2
                        chunk.embedding,     # $3 — our pgvector codec
                        #                          converts this list
                        #                          to VECTOR automatically
                        chunk.chunk_index,   # $4
                        chunk.source,        # $5
                    )

                    saved_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to save chunk | "
                        f"chunk_id={chunk.chunk_id} | "
                        f"error={e}"
                    )
                    # Log and continue — one failed chunk does not
                    # stop the other 300 from being saved.

        logger.info(
            f"Chunks saved | "
            f"saved={saved_count} | "
            f"total_input={len(chunks)} | "
            f"skipped={len(chunks) - saved_count}"
        )

        return saved_count

    # ── SAVE ONE STUDY RECORD TO THE STUDIES TABLE ────────────

    async def save_study(
        self,
        study_data: dict[str, Any],
        # A dictionary containing all the study fields to save.
        # This comes from ParsedStudy.model_dump() — converting
        # the Pydantic object into a plain Python dict.

    ) -> None:
        """
        Saves one study record into the studies table.

        Uses INSERT ... ON CONFLICT (nct_id) DO UPDATE so that
        if a study already exists, its fields get refreshed
        with the latest data instead of being skipped.

        Args:
            study_data: Dictionary of study fields to save.
        """

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO studies
                    (nct_id, title, sponsor, phase, status,
                     conditions, interventions, primary_outcome,
                     secondary_outcomes, start_date, completion_date,
                     results_posted, enrollment, gcs_path)
                VALUES
                    ($1, $2, $3, $4, $5,
                     $6, $7, $8,
                     $9, $10, $11,
                     $12, $13, $14)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title            = EXCLUDED.title,
                    sponsor          = EXCLUDED.sponsor,
                    phase            = EXCLUDED.phase,
                    status           = EXCLUDED.status,
                    conditions       = EXCLUDED.conditions,
                    interventions    = EXCLUDED.interventions,
                    primary_outcome  = EXCLUDED.primary_outcome,
                    secondary_outcomes = EXCLUDED.secondary_outcomes,
                    start_date       = EXCLUDED.start_date,
                    completion_date  = EXCLUDED.completion_date,
                    results_posted   = EXCLUDED.results_posted,
                    enrollment       = EXCLUDED.enrollment,
                    gcs_path         = EXCLUDED.gcs_path
                """,
                # ON CONFLICT (nct_id) DO UPDATE SET means:
                # if this nct_id already exists in the table,
                # UPDATE all its fields with the new values.
                # EXCLUDED.field refers to the value we tried to INSERT.
                # This pattern is called "upsert" —
                # UPDATE if exists, INSERT if new.

                study_data.get("nct_id"),              # $1
                study_data.get("title"),               # $2
                study_data.get("sponsor"),             # $3
                study_data.get("phase"),               # $4
                study_data.get("status"),              # $5
                study_data.get("conditions", []),      # $6
                study_data.get("interventions", []),   # $7
                study_data.get("primary_outcome"),     # $8
                study_data.get("secondary_outcomes", []),  # $9
                study_data.get("start_date"),          # $10
                study_data.get("completion_date"),     # $11
                study_data.get("results_posted"),      # $12
                study_data.get("enrollment"),          # $13
                study_data.get("gcs_path"),            # $14
            )

    # ── SEMANTIC SEARCH — THE CORE INTELLIGENCE OPERATION ─────

    async def search(
        self,
        query_embedding: list[float],
        # The search query converted to 1536 numbers.
        # The caller (an agent) creates this by embedding its
        # question text using the same embedding model.

        top_k: int = TOP_K_DEFAULT,
        # How many results to return.
        # Default 5 — agents get the 5 most relevant chunks.

        source_filter: str | None = None,
        # Optional filter — "study", "paper", or None (both).
        # Lets agents search only study chunks or only paper chunks.

        nct_id_filter: str | None = None,
        # Optional filter — search only chunks from a specific study.
        # Useful when an agent is analysing one specific trial.

    ) -> list[dict[str, Any]]:
        """
        Finds the most semantically similar chunks to a query embedding.

        Uses pgvector's cosine distance operator (<=>)  to compare
        the query embedding against every stored chunk embedding
        and returns the TOP_K closest ones.

        This is the method every agent calls when it needs context.
        It is the bridge between a natural language question and
        the relevant chunks stored in Cloud SQL.

        Args:
            query_embedding:  The search query as 1536 numbers.
            top_k:            How many results to return.
            source_filter:    Optional filter by source type.
            nct_id_filter:    Optional filter by specific study.

        Returns:
            List of dictionaries, each containing:
            - nct_id:      Which study this chunk belongs to
            - chunk_text:  The actual text content
            - chunk_index: Position in the original document
            - source:      "study" or "paper"
            - distance:    Cosine distance (lower = more similar)
                           0.0 = identical meaning
                           1.0 = completely different meaning
                           2.0 = opposite meaning
        """

        # ── BUILD THE QUERY DYNAMICALLY ────────────────────────
        # The base query always uses cosine distance (<=>).
        # We add WHERE clauses only if filters were provided.

        conditions = []
        # List of SQL WHERE conditions to add if filters are set.

        params: list[Any] = [query_embedding]
        # Start with the query embedding as $1.
        # Additional parameters ($2, $3) are added as filters are added.

        param_count = 1
        # Tracks our parameter numbering ($1, $2, $3...).
        # We increment this each time we add a filter parameter.

        if source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(source_filter)
            # Example: source = $2 with params = [embedding, "study"]

        if nct_id_filter:
            param_count += 1
            conditions.append(f"nct_id = ${param_count}")
            params.append(nct_id_filter)
            # Example: nct_id = $3 with params = [embedding, "study", "NCT123"]

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
            # Joins all conditions with AND:
            # "WHERE source = $2 AND nct_id = $3"

        param_count += 1
        params.append(top_k)
        # top_k is always the last parameter.

        query = f"""
            SELECT
                nct_id,
                chunk_text,
                chunk_index,
                source,
                embedding <=> $1 AS distance
            FROM chunks
            {where_clause}
            ORDER BY distance ASC
            LIMIT ${param_count}
        """
        # This SQL query is the heart of semantic search.
        #
        # embedding <=> $1
        #   The <=> operator is pgvector's cosine distance.
        #   It compares each stored embedding against our query embedding.
        #   Returns a number between 0 and 2.
        #   0 = identical, 1 = orthogonal (unrelated), 2 = opposite.
        #
        # ORDER BY distance ASC
        #   Sort by distance, smallest first.
        #   Smallest distance = most similar meaning = most relevant.
        #
        # LIMIT $N
        #   Return only the top N results.
        #   We do not want all 300 chunks — just the most relevant ones.

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            # conn.fetch() runs the query and returns ALL matching rows.
            # *params unpacks our list into separate arguments:
            # [embedding, "study", 5] → $1=embedding, $2="study", $3=5

        results = [dict(row) for row in rows]
        # Convert each asyncpg Record object into a plain Python dict.
        # asyncpg returns its own Record type — dicts are easier
        # for the rest of our code to work with.

        logger.info(
            f"Semantic search complete | "
            f"results_found={len(results)} | "
            f"top_k={top_k} | "
            f"source_filter={source_filter} | "
            f"nct_id_filter={nct_id_filter}"
        )

        return results

    # ── CHECK HOW MANY CHUNKS ARE STORED ──────────────────────

    async def get_chunk_count(self) -> int:
        """
        Returns the total number of chunks currently in the database.
        Used by run_processing.py to report progress after saving.

        Returns:
            Total count of rows in the chunks table.
        """

        async with self._pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            # fetchval() returns a single value — perfect for COUNT(*).
            # No need to navigate rows and columns like with fetch().

        logger.info(f"Total chunks in database: {result}")
        return result

    # ── CHECK IF A STUDY HAS ALREADY BEEN PROCESSED ───────────

    async def study_exists(self, nct_id: str) -> bool:
        """
        Checks if a study already has chunks saved in the database.

        Used by run_processing.py to skip studies that were already
        processed in a previous run — avoids duplicate work.

        Args:
            nct_id: The study to check.

        Returns:
            True if this study already has chunks in the database.
            False if it has not been processed yet.
        """

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE nct_id = $1",
                nct_id,
            )

        exists = count > 0
        # If count is 0 → no chunks → study not processed yet.
        # If count > 0 → chunks exist → study already processed.

        return exists