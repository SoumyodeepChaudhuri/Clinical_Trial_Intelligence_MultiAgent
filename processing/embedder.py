##############################################################################
# processing/embedder.py
#
# PURPOSE:
#   This file takes every TextChunk produced by chunker.py and
#   converts it into a vector embedding — a list of 1536 numbers
#   that mathematically represent the MEANING of that chunk.
#
# WHAT IS A VECTOR EMBEDDING — EXPLAINED SIMPLY:
#   Imagine plotting every word in the English language on a giant
#   map. Words that mean similar things end up close together on
#   that map. "Heart attack" and "myocardial infarction" would be
#   right next to each other even though the words look completely
#   different. "Results never posted" and "no outcomes published"
#   would also be very close together.
#
#   A vector embedding is exactly this — a position on that giant
#   map, expressed as 1536 numbers (coordinates).
#
#   WHY DOES THIS MATTER FOR MOSAIC?
#   When our Missing Results agent asks "find studies where sponsor
#   never posted results", we convert that QUESTION into 1536 numbers
#   too. Then pgvector finds the chunks whose numbers are closest
#   to the question's numbers. That is semantic search —
#   finding by MEANING, not by keyword matching.
#
# WHY text-embedding-3-small AND NOT text-embedding-3-large?
#   text-embedding-3-large produces 3072 dimensions but pgvector's
#   hnsw index has a 2000-dimension hard limit. We discovered this
#   during our build — the index creation failed with a clear error.
#   text-embedding-3-small produces 1536 dimensions — well within
#   the limit, cheaper, faster, and more than sufficient quality
#   for clinical trial signal detection.
#   This is a real production constraint we hit and solved.
#
# BATCHING — WHY WE DON'T EMBED ONE CHUNK AT A TIME:
#   If we sent one API call per chunk, 300 chunks = 300 API calls.
#   That is slow and expensive. OpenAI accepts up to 100 chunks
#   in a single API call. We batch 50 at a time — fast, efficient,
#   and within OpenAI's rate limits comfortably.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. This file defines a class used by run_processing.py.
#
# HOW OTHER FILES USE THIS:
#   from processing.embedder import Embedder
#
#   embedder = Embedder()
#   embedded_chunks = await embedder.embed_chunks(list_of_text_chunks)
##############################################################################


import asyncio
# asyncio for async/await — OpenAI calls are network requests
# and we want our program to stay responsive while waiting.

from dataclasses import dataclass
# dataclass for EmbeddedChunk — same pattern as TextChunk
# in chunker.py. A lightweight container for our output data.

from openai import AsyncOpenAI
# AsyncOpenAI is OpenAI's official async Python client.
# The "Async" version lets us use await — meaning our program
# does not freeze while waiting for OpenAI to respond.
# Regular OpenAI() client is synchronous and would block everything.

from processing.chunker import TextChunk
# We receive TextChunk objects as INPUT and return
# EmbeddedChunk objects as OUTPUT. TextChunk is defined
# in chunker.py — we import it here so Python knows its shape.

from config.settings import settings
# Gives us settings.openai_api_key and settings.openai_embedding_model
# from our .env file. Never hardcode API keys — always read from config.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "processing.embedder"


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

BATCH_SIZE = 50
# How many chunks to send to OpenAI in one API call.
# OpenAI accepts up to 100 inputs per call.
# We use 50 as a safe, comfortable batch size that:
#   - Stays well within OpenAI's limits
#   - Processes 300 chunks in just 6 API calls instead of 300
#   - Gives us a natural retry unit if one batch fails

RETRY_ATTEMPTS = 3
# How many times to retry a failed embedding API call.
# OpenAI occasionally returns 429 (rate limit) or 500 (server error).
# Retrying automatically handles these transient failures.

RETRY_SLEEP_SECONDS = 2
# How long to wait between retry attempts.
# 2 seconds gives OpenAI time to recover from a rate limit hit
# before we try again.


# ─────────────────────────────────────────────────────────────
# THE EmbeddedChunk DATACLASS
#
# This is the OUTPUT of the embedder — a TextChunk that now
# has its vector embedding attached to it.
# Think of it as a TextChunk with one extra field: embedding.
# The vector_store.py file receives EmbeddedChunks and saves
# them to Cloud SQL's chunks table.
# ─────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    """
    A TextChunk that has been enriched with its vector embedding.

    This is what gets saved to the Cloud SQL chunks table.
    Every field from TextChunk is carried over, plus one new field:
    embedding — the list of 1536 numbers representing this chunk's meaning.

    Fields:
        chunk_id:    Unique identifier. Example: "NCT04788680_chunk_0"
        nct_id:      Which study or paper this chunk belongs to.
        chunk_text:  The actual text content.
        chunk_index: Position in the original document (0, 1, 2...)
        source:      "study" or "paper"
        word_count:  Number of words in this chunk.
        embedding:   1536 floating point numbers from OpenAI.
                     This is the mathematical representation of meaning.
                     Two chunks that mean similar things will have
                     embeddings that are numerically close to each other.
    """

    chunk_id:    str
    nct_id:      str
    chunk_text:  str
    chunk_index: int
    source:      str
    word_count:  int
    embedding:   list[float]
    # list[float] means: a list of floating point numbers.
    # text-embedding-3-small always produces exactly 1536 of them.
    # Example (first 3 of 1536): [0.023, -0.041, 0.891, ...]
    # These numbers have no human-readable meaning on their own —
    # their meaning only emerges when you COMPARE them to other
    # embeddings using cosine similarity.


# ─────────────────────────────────────────────────────────────
# THE EMBEDDER CLASS
# ─────────────────────────────────────────────────────────────

class Embedder:
    """
    Converts TextChunks into EmbeddedChunks using OpenAI's
    text-embedding-3-small model.

    Processes chunks in batches of BATCH_SIZE for efficiency.
    Automatically retries failed API calls up to RETRY_ATTEMPTS times.

    Usage:
        embedder = Embedder()
        embedded = await embedder.embed_chunks(list_of_text_chunks)
    """

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        # Create the async OpenAI client using our API key from .env.
        # This client is reused for every API call in this session —
        # more efficient than creating a new client per batch.

        self._model = settings.openai_embedding_model
        # The embedding model name from .env.
        # Value: "text-embedding-3-small"
        # Stored as an instance variable so it is easy to see
        # and change in one place if needed.

        logger.info(
            f"Embedder initialised | model={self._model}"
        )

    # ── CORE METHOD: EMBED A LIST OF CHUNKS ───────────────────

    async def embed_chunks(
        self,
        chunks: list[TextChunk],
        # The list of TextChunk objects to embed.
        # These come from chunker.py — already split and labelled.

    ) -> list[EmbeddedChunk]:
        """
        Converts a list of TextChunks into EmbeddedChunks.

        Splits the input into batches of BATCH_SIZE and processes
        each batch with one OpenAI API call. Much more efficient
        than one API call per chunk.

        Args:
            chunks: List of TextChunk objects from chunker.py

        Returns:
            List of EmbeddedChunk objects — same chunks but now
            each has a 1536-number embedding attached.
            Any chunk that fails to embed is skipped — not fatal.
        """

        if not chunks:
            # Nothing to embed — return empty list immediately.
            logger.warning("embed_chunks called with empty list — nothing to do")
            return []

        logger.info(
            f"Starting embedding | "
            f"total_chunks={len(chunks)} | "
            f"batch_size={BATCH_SIZE} | "
            f"model={self._model}"
        )

        all_embedded: list[EmbeddedChunk] = []
        # Accumulates results across all batches.

        batches = self._create_batches(chunks)
        # Split the full list into smaller batches.
        # Example: 150 chunks → 3 batches of 50.

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Embedding batch {batch_num + 1}/{len(batches)} | "
                f"chunks_in_batch={len(batch)}"
            )

            embedded_batch = await self._embed_batch_with_retry(
                batch=batch,
                batch_num=batch_num,
            )
            # Embed this batch with automatic retry on failure.

            all_embedded.extend(embedded_batch)
            # Add this batch's results to our running total.

            if batch_num < len(batches) - 1:
                await asyncio.sleep(0.5)
                # Small pause between batches.
                # 0.5 seconds is enough to avoid hitting OpenAI's
                # rate limits while not slowing us down much.
                # We do NOT sleep after the last batch — no point.

        logger.info(
            f"Embedding complete | "
            f"total_embedded={len(all_embedded)} | "
            f"total_input={len(chunks)} | "
            f"skipped={len(chunks) - len(all_embedded)}"
        )
        # Log the final counts — including how many were skipped
        # if any batches failed after all retries.

        return all_embedded

    # ── PRIVATE METHOD: CREATE BATCHES ────────────────────────

    def _create_batches(
        self,
        chunks: list[TextChunk],
    ) -> list[list[TextChunk]]:
        """
        Splits a flat list of chunks into smaller batches.

        Example:
            150 chunks with BATCH_SIZE=50 →
            [[chunk_0..chunk_49], [chunk_50..chunk_99], [chunk_100..chunk_149]]

        Args:
            chunks: The full list of chunks to split.

        Returns:
            A list of lists — each inner list is one batch.
        """

        return [
            chunks[i : i + BATCH_SIZE]
            for i in range(0, len(chunks), BATCH_SIZE)
        ]
        # List comprehension that creates batches.
        # range(0, 150, 50) generates: 0, 50, 100
        # chunks[0:50]   → first batch
        # chunks[50:100] → second batch
        # chunks[100:150] → third batch

    # ── PRIVATE METHOD: EMBED ONE BATCH WITH RETRY ────────────

    async def _embed_batch_with_retry(
        self,
        batch: list[TextChunk],
        batch_num: int,
    ) -> list[EmbeddedChunk]:
        """
        Embeds one batch of chunks, retrying on failure.

        Attempts the embedding up to RETRY_ATTEMPTS times.
        Waits RETRY_SLEEP_SECONDS between each attempt.
        Returns an empty list if all attempts fail — the pipeline
        continues with the remaining batches rather than crashing.

        Args:
            batch:     One batch of TextChunks to embed.
            batch_num: The batch number (for logging only).

        Returns:
            List of EmbeddedChunks for this batch.
            Empty list if all retry attempts failed.
        """

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            # attempt goes: 1, 2, 3
            # RETRY_ATTEMPTS + 1 because range() is exclusive at the top.

            try:
                return await self._embed_batch(batch=batch)
                # Try to embed the batch.
                # If it succeeds, return immediately — no more retries needed.

            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    # We still have retries left — log a warning and wait.
                    logger.warning(
                        f"Embedding failed | "
                        f"batch={batch_num + 1} | "
                        f"attempt={attempt}/{RETRY_ATTEMPTS} | "
                        f"error={e} | "
                        f"retrying in {RETRY_SLEEP_SECONDS}s..."
                    )
                    await asyncio.sleep(RETRY_SLEEP_SECONDS)
                    # Wait before retrying.
                    # asyncio.sleep() is non-blocking — other async
                    # tasks can run during this pause.

                else:
                    # This was the LAST attempt and it still failed.
                    # Log an error and give up on this batch.
                    logger.error(
                        f"Embedding failed after {RETRY_ATTEMPTS} attempts | "
                        f"batch={batch_num + 1} | "
                        f"error={e} | "
                        f"skipping this batch"
                    )
                    return []
                    # Return empty list — the pipeline continues.
                    # Losing one batch of chunks is not fatal —
                    # the other 150 chunks are still processed.

        return []
        # Safety fallback — this line should never be reached
        # because the loop above always returns inside the loop.
        # Python requires a return outside the loop to be safe.

    # ── PRIVATE METHOD: EMBED ONE BATCH ───────────────────────

    async def _embed_batch(
        self,
        batch: list[TextChunk],
    ) -> list[EmbeddedChunk]:
        """
        Makes ONE OpenAI API call to embed an entire batch of chunks.

        This is the method that actually talks to OpenAI.
        It sends up to BATCH_SIZE chunk texts in one request
        and gets back one embedding per chunk.

        Args:
            batch: One batch of TextChunks (up to BATCH_SIZE).

        Returns:
            List of EmbeddedChunks with embeddings attached.

        Raises:
            Exception: If the OpenAI API call fails.
                       The caller (_embed_batch_with_retry) handles this.
        """

        texts = [chunk.chunk_text for chunk in batch]
        # Extract just the text from each chunk.
        # OpenAI only needs the text — not the metadata.
        # We send: ["Study NCT04788680 TITLE: ...", "SPONSOR: ...", ...]

        response = await self._client.embeddings.create(
            model=self._model,
            # Which embedding model to use.
            # Value: "text-embedding-3-small"
            # This produces 1536-dimensional embeddings.

            input=texts,
            # The list of texts to embed.
            # OpenAI processes ALL of them in one shot
            # and returns one embedding per text in the same order.
        )
        # await means: pause here until OpenAI responds.
        # The response contains one embedding object per input text.
        # response.data is a list of embedding objects.
        # response.data[0].embedding is the first list of 1536 numbers.
        # response.data[1].embedding is the second list of 1536 numbers.
        # And so on — one per chunk in our batch.

        embedded_chunks: list[EmbeddedChunk] = []

        for i, chunk in enumerate(batch):
            # Loop through each chunk alongside its index.
            # i = 0, 1, 2, ... matches the order of response.data.
            # OpenAI GUARANTEES the response order matches input order.

            embedding_vector = response.data[i].embedding
            # Get the embedding for THIS specific chunk.
            # response.data[i] is the i-th result from OpenAI.
            # .embedding is the list of 1536 floats.

            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                nct_id=chunk.nct_id,
                chunk_text=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                source=chunk.source,
                word_count=chunk.word_count,
                embedding=embedding_vector,
                # All fields are copied from the original TextChunk
                # and the embedding is added as the new field.
            )

            embedded_chunks.append(embedded_chunk)

        logger.info(
            f"Batch embedded successfully | "
            f"chunks={len(embedded_chunks)} | "
            f"embedding_dims={len(embedded_chunks[0].embedding) if embedded_chunks else 0}"
        )
        # Log the embedding dimensions as a sanity check.
        # Should always show 1536 for text-embedding-3-small.
        # If it ever shows something different — something is wrong.

        return embedded_chunks