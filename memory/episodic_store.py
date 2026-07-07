##############################################################################
# memory/episodic_store.py
#
# PURPOSE:
#   This file gives MOSAIC the ability to REMEMBER what it found
#   in past analysis sessions — and search through those memories
#   by meaning, not just by keyword.
#
# WHAT IS EPISODIC MEMORY — THE SIMPLE VERSION:
#   Think of a detective who keeps a case notebook.
#   Every time they investigate a case, they write down what they
#   found, who was involved, and what conclusions they drew.
#   Next time a similar case comes up, they flip through the notebook
#   and ask "have I seen something like this before?"
#
#   That is exactly what episodic memory does for our agents.
#   Every time an agent runs and finds a signal, that session
#   gets saved as an "episode" — a record of what was investigated,
#   what was found, and what the agent concluded.
#
# WHAT IS AN EPISODE EXACTLY?
#   One episode = one agent reasoning session.
#   Example episode:
#     agent_name : "missing_results_agent"
#     nct_id     : "NCT04788680"
#     content    : "Investigated Novo Nordisk trial. Completed May 2019.
#                   Results never posted. 1200 participants enrolled."
#     outcome    : "signal_generated"
#
#   This gets saved with a vector embedding of the content so that
#   future searches can find it by MEANING — not just keyword match.
#
# WHY THIS IS THE KEY DIFFERENTIATOR:
#   99% of agent tutorials build agents that start fresh every time.
#   No memory. No learning. No continuity. Amnesia on every run.
#
#   MOSAIC is different. Every agent run BUILDS on previous runs.
#   Before investigating, the agent asks: "what do I already know?"
#   It searches its past episodes and uses that context.
#   The system gets smarter every time it runs.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. This defines a class used by the agents.
#   Agents import and use EpisodicStore automatically.
##############################################################################


import json
# json is Python's built-in library for converting between
# Python objects and JSON text strings.
# We use it here specifically for the pgvector codec:
#   Python list [0.023, -0.041, ...] → JSON string "[0.023, -0.041, ...]"
# This conversion is required because asyncpg does not natively
# understand pgvector's VECTOR type — we have to teach it.

import uuid
# uuid stands for "Universally Unique Identifier".
# uuid.uuid4() generates a random ID like:
#   "a3f8c2d1-4e5b-6f7a-8b9c-0d1e2f3a4b5c"
# It is "universally unique" — the probability of two UUIDs
# being the same is astronomically small (essentially impossible).
# We use this to give every episode its own unique ID so no
# two episodes ever clash in the database.

from datetime import datetime
# datetime is Python's built-in library for working with dates and times.
# datetime.utcnow() gives us the current moment in UTC timezone.
# UTC = Coordinated Universal Time — a standard time with no timezone
# offset. We always store timestamps in UTC so there is no confusion
# between timezones (India, US, Europe all see the same timestamp).
# We stamp every episode with its creation time so we can later
# ask "what did we find in the last 7 days?" or sort by recency.

from openai import AsyncOpenAI
# AsyncOpenAI is OpenAI's official Python client — the ASYNC version.
# "Async" means it supports Python's await keyword, so our program
# does not freeze while waiting for OpenAI to respond.
# The regular OpenAI() client is synchronous — it blocks everything
# while waiting. AsyncOpenAI lets other things happen in parallel.
# We use this to embed episode content into 1536-number vectors,
# the same way we embedded study chunks in the processing layer.

import asyncpg
# asyncpg is the fastest async PostgreSQL driver for Python.
# It lets us talk to our Cloud SQL database using Python's
# async/await pattern — non-blocking database calls.
# We need raw asyncpg (instead of SQLAlchemy) because we need
# to register a custom codec for pgvector's VECTOR type.
# SQLAlchemy does not support VECTOR natively.

from config.settings import settings
# Our centralised settings object — reads all values from .env.
# We use it for:
#   settings.db_host, db_port, db_name, db_user, db_password
#   settings.openai_api_key
#   settings.openai_embedding_model

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ in this file = "memory.episodic_store"
# Every log line from this file shows that path so we always
# know exactly which file is printing each log message.


##############################################################################
# THE EpisodicStore CLASS
#
# A class is a blueprint for creating objects.
# EpisodicStore is a blueprint for an object that knows how to:
#   - Connect to our Cloud SQL database
#   - Save agent reasoning sessions as episodes
#   - Search through past episodes by meaning
#   - Return recent episodes for the API
#
# We create ONE instance of this class and reuse it throughout
# the application rather than creating a new instance for every call.
# This is efficient because the database connection pool is expensive
# to create — we want to create it once and reuse it many times.
##############################################################################

class EpisodicStore:
    """
    Stores and retrieves agent reasoning sessions as episodes.

    Each episode is one agent's reasoning session — what it
    investigated, what it found, and what it concluded.
    Episodes are embedded and stored so future sessions can
    search through past findings by meaning.

    Usage:
        store = EpisodicStore()

        # Save what an agent found
        await store.save_episode(
            agent_name="missing_results_agent",
            nct_id="NCT04788680",
            content="Novo Nordisk trial completed 2019. Results never posted.",
            outcome="signal_generated"
        )

        # Search past episodes by meaning
        past = await store.search_episodes(
            query="sponsor never posted results",
            agent_name="missing_results_agent",
            top_k=3
        )
    """

    def __init__(self):
        # __init__ is the CONSTRUCTOR — Python calls this automatically
        # when you create a new EpisodicStore object.
        # Example: store = EpisodicStore()  ← __init__ runs here
        # We use it to set up the initial state of the object.

        self._pool: asyncpg.Pool | None = None
        # self._pool will hold our database connection pool.
        # A connection pool is a group of reusable database connections
        # that stay open and ready — much faster than opening a new
        # connection for every single database call.
        # We initialise it as None because we do not open the database
        # connection immediately when the object is created.
        # Instead we open it lazily — only when someone actually
        # calls a method that needs the database.
        # The underscore prefix (_pool) is a Python convention meaning
        # "this is private — only use it inside this class, not outside."
        # asyncpg.Pool | None means: this can be either a Pool object
        # OR None (before it has been initialised).

        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        # Create the async OpenAI client using our API key from .env.
        # We create it here in __init__ so it is ready immediately
        # without any extra setup calls.
        # This client is used for ONE purpose in this file:
        # converting text into 1536-number vector embeddings.

        self._embedding_model = settings.openai_embedding_model
        # Store the embedding model name as an instance variable.
        # Value: "text-embedding-3-small"
        # Using an instance variable means we can see it in one place
        # and change it easily if needed.
        # IMPORTANT: This must match the model used in embedder.py
        # because episode embeddings and chunk embeddings must be
        # in the SAME vector space to be comparable.
        # If we used different models, similarity search would give
        # meaningless results.

        logger.info("EpisodicStore initialised")

    # ──────────────────────────────────────────────────────────
    # PRIVATE HELPER METHOD: _ensure_pool
    #
    # WHAT IS A "PRIVATE" METHOD?
    # Methods starting with underscore (_) are private by convention.
    # They are internal helpers — only called by other methods
    # INSIDE this class, never directly from outside.
    # Example: store._ensure_pool()  ← you would never write this
    #          Instead it is called automatically by save_episode()
    #          and search_episodes() whenever they need the database.
    # ──────────────────────────────────────────────────────────

    async def _ensure_pool(self) -> None:
        # async def means this is an ASYNCHRONOUS method.
        # You must use "await" when calling it:
        #   await self._ensure_pool()
        # Without await, Python would not actually run the method —
        # it would just return a coroutine object, which is useless.
        #
        # -> None means this method returns nothing.
        # It just makes sure the pool exists — no value comes back.
        """
        Makes sure the database connection pool is open.

        WHAT IS LAZY INITIALISATION?
        Instead of opening the database connection the moment
        EpisodicStore() is created, we wait until someone actually
        tries to use it. This is called "lazy initialisation."

        WHY DO WE DO THIS?
        When MOSAIC starts up, it imports many classes including
        EpisodicStore. If we opened the database connection in
        __init__, every import would immediately try to connect
        to Cloud SQL — even if that class is never actually used
        in that run. Lazy initialisation avoids wasted connections.

        This method is called at the START of every public method
        (save_episode, search_episodes, etc.) to guarantee the
        pool is open before we try to use it.
        """

        if self._pool is not None:
            # self._pool is not None means the pool already exists.
            # Nothing to do — return immediately.
            # "is not None" is more precise than just "if self._pool"
            # because an empty pool could theoretically be falsy.
            return

        # If we get here, self._pool is None — pool does not exist yet.
        # Create it now.

        self._pool = await asyncpg.create_pool(
            # create_pool() creates a group of database connections
            # that stay open and ready to use.
            # We "await" it because connecting to the database is a
            # network operation that takes time — await lets other
            # async tasks run while we wait.

            host=settings.db_host,
            # The IP address of our Cloud SQL instance.
            # Value from .env: "34.133.55.17"

            port=settings.db_port,
            # Which port PostgreSQL listens on.
            # Value from .env: 5432 (standard PostgreSQL port)

            database=settings.db_name,
            # Which database inside the Cloud SQL instance to connect to.
            # Value from .env: "clinical_trial_db"

            user=settings.db_user,
            # The database username.
            # Value from .env: "mosaic_user"

            password=settings.db_password,
            # The database password.
            # Value from .env: "YourNewSecurePassword123"
            # Never hardcode passwords — always read from .env.

            min_size=1,
            # Keep at least 1 connection open at all times.
            # This means the first query after startup does not
            # have to wait for a new connection to be established.
            # There is always at least one ready.

            max_size=5,
            # Allow at most 5 simultaneous connections.
            # EpisodicStore is called less frequently than VectorStore
            # (agents check memory at the START of a session, not
            # on every query), so 5 connections is more than enough.
            # Using fewer connections also reduces Cloud SQL load.

            init=self._init_connection,
            # init= is a special asyncpg parameter.
            # It tells asyncpg: "run this function on EVERY new
            # connection you create before using it."
            # We pass self._init_connection which registers our
            # custom pgvector codec.
            # Without this, asyncpg would not know how to read
            # or write VECTOR columns and every query would fail.
        )

        logger.info("EpisodicStore pool created")

    # ──────────────────────────────────────────────────────────
    # WHAT IS A STATIC METHOD?
    #
    # A @staticmethod is a method that belongs to the CLASS
    # but does NOT have access to the instance (self) or the
    # class itself (cls).
    #
    # Normal method:   def my_method(self, ...)
    #                  → has access to self._pool, self._openai etc.
    #
    # Static method:   @staticmethod
    #                  def my_method(...)
    #                  → no self, no access to instance variables
    #
    # WHY USE STATIC HERE?
    # _init_connection does not need to access ANY instance
    # variables — it only works with the "conn" parameter passed
    # to it by asyncpg. Using @staticmethod makes this explicit
    # and prevents accidental use of self inside the method.
    # It is also slightly more memory-efficient.
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        # conn: asyncpg.Connection means the parameter "conn" must
        # be an asyncpg Connection object. asyncpg passes this
        # automatically when calling our init function —
        # we never call _init_connection ourselves.
        """
        Registers the custom pgvector VECTOR codec on a connection.

        WHAT IS A CODEC?
        A codec is a pair of functions:
          ENCODER: converts Python data → database format
          DECODER: converts database format → Python data

        WHY DO WE NEED A CUSTOM CODEC FOR VECTOR?
        PostgreSQL knows about the VECTOR type (added by pgvector).
        But asyncpg is a generic driver — it only knows about
        standard PostgreSQL types like TEXT, INTEGER, FLOAT etc.
        It has NO idea what to do with VECTOR.

        Without registering this codec:
          - Writing: asyncpg cannot convert our Python list of
            floats into the format pgvector expects → error
          - Reading: asyncpg cannot convert the VECTOR value it
            gets back from the database into a Python list → error

        With this codec registered:
          - Writing: [0.023, -0.041, 0.891] → "[0.023,-0.041,0.891]"
          - Reading: "[0.023,-0.041,0.891]" → [0.023, -0.041, 0.891]

        asyncpg calls this function automatically on every new
        connection it creates, via the init= parameter above.

        Args:
            conn: One fresh database connection from the pool.
                  asyncpg passes this automatically.
        """

        await conn.set_type_codec(
            # set_type_codec() teaches asyncpg about a custom type.
            # We call it with await because it is an async operation —
            # it sends a query to PostgreSQL to look up type information.

            "vector",
            # The name of the custom type we are teaching asyncpg about.
            # Must be lowercase and must exactly match what pgvector
            # registered in PostgreSQL when we ran CREATE EXTENSION vector.

            encoder=lambda v: json.dumps(v),
            # ENCODER — Python → PostgreSQL
            #
            # lambda v: json.dumps(v) is a small anonymous function.
            # lambda v:       means: "given a value called v"
            # json.dumps(v)   means: "convert v to a JSON string"
            #
            # When we INSERT a chunk embedding (a Python list of floats),
            # asyncpg calls this encoder to convert it:
            # [0.023, -0.041, 0.891] → "[0.023, -0.041, 0.891]"
            # pgvector accepts this JSON array string as a VECTOR.
            #
            # Why json.dumps? Because pgvector's text representation
            # is a JSON array — square brackets, comma-separated numbers.
            # json.dumps() produces exactly that format.

            decoder=lambda v: json.loads(v),
            # DECODER — PostgreSQL → Python
            #
            # lambda v: json.loads(v) is the reverse operation.
            # When we SELECT an embedding from the database,
            # asyncpg receives a text string like "[0.023, -0.041, 0.891]"
            # and calls this decoder to convert it back:
            # "[0.023, -0.041, 0.891]" → [0.023, -0.041, 0.891]
            # Now we have a proper Python list we can work with.

            schema="public",
            # Tells asyncpg which PostgreSQL schema the "vector" type
            # lives in. "public" is the default schema — all our tables,
            # indexes, and the pgvector extension live in "public".
            # Without specifying the schema, asyncpg might look in
            # the wrong place and fail to find the vector type.
        )

    # ──────────────────────────────────────────────────────────
    # PRIVATE HELPER: _embed
    # ──────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        # text: str  → this method takes one string as input
        # -> list[float] → and returns a list of floating point numbers
        """
        Converts any text string into a 1536-number vector embedding.

        WHAT IS AN EMBEDDING — ONE MORE TIME SIMPLY:
        An embedding is a position on a giant meaning map.
        Similar texts end up close together on that map.
        "Results never posted" and "outcomes not published" would
        be very close together even though the words are different.

        We use this method in TWO places:
        1. When SAVING an episode — embed the content so it can
           be found by semantic search later.
        2. When SEARCHING episodes — embed the query so we can
           compare it against stored episode embeddings.

        Using the SAME model for both is critical — embeddings
        from different models cannot be compared meaningfully.

        Args:
            text: Any text string to convert to an embedding.

        Returns:
            A list of exactly 1536 floating point numbers.
            Example (first 3 of 1536): [0.023, -0.041, 0.891, ...]
        """

        response = await self._openai.embeddings.create(
            # embeddings.create() sends text to OpenAI and gets
            # back vector embeddings. We await it because it is
            # a network request — we pause here until OpenAI responds.

            model=self._embedding_model,
            # Which embedding model to use.
            # Value: "text-embedding-3-small"
            # Always produces exactly 1536 dimensions.

            input=text,
            # The text to embed. Can be a string or list of strings.
            # Here we pass one string at a time because episodes
            # are saved and searched one at a time, unlike the
            # processing layer where we batched 50 chunks at once.
        )

        return response.data[0].embedding
        # response.data is a list of embedding results.
        # [0] gets the first (and only) result — we only sent one text.
        # .embedding is the list of 1536 floats we need.
        # This is what gets stored in the database's VECTOR column.

    # ──────────────────────────────────────────────────────────
    # CORE PUBLIC METHOD: save_episode
    # ──────────────────────────────────────────────────────────

    async def save_episode(
        self,
        # self refers to the EpisodicStore instance.
        # All instance methods receive self as their first parameter.
        # Through self we can access self._pool, self._openai etc.

        agent_name: str,
        # Which agent is saving this episode.
        # Type hint "str" means this must be a string.
        # Example values: "missing_results_agent", "broken_promises_agent"
        # Used later to filter episodes by agent —
        # "show me only missing_results_agent's past sessions."

        content: str,
        # What the agent investigated and what it found.
        # This is the main body of the episode — written like a
        # detective's case note.
        # Example:
        # "Investigated NCT04788680. Sponsor: Novo Nordisk.
        #  Trial completed May 2019. Results never posted.
        #  Enrolled 1200 participants. Phase 3 diabetes study."
        # THIS is what gets converted to 1536 numbers and stored.
        # Future searches compare against this content.

        nct_id: str | None = None,
        # Which study this episode is about.
        # str | None means this can be a string OR None (optional).
        # Default is None — some episodes span multiple studies
        # and do not belong to one specific NCT ID.
        # When provided, allows filtering: "show me all episodes
        # about study NCT04788680."

        outcome: str | None = None,
        # What happened as a result of this investigation.
        # Example values:
        #   "signal_generated"  → agent found something worth flagging
        #   "no_signal"         → agent investigated but found nothing
        #   "sent_to_review"    → signal was low confidence, sent to HITL
        # Used for filtering and analytics.

    ) -> str:
        # -> str means this method returns a string (the episode_id).
        """
        Saves one agent reasoning session as an episode in Cloud SQL.

        WHAT HAPPENS INSIDE THIS METHOD:
        1. Makes sure the database connection is open
        2. Generates a unique ID for this episode
        3. Converts the content to a 1536-number embedding
        4. Inserts all of this into the episodes table
        5. Returns the episode_id

        Args:
            agent_name: Which agent is saving this.
            content:    What the agent found — plain text.
            nct_id:     Which study this is about (optional).
            outcome:    What happened as a result (optional).

        Returns:
            episode_id — the unique ID of the saved episode.
        """

        await self._ensure_pool()
        # Call our private helper to make sure the database
        # connection pool is open before we try to use it.
        # If the pool is already open, this returns immediately.
        # If not, it creates the pool first.
        # We call this at the START of every public method.

        episode_id = str(uuid.uuid4())
        # uuid.uuid4() generates a random unique ID object.
        # str() converts it to a plain string we can store in the DB.
        # Example result: "a3f8c2d1-4e5b-6f7a-8b9c-0d1e2f3a4b5c"
        # Every episode gets its own unique ID — no two ever clash.

        embedding = await self._embed(content)
        # Convert the episode content to a 1536-number vector.
        # We await this because it makes a network call to OpenAI.
        # This embedding is what enables semantic search later —
        # "find episodes similar in meaning to this query."
        # We embed the CONTENT (what the agent found) not the
        # agent_name or outcome — content is the meaningful part.

        async with self._pool.acquire() as conn:
            # self._pool.acquire() checks out one connection from the pool.
            # Think of the pool like a parking lot of database connections.
            # acquire() says: "give me one connection to use right now."
            # "async with" means: when this block finishes (or crashes),
            # the connection is automatically returned to the pool.
            # We never manually close the connection — the pool handles it.
            # conn is the connection object we use for our SQL query below.

            await conn.execute(
                # conn.execute() runs a SQL statement on the database.
                # We await it because sending SQL to the database is
                # a network operation that takes time.
                # execute() is used for INSERT, UPDATE, DELETE —
                # statements that do not return rows.
                # (For SELECT statements we use fetch() instead.)

                """
                INSERT INTO episodes (
                    episode_id,
                    agent_name,
                    nct_id,
                    content,
                    outcome,
                    embedding,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                # This is a PARAMETERISED SQL statement.
                # $1, $2, $3... are placeholders for the actual values.
                # asyncpg fills them in from the arguments below.
                # WHY NOT JUST PUT VALUES DIRECTLY IN THE SQL STRING?
                # Because that would be vulnerable to SQL injection attacks.
                # A malicious value like "'; DROP TABLE episodes; --"
                # could destroy our database if concatenated directly.
                # Parameterised queries prevent this — the values are
                # always treated as data, never as SQL commands.

                episode_id,
                # $1 — the unique ID we generated above

                agent_name,
                # $2 — which agent is saving this episode

                nct_id,
                # $3 — which study (can be None)

                content,
                # $4 — what the agent found (plain text)

                outcome,
                # $5 — what happened (can be None)

                embedding,
                # $6 — the 1536-number list from OpenAI
                # Our custom codec (registered in _init_connection)
                # automatically converts this Python list to the
                # format pgvector expects. We do not need to do
                # any conversion manually here.

                datetime.utcnow(),
                # $7 — the current timestamp in UTC
                # datetime.utcnow() returns the current moment
                # as a Python datetime object.
                # asyncpg knows how to convert this to PostgreSQL
                # TIMESTAMPTZ format automatically — no custom codec needed.
            )
            # After execute() completes, the episode is saved.
            # The "async with" block ends here and the connection
            # is automatically returned to the pool.

        logger.info(
            f"Episode saved | "
            f"agent={agent_name} | "
            f"nct_id={nct_id} | "
            f"outcome={outcome} | "
            f"episode_id={episode_id}"
        )
        # Log a confirmation that the episode was saved successfully.
        # If this log line appears, the episode is in the database.
        # If it does NOT appear after calling save_episode(), something
        # went wrong before this point.

        return episode_id
        # Return the unique ID of the saved episode.
        # The caller can store this if they need to reference or
        # update this specific episode later.

    # ──────────────────────────────────────────────────────────
    # CORE PUBLIC METHOD: search_episodes
    # ──────────────────────────────────────────────────────────

    async def search_episodes(
        self,
        query: str,
        # The question we are asking about past sessions.
        # This is converted to a 1536-number embedding and compared
        # against all stored episode embeddings.
        # Example: "find episodes where sponsor never posted results"
        # Example: "find episodes about Novo Nordisk"
        # The search is SEMANTIC — it finds episodes with SIMILAR
        # MEANING even if the exact words are different.

        agent_name: str | None = None,
        # Optional filter — only search episodes from this specific agent.
        # Example: agent_name="missing_results_agent"
        # When set, only that agent's past sessions are searched.
        # When None, all agents' episodes are searched.
        # Agents usually filter to their OWN past sessions —
        # the missing results agent only wants to know what IT
        # found before, not what the broken promises agent found.

        top_k: int = 5,
        # How many past episodes to return.
        # "top_k" means "top K results" — a common term in search.
        # Default is 5 — return the 5 most similar past episodes.
        # The results are sorted most similar first.
        # Agents typically use 3-5 past episodes as context.

        min_similarity: float = 0.5,
        # Minimum similarity score for a result to be included.
        # float means a decimal number between 0.0 and 1.0.
        # 1.0 = identical meaning
        # 0.5 = at least 50% similar in meaning
        # 0.0 = completely different
        # Episodes below 0.5 similarity are probably not relevant
        # and would just add noise to the agent's context.
        # This threshold is tunable — increase it for stricter
        # matching, decrease it to cast a wider net.

    ) -> list[dict]:
        # -> list[dict] means this returns a list of dictionaries.
        # Each dictionary represents one past episode.
        """
        Searches past episodes by semantic similarity to a query.

        WHAT HAPPENS INSIDE THIS METHOD:
        1. Makes sure the database connection is open
        2. Converts the query text to a 1536-number embedding
        3. Builds a SQL query with optional filters
        4. Uses pgvector's <=> operator to find similar episodes
        5. Returns the top_k most similar episodes as dictionaries

        Think of this as the agent "consulting its memory" before
        starting a new investigation.

        Args:
            query:          What to search for in past episodes.
            agent_name:     Optional filter — only this agent's episodes.
            top_k:          How many results to return.
            min_similarity: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of episode dictionaries, most similar first.
            Each dict contains: episode_id, agent_name, nct_id,
                               content, outcome, similarity, created_at
            Empty list if no relevant past episodes found.
        """

        await self._ensure_pool()
        # Make sure the pool is open before using it.
        # Same call as in save_episode() — always the first step.

        query_embedding = await self._embed(query)
        # Convert the search query into 1536 numbers.
        # Example: "sponsor never posted results"
        # → [0.023, -0.041, 0.891, ...] (1536 numbers)
        # These numbers are then compared against every stored
        # episode embedding using cosine distance.
        # Episodes whose embeddings are numerically close to the
        # query embedding are returned as results.

        # ── BUILD THE SQL QUERY DYNAMICALLY ───────────────────
        # We build the SQL query as a string because different
        # calls may or may not include the agent_name filter.
        # A fixed SQL string would need to handle all combinations —
        # building it dynamically is cleaner.

        sql = """
            SELECT
                episode_id,
                agent_name,
                nct_id,
                content,
                outcome,
                created_at,
                1 - (embedding <=> $1) AS similarity
                -- 1 - (embedding <=> $1) EXPLAINED:
                --
                -- embedding <=> $1
                -- The <=> operator is pgvector's cosine DISTANCE.
                -- It compares the embedding column (stored in the DB)
                -- against $1 (our query embedding).
                -- Result: a number from 0.0 to 2.0
                --   0.0 = identical (same meaning, same direction)
                --   2.0 = opposite (completely different meaning)
                --
                -- 1 - (cosine distance) converts to cosine SIMILARITY.
                -- This flips the scale so HIGHER = MORE SIMILAR:
                --   distance 0.0 → similarity 1.0 (identical)
                --   distance 0.5 → similarity 0.5
                --   distance 1.0 → similarity 0.0 (unrelated)
                --
                -- Similarity is more intuitive than distance —
                -- "similarity of 0.85" means clearly relevant.
                -- "distance of 0.15" means the same thing but
                -- is less obvious to humans reading the output.
                -- We use similarity for display, distance for sorting.
            FROM episodes
            WHERE 1 - (embedding <=> $1) >= $2
            -- WHERE filters out episodes below our threshold.
            -- Only episodes with similarity >= min_similarity pass.
            -- $1 = query_embedding, $2 = min_similarity
            --
            -- "WHERE 1=1" at the end of the FROM would also work
            -- as a base, but here we always have the similarity
            -- filter so we use it directly in WHERE.
        """

        params: list = [query_embedding, min_similarity]
        # params is the list of values that replace $1, $2, $3... in the SQL.
        # We start with:
        #   $1 = query_embedding  (the 1536-number query vector)
        #   $2 = min_similarity   (the threshold, e.g. 0.5)
        # As we add more optional filters below, we add to this list.

        param_idx = 3
        # param_idx tracks the NEXT parameter number to use.
        # We start at 3 because $1 and $2 are already taken.
        # Each time we add a new filter condition, we:
        #   1. Add f"AND something = ${param_idx}" to the SQL string
        #   2. Append the value to params list
        #   3. Increment param_idx by 1
        # This ensures the parameter numbers always match their values.
        # Example after adding agent_name filter:
        #   params = [query_embedding, min_similarity, "missing_results_agent"]
        #   SQL has: WHERE ... AND agent_name = $3
        # The $3 correctly maps to "missing_results_agent" in params.

        if agent_name:
            # Only add this filter if agent_name was provided.
            # If agent_name is None (the default), we skip this block
            # and search ALL agents' episodes.
            sql += f" AND agent_name = ${param_idx}"
            # f-string inserts the current param_idx number into the SQL.
            # Example if param_idx=3: adds " AND agent_name = $3"

            params.append(agent_name)
            # Add the agent_name value to our params list.
            # It becomes the 3rd element — matching $3 in the SQL.

            param_idx += 1
            # Increment so the next filter uses $4, not $3 again.

        sql += f"""
            ORDER BY embedding <=> $1
            -- Sort results by cosine DISTANCE ascending.
            -- Ascending means SMALLEST distance first.
            -- Smallest distance = MOST SIMILAR = most relevant.
            -- We use the raw distance for sorting (not similarity)
            -- because it is more numerically precise for ordering.
            -- The similarity score we calculated above is for display.
            LIMIT ${param_idx}
            -- Return only the top_k most similar results.
            -- $param_idx will be the last parameter in our list.
        """
        params.append(top_k)
        # Add top_k as the last parameter.
        # It matches the ${param_idx} in the LIMIT clause above.

        # ── EXECUTE THE SEARCH QUERY ───────────────────────────

        async with self._pool.acquire() as conn:
            # Check out one connection from the pool.
            # Same pattern as save_episode() — acquire, use, return.

            rows = await conn.fetch(sql, *params)
            # conn.fetch() runs a SELECT query and returns ALL rows.
            # Note: fetch() NOT execute() — fetch() is for SELECT
            # statements that return data. execute() is for INSERT,
            # UPDATE, DELETE that change data but return no rows.
            #
            # *params unpacks our list into individual arguments.
            # Example: params = [query_embedding, 0.5, "agent", 5]
            # conn.fetch(sql, *params) becomes:
            # conn.fetch(sql, query_embedding, 0.5, "agent", 5)
            # asyncpg maps these to $1, $2, $3, $4 in the SQL.
            #
            # rows is a list of asyncpg Record objects.
            # Each Record is like a dictionary — access fields by name:
            #   row["episode_id"], row["content"], row["similarity"]

        episodes = [
            # This is a LIST COMPREHENSION — a compact way to build
            # a new list by transforming each item in another list.
            # It is equivalent to:
            #   episodes = []
            #   for row in rows:
            #       episodes.append({...})
            # But written in one readable expression.

            {
                "episode_id": row["episode_id"],
                # The unique UUID of this episode.

                "agent_name": row["agent_name"],
                # Which agent created this episode.

                "nct_id":     row["nct_id"],
                # Which study this episode is about (may be None).

                "content":    row["content"],
                # What the agent found — the full text of the episode.
                # This is what the agent reads to remember past findings.

                "outcome":    row["outcome"],
                # What happened — "signal_generated", "no_signal" etc.

                "similarity": round(float(row["similarity"]), 3),
                # The similarity score between this episode and the query.
                # float() converts the asyncpg Decimal to a Python float.
                # round(..., 3) rounds to 3 decimal places:
                #   0.847392847362... → 0.847
                # Cleaner for logging and display.

                "created_at": str(row["created_at"]),
                # When this episode was saved.
                # str() converts the datetime object to a readable string:
                # "2024-03-15 14:32:11.123456+00:00"
            }
            for row in rows
            # For each row returned by the database, build one dict.
        ]

        logger.info(
            f"Episode search complete | "
            f"query='{query[:50]}...' | "
            # query[:50] takes only the first 50 characters of the query.
            # Prevents extremely long queries from flooding the log.
            f"agent_filter={agent_name} | "
            f"results_found={len(episodes)}"
        )

        return episodes
        # Return the list of episode dictionaries.
        # If no episodes matched, this is an empty list [].
        # The agent handles empty results gracefully —
        # "no past episodes found, starting fresh."

    # ──────────────────────────────────────────────────────────
    # UTILITY METHOD: get_recent_episodes
    # ──────────────────────────────────────────────────────────

    async def get_recent_episodes(
        self,
        agent_name: str | None = None,
        limit: int = 10,
        # limit: int = 10 means:
        # "limit is an integer, default value is 10"
        # If the caller does not pass a limit, we return 10 episodes.
        # The caller can override: get_recent_episodes(limit=20)

    ) -> list[dict]:
        """
        Returns the most recent episodes, newest first.

        Unlike search_episodes() which finds episodes by MEANING,
        this method finds episodes by TIME — just the most recent ones.

        Used by the API endpoint GET /api/v1/memory/episodes so
        analysts can browse what the agents have been doing recently.

        Args:
            agent_name: Optional — filter to one agent's episodes.
            limit:      Maximum number of episodes to return.

        Returns:
            List of episode dicts ordered newest first.
        """

        await self._ensure_pool()

        if agent_name:
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                WHERE agent_name = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            # ORDER BY created_at DESC means:
            # Sort by creation timestamp DESCENDING.
            # DESC = descending = newest first, oldest last.
            # (ASC = ascending = oldest first, newest last.)
            params = [agent_name, limit]
        else:
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                ORDER BY created_at DESC
                LIMIT $1
            """
            params = [limit]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id":     row["nct_id"],
                "content":    row["content"],
                "outcome":    row["outcome"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    # ──────────────────────────────────────────────────────────
    # UTILITY METHOD: count_episodes
    # ──────────────────────────────────────────────────────────

    async def count_episodes(
        self,
        agent_name: str | None = None,
    ) -> int:
        # -> int means this method returns a whole number (integer).
        """
        Returns the total number of episodes stored.

        Used by the health check endpoint to show how much memory
        the system has accumulated — how many past sessions exist.

        Args:
            agent_name: Optional — count only this agent's episodes.

        Returns:
            Integer count. Example: 47 (meaning 47 episodes stored.)
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:

            if agent_name:
                count = await conn.fetchval(
                    # fetchval() is like fetch() but returns a SINGLE VALUE
                    # instead of a list of rows.
                    # Perfect for COUNT(*) queries that return one number.
                    # fetch() would return [{"count": 47}] — a list with one dict.
                    # fetchval() returns just 47 — the number directly.
                    "SELECT COUNT(*) FROM episodes WHERE agent_name = $1",
                    agent_name,
                )
            else:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes"
                )

        return count or 0
        # "count or 0" handles the case where count is None.
        # fetchval() returns None if no rows exist (empty table).
        # None or 0 evaluates to 0 — a safe default.
        # Without this, code that expects an integer would crash
        # when it received None from an empty episodes table.

    # ──────────────────────────────────────────────────────────
    # CLEANUP METHOD: close
    # ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """
        Closes the connection pool and releases all connections.

        Call this when the application shuts down cleanly.
        Without closing, connections may stay open on Cloud SQL
        unnecessarily — wasting resources and potentially hitting
        connection limits.

        In FastAPI, this is called in the lifespan shutdown handler.
        """

        if self._pool:
            # Only try to close if the pool was actually created.
            # If _ensure_pool() was never called (store was never used),
            # self._pool is still None — nothing to close.

            await self._pool.close()
            # close() gracefully closes ALL connections in the pool.
            # It waits for any active queries to finish first,
            # then closes each connection cleanly.
            # We await it because this is an async operation.

            self._pool = None
            # Reset to None so the object is in a clean state.
            # If someone accidentally calls a method after close(),
            # _ensure_pool() will create a new pool rather than
            # trying to use a closed one.

            logger.info("EpisodicStore pool closed")