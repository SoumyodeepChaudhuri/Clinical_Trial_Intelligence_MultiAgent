##############################################################################
# memory/semantic_store.py
#
# PURPOSE:
#   This file manages the SPONSOR KNOWLEDGE BASE — a growing collection
#   of facts about every research sponsor MOSAIC has ever encountered.
#
# WHAT IS SEMANTIC MEMORY — THE SIMPLE VERSION:
#   Think of a credit rating agency like CIBIL in India or
#   Experian in the US. Every time you take a loan or miss a payment,
#   your credit score changes. The agency builds up a profile of your
#   financial behaviour over years of transactions.
#
#   That is exactly what semantic memory does for sponsors.
#   Every time MOSAIC analyses a study, it updates the sponsor's profile:
#     - Did they post results on time? → credibility goes UP
#     - Did they miss results? → credibility goes DOWN
#     - Did they switch outcomes? → broken promises count goes UP
#     - Did they delay the trial silently? → average delay goes UP
#
#   Over time, MOSAIC builds a rich, data-driven picture of every
#   sponsor's behaviour — not based on reputation, but on EVIDENCE.
#
# EPISODIC vs PROCEDURAL vs SEMANTIC — THE FULL PICTURE:
#   Episodic   = WHAT happened in past sessions (case diary)
#                "I found missing results for NCT04788680 on March 15"
#
#   Procedural = HOW to reason — the rules (rulebook)
#                "Do not flag terminated trials as missing results"
#
#   Semantic   = FACTS about the world — accumulated knowledge (encyclopedia)
#                "Novo Nordisk: 47 studies, credibility 0.82, 2 broken promises"
#
# WHERE SPONSOR PROFILES ARE STORED:
#   In the existing "sponsor_profiles" table in Cloud SQL.
#   This table was created as part of our original schema —
#   no new table creation needed for this file.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. Agents import and use SemanticStore automatically.
#   The Track Record agent uses it most heavily.
#
# HOW AGENTS USE THIS:
#   from memory.semantic_store import SemanticStore
#
#   store = SemanticStore()
#   profile = await store.get_sponsor_profile("Novo Nordisk")
#   await store.update_sponsor_knowledge(
#       sponsor="Novo Nordisk",
#       results_posted=True,
#       had_broken_promise=False,
#       delay_days=12
#   )
##############################################################################


import asyncpg
# asyncpg is our async PostgreSQL driver.
# All database communication goes through this library.
# We use raw asyncpg instead of an ORM because we need fine
# control over the SQL — especially for the UPSERT pattern
# (INSERT or UPDATE in one statement).

from datetime import datetime
# datetime.utcnow() gives the current UTC timestamp.
# We update the "last_updated" column every time a sponsor
# profile changes — so we always know how fresh the data is.

from typing import Any
# Any is a type hint meaning "this value can be any Python type."
# Used for the profile dictionary since different fields have
# different types — str, int, float, datetime etc.

from config.settings import settings
# Our centralised settings — reads all DB credentials from .env.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "memory.semantic_store"


class SemanticStore:
    """
    Manages the sponsor knowledge base — credibility profiles
    built up over time as MOSAIC analyses more studies.

    Each sponsor gets ONE profile row in the sponsor_profiles table.
    That row is updated (never replaced) every time new information
    about that sponsor is discovered during an analysis run.

    Think of it as a living document about each sponsor —
    it grows richer with every analysis, never starts from scratch.

    Usage:
        store = SemanticStore()

        # Get what we know about a sponsor
        profile = await store.get_sponsor_profile("Novo Nordisk")

        # Update after analysing a study
        await store.update_sponsor_knowledge(
            sponsor="Novo Nordisk",
            results_posted=True,
            had_broken_promise=False,
            delay_days=5
        )
    """

    def __init__(self):
        # Constructor — runs when store = SemanticStore() is created.

        self._pool: asyncpg.Pool | None = None
        # Connection pool — starts as None, created lazily on first use.
        # We follow the same lazy initialisation pattern as
        # EpisodicStore and ProceduralStore — only open the database
        # connection when we actually need it.

        logger.info("SemanticStore initialised")

    # ──────────────────────────────────────────────────────────
    # PRIVATE HELPER: _ensure_pool
    # ──────────────────────────────────────────────────────────

    async def _ensure_pool(self) -> None:
        """
        Creates the connection pool if it does not exist yet.
        Called at the start of every public method.
        Same pattern as EpisodicStore and ProceduralStore.
        """

        if self._pool is not None:
            # Pool already open — nothing to do.
            return

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,

            min_size=1,
            # Keep 1 connection always open.
            # Semantic store is called after every study analysis,
            # so there is always at least one active call during a run.

            max_size=5,
            # Allow up to 5 simultaneous connections.
            # The Track Record agent runs in parallel with other agents,
            # so multiple sponsor profile updates can happen at once.
        )

        logger.info("SemanticStore pool created")

    # ──────────────────────────────────────────────────────────
    # CORE METHOD: get_sponsor_profile
    # ──────────────────────────────────────────────────────────

    async def get_sponsor_profile(
        self,
        sponsor: str,
        # The name of the sponsor to look up.
        # Example: "Novo Nordisk", "Pfizer", "National Cancer Institute"
        # Must match exactly how the sponsor appears in the studies table.

    ) -> dict[str, Any] | None:
        # -> dict[str, Any] | None means:
        # Returns a dictionary if the sponsor exists in our database,
        # OR returns None if we have never seen this sponsor before.
        # The caller must handle both cases — check for None before
        # trying to access fields like profile["credibility_score"].
        """
        Retrieves everything we know about a specific sponsor.

        WHAT IS RETURNED:
        A dictionary containing the sponsor's full profile:
          - sponsor           → the sponsor's name
          - credibility_score → 0.0 (worst) to 1.0 (best)
          - total_studies     → how many studies we have analysed
          - results_posted    → how many times they posted results on time
          - results_missing   → how many times results were NOT posted
          - broken_promises   → how many outcome switches detected
          - avg_delay_days    → average days late on timeline
          - last_updated      → when this profile was last modified

        HOW THE CREDIBILITY SCORE IS CALCULATED:
        It is not a simple average — it weights different factors:
          70% → results compliance rate (posted / total studies)
          30% → promise keeping (reduced per broken promise)
        Range: 0.0 to 1.0. Below 0.6 triggers a LOW_CREDIBILITY signal.

        Args:
            sponsor: The sponsor name to look up.

        Returns:
            Dictionary with all profile fields, or None if not found.
        """

        await self._ensure_pool()
        # Make sure the database connection is open before using it.

        async with self._pool.acquire() as conn:
            # Check out one connection from the pool.
            # "async with" returns it automatically when this block ends.

            row = await conn.fetchrow(
                # fetchrow() returns ONE row — not a list of rows.
                # This is correct here because each sponsor has exactly
                # ONE profile row in the sponsor_profiles table.
                # If no row exists for this sponsor, fetchrow() returns None.

                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE sponsor = $1
                """,
                # We SELECT every column — the caller gets the
                # full picture of what we know about this sponsor.
                # WHERE sponsor = $1 filters to just this one sponsor.

                sponsor,
                # $1 — the sponsor name to filter by.
            )

        if row is None:
            # fetchrow() returns None when no matching row exists.
            # This means we have never analysed a study from this sponsor.
            # Return None — the caller decides what to do with a new sponsor.
            # The Track Record agent handles this: "first time seeing this
            # sponsor — not enough data to make a reliable judgment."
            logger.info(
                f"No profile found for sponsor | sponsor={sponsor}"
            )
            return None

        return {
            "sponsor":           row["sponsor"],
            # The sponsor name — same string we searched for.

            "credibility_score": float(row["credibility_score"] or 0.0),
            # float() converts the database Decimal type to Python float.
            # "or 0.0" handles the case where the value is NULL in the DB —
            # NULL or 0.0 evaluates to 0.0, giving us a safe default.

            "total_studies":     int(row["total_studies"] or 0),
            # int() converts to a whole number.
            # Total studies we have ever analysed from this sponsor.

            "results_posted":    int(row["results_posted"] or 0),
            # How many times this sponsor posted results on time.
            # results_posted / total_studies = compliance rate.

            "results_missing":   int(row["results_missing"] or 0),
            # How many times results were NOT posted.
            # Added this field — tracks non-compliance count directly.

            "broken_promises":   int(row["broken_promises"] or 0),
            # How many outcome switches we detected across all their studies.

            "avg_delay_days":    float(row["avg_delay_days"] or 0.0),
            # Average number of days past the stated completion date
            # across all their studies. Higher = more consistently late.

            "last_updated":      str(row["last_updated"]),
            # str() converts datetime to a readable string.
            # Tells us when this profile was last updated —
            # useful for knowing how fresh the data is.
        }

    # ──────────────────────────────────────────────────────────
    # CORE METHOD: update_sponsor_knowledge
    # ──────────────────────────────────────────────────────────

    async def update_sponsor_knowledge(
        self,
        sponsor: str,
        # The sponsor name to update.
        # If this sponsor does not exist yet — CREATE a new profile.
        # If they already exist — UPDATE the existing profile.
        # This create-or-update pattern is called "UPSERT" in databases.

        results_posted: bool = False,
        # Did this sponsor post results for the study we just analysed?
        # True  → compliance count goes up, credibility improves
        # False → missing count goes up, credibility decreases
        # Default False — caller must explicitly pass True for compliance.

        had_broken_promise: bool = False,
        # Did we detect outcome switching in the study we just analysed?
        # True  → broken promises count goes up, credibility decreases
        # False → no change to broken promises count
        # Default False — most studies do not have outcome switching.

        delay_days: int = 0,
        # How many days past the completion date was this study?
        # 0 = on time or completed within the grace period
        # Positive number = days late. Example: 45 means 45 days late.
        # This updates the rolling average delay for this sponsor.
        # Default 0 — assume on time unless the caller specifies otherwise.

    ) -> None:
        # -> None means this method returns nothing.
        # It just updates the database and logs what happened.
        """
        Updates a sponsor's profile with new information from one study.

        USES THE UPSERT PATTERN:
        "UPSERT" = INSERT if the sponsor does not exist,
                   UPDATE if they already do.
        PostgreSQL does this in one statement using:
        INSERT ... ON CONFLICT ... DO UPDATE

        This is more efficient and safer than checking first with
        SELECT, then deciding whether to INSERT or UPDATE.
        The two-step approach has a "race condition" — two agents
        running in parallel could both check, both see "not exists",
        and both try to INSERT, causing a duplicate key error.
        UPSERT handles this atomically — it is thread-safe.

        HOW THE CREDIBILITY SCORE IS RECALCULATED:
        After updating the counts, we recalculate credibility:
          compliance_rate = results_posted_count / total_studies
          promise_penalty = broken_promises * 0.1
          credibility = (compliance_rate * 0.7) - promise_penalty
          credibility = max(0.0, min(1.0, credibility))
          (clamped between 0.0 and 1.0 — cannot go negative or above 1)

        Args:
            sponsor:            The sponsor name.
            results_posted:     Whether results were posted for this study.
            had_broken_promise: Whether outcome switching was detected.
            delay_days:         How many days late this study was.
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:

            # ── STEP 1: UPSERT THE SPONSOR PROFILE ────────────
            await conn.execute(
                """
                INSERT INTO sponsor_profiles (
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                )
                VALUES ($1, 0.5, 1, $2, $3, $4, $5, NOW())
                ON CONFLICT (sponsor) DO UPDATE SET
                    total_studies   = sponsor_profiles.total_studies + 1,
                    results_posted  = sponsor_profiles.results_posted + $2,
                    results_missing = sponsor_profiles.results_missing + $3,
                    broken_promises = sponsor_profiles.broken_promises + $4,
                    avg_delay_days  = (
                        (sponsor_profiles.avg_delay_days *
                         sponsor_profiles.total_studies) + $5
                    ) / (sponsor_profiles.total_studies + 1),
                    last_updated    = NOW()
                """,
                # INSERT section (new sponsor — first time we see them):
                # credibility_score starts at 0.5 — neutral, no data yet.
                # total_studies starts at 1 — this is their first study.
                # $2 = results_posted as int (True=1, False=0)
                # $3 = results_missing as int (True=1, False=0)
                # $4 = broken_promises as int (True=1, False=0)
                # $5 = delay_days (integer)
                #
                # ON CONFLICT (sponsor) DO UPDATE section (existing sponsor):
                # total_studies   → increment by 1 (one more study analysed)
                # results_posted  → add 1 if True, add 0 if False
                # results_missing → add 1 if True, add 0 if False
                # broken_promises → add 1 if True, add 0 if False
                #
                # avg_delay_days CALCULATION EXPLAINED:
                # We compute a RUNNING AVERAGE — updating the average
                # without storing every individual delay value.
                # Formula: new_avg = (old_avg * old_count + new_value)
                #                    / (old_count + 1)
                # Example: old_avg=10 days, old_count=4, new_value=20 days
                #   new_avg = (10 * 4 + 20) / (4 + 1)
                #           = (40 + 20) / 5
                #           = 60 / 5
                #           = 12 days
                # This is the standard "incremental mean" formula —
                # it gives us the running average without storing history.
                #
                # sponsor_profiles.column_name refers to the EXISTING value
                # in the row BEFORE the update — this is PostgreSQL syntax
                # for referencing the current row's values in an UPDATE.

                sponsor,
                # $1 — the sponsor name (primary key for conflict detection)

                int(results_posted),
                # $2 — Python bool True/False → PostgreSQL int 1/0
                # int(True) = 1, int(False) = 0
                # This is how we use boolean values in arithmetic SQL.

                int(not results_posted),
                # $3 — results_missing is the OPPOSITE of results_posted.
                # If results_posted=True → results_missing=0 (not missing)
                # If results_posted=False → results_missing=1 (is missing)
                # int(not True) = int(False) = 0
                # int(not False) = int(True) = 1

                int(had_broken_promise),
                # $4 — 1 if outcome switching was detected, 0 if not.

                float(delay_days),
                # $5 — days late as a float for the avg calculation.
                # float() ensures consistent arithmetic in the SQL.
            )

            # ── STEP 2: RECALCULATE CREDIBILITY SCORE ─────────
            # We do this as a SECOND query after the update above.
            # Why separate? Because the credibility formula needs
            # the UPDATED counts (after step 1), not the old ones.
            # Doing it in one query would use the pre-update values.

            await conn.execute(
                """
                UPDATE sponsor_profiles
                SET credibility_score = GREATEST(0.0, LEAST(1.0,
                    (
                        CASE
                            WHEN total_studies = 0 THEN 0.5
                            ELSE (results_posted::float / total_studies) * 0.7
                        END
                    ) - (broken_promises * 0.1)
                ))
                WHERE sponsor = $1
                """,
                # CREDIBILITY FORMULA EXPLAINED LINE BY LINE:
                #
                # CASE WHEN total_studies = 0 THEN 0.5
                #   → If we somehow have no studies, default to 0.5 (neutral)
                #   → This prevents division by zero
                #
                # ELSE (results_posted::float / total_studies) * 0.7
                #   → results_posted::float casts the integer to float
                #      so PostgreSQL does decimal division, not integer division
                #      Example: 3::float / 4 = 0.75, not 0 (integer division)
                #   → Divide posted results by total studies = compliance rate
                #      Example: 40 posted / 47 total = 0.851
                #   → Multiply by 0.7 = 70% weight on compliance
                #      Example: 0.851 * 0.7 = 0.596
                #
                # - (broken_promises * 0.1)
                #   → Each broken promise reduces credibility by 0.1
                #   → Example: 2 broken promises = -0.2
                #   → Final: 0.596 - 0.2 = 0.396
                #
                # GREATEST(0.0, LEAST(1.0, ...))
                #   → LEAST(1.0, value) caps the score at 1.0 maximum
                #   → GREATEST(0.0, value) prevents the score going negative
                #   → Together they clamp the result between 0.0 and 1.0
                #   → A sponsor cannot have credibility above 1.0 or below 0.0

                sponsor,
                # $1 — update only this specific sponsor's row
            )

        logger.info(
            f"Sponsor knowledge updated | "
            f"sponsor={sponsor} | "
            f"results_posted={results_posted} | "
            f"broken_promise={had_broken_promise} | "
            f"delay_days={delay_days}"
        )

    # ──────────────────────────────────────────────────────────
    # UTILITY METHOD: get_low_credibility_sponsors
    # ──────────────────────────────────────────────────────────

    async def get_low_credibility_sponsors(
        self,
        threshold: float = 0.6,
        # threshold: float = 0.6 means:
        # Return sponsors whose credibility score is BELOW this number.
        # Default is 0.6 — sponsors below this are considered concerning.
        # The caller can override: get_low_credibility_sponsors(threshold=0.5)
        # for stricter filtering.

        min_studies: int = 3,
        # Minimum number of studies required before we flag a sponsor.
        # A sponsor with only 1 study and 1 issue might just be unlucky.
        # We require at least 3 studies before making a judgment —
        # ensures our credibility assessment has enough data to be meaningful.
        # Default is 3 — the minimum for a statistically meaningful pattern.

    ) -> list[dict]:
        # -> list[dict] means: returns a list of sponsor profile dictionaries.
        # Could be empty if no sponsors fall below the threshold.
        """
        Returns all sponsors whose credibility is below the threshold.

        Used by:
        1. The Track Record agent — to quickly identify problematic sponsors
        2. The Pattern Finder agent — to check if a sponsor is a repeat offender
        3. The API endpoint GET /api/v1/sponsors — for analyst dashboards

        Args:
            threshold:   Credibility below this score qualifies as "low".
            min_studies: Minimum studies needed before flagging a sponsor.

        Returns:
            List of sponsor profile dicts ordered by credibility ascending.
            Lowest credibility (worst) sponsors appear first.
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE credibility_score < $1
                  AND total_studies >= $2
                ORDER BY credibility_score ASC
                """,
                # WHERE credibility_score < $1
                #   → Only sponsors below our threshold
                # AND total_studies >= $2
                #   → Only sponsors with enough data to judge
                # ORDER BY credibility_score ASC
                #   → Worst sponsors first (ASC = lowest number first)
                #   → The most concerning sponsors appear at the top

                threshold,
                # $1 — the credibility threshold (default 0.6)

                min_studies,
                # $2 — minimum studies required (default 3)
            )

        sponsors = [
            {
                "sponsor":           row["sponsor"],
                "credibility_score": float(row["credibility_score"] or 0.0),
                "total_studies":     int(row["total_studies"] or 0),
                "results_posted":    int(row["results_posted"] or 0),
                "results_missing":   int(row["results_missing"] or 0),
                "broken_promises":   int(row["broken_promises"] or 0),
                "avg_delay_days":    float(row["avg_delay_days"] or 0.0),
                "last_updated":      str(row["last_updated"]),
            }
            for row in rows
        ]

        logger.info(
            f"Low credibility sponsors found | "
            f"count={len(sponsors)} | "
            f"threshold={threshold} | "
            f"min_studies={min_studies}"
        )

        return sponsors

    # ──────────────────────────────────────────────────────────
    # UTILITY METHOD: get_all_sponsor_profiles
    # ──────────────────────────────────────────────────────────

    async def get_all_sponsor_profiles(
        self,
        limit: int = 50,
        # Maximum number of sponsor profiles to return.
        # Default 50 — enough for an API dashboard view.
        # We do not return ALL sponsors at once — could be thousands.

    ) -> list[dict]:
        """
        Returns all sponsor profiles ordered by credibility.

        Used by the API for analytics dashboards — showing analysts
        the full picture of every sponsor we have knowledge about.

        Args:
            limit: Maximum profiles to return.

        Returns:
            List of all sponsor profiles, lowest credibility first.
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                ORDER BY credibility_score ASC
                LIMIT $1
                """,
                limit,
                # $1 — how many profiles to return
            )

        return [
            {
                "sponsor":           row["sponsor"],
                "credibility_score": float(row["credibility_score"] or 0.0),
                "total_studies":     int(row["total_studies"] or 0),
                "results_posted":    int(row["results_posted"] or 0),
                "results_missing":   int(row["results_missing"] or 0),
                "broken_promises":   int(row["broken_promises"] or 0),
                "avg_delay_days":    float(row["avg_delay_days"] or 0.0),
                "last_updated":      str(row["last_updated"]),
            }
            for row in rows
        ]

    # ──────────────────────────────────────────────────────────
    # UTILITY METHOD: sponsor_exists
    # ──────────────────────────────────────────────────────────

    async def sponsor_exists(self, sponsor: str) -> bool:
        # -> bool means returns True or False.
        """
        Checks if a sponsor profile already exists in the database.

        Used before creating a new profile — avoids duplicate entries.
        Also used by agents to decide whether to load a profile or
        note that "we have never seen this sponsor before."

        Args:
            sponsor: The sponsor name to check.

        Returns:
            True if a profile exists, False if this is a new sponsor.
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                # fetchval() returns a SINGLE VALUE — not a row or list.
                # COUNT(*) returns one number — fetchval() is perfect here.

                "SELECT COUNT(*) FROM sponsor_profiles WHERE sponsor = $1",
                sponsor,
            )

        return (count or 0) > 0
        # count > 0 means at least one row exists → sponsor profile exists.
        # count == 0 means no rows → this sponsor is new to our system.
        # "count or 0" handles None safely — fetchval on empty table returns None.
        # None > 0 would raise TypeError — "None or 0" converts None to 0 first.

    # ──────────────────────────────────────────────────────────
    # CLEANUP METHOD: close
    # ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """
        Closes the connection pool gracefully.
        Call this when the application shuts down.
        """

        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("SemanticStore pool closed")