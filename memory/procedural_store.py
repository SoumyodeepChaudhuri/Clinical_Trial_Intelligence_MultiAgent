##############################################################################
# memory/procedural_store.py
#
# PURPOSE:
#   This file gives MOSAIC agents the ability to learn HOW to reason
#   better over time — based on feedback from human reviewers.
#
# WHAT IS PROCEDURAL MEMORY — THE SIMPLE VERSION:
#   Think of a junior doctor who just started working.
#   On day 1, they follow a basic checklist when diagnosing patients.
#   Over time, senior doctors correct them:
#     "When a patient has BOTH symptom A and symptom B together,
#      always check for condition X first — not condition Y."
#   The junior doctor writes this correction into their personal
#   rulebook. Next time they see that combination, they follow
#   the updated rule automatically.
#
#   That is exactly what procedural memory does for our agents.
#   Each agent starts with a set of DEFAULT reasoning rules.
#   When a human reviewer REJECTS a signal and explains why,
#   that rejection reason gets written into the agent's rulebook
#   as a new rule — permanently changing how it reasons.
#
# EPISODIC vs PROCEDURAL — WHAT IS THE DIFFERENCE?
#   Episodic memory  = WHAT happened in past sessions
#                      "I found missing results for NCT04788680"
#                      Like a diary — records of past events.
#
#   Procedural memory = HOW to reason — the rules themselves
#                       "When a trial was terminated early,
#                        missing results are expected — not a violation"
#                       Like a rulebook — guidelines for behaviour.
#
# THE LEARNING LOOP — THIS IS THE MOST IMPORTANT CONCEPT:
#   1. Agent generates a signal → "NCT04788680 has missing results"
#   2. Human reviewer looks at it
#   3. Human REJECTS it with reason:
#      "This trial was terminated early due to COVID —
#       missing results for terminated trials are expected"
#   4. That reason gets written into the agent's procedures table
#      as a new rule: "Check if trial was TERMINATED before flagging
#      missing results — terminated trials are exempt"
#   5. Next time the agent runs, it LOADS its procedures first
#   6. It applies the new rule and correctly skips terminated trials
#
#   ONE human correction → agent reasons differently FOREVER after.
#   This is the learning loop that makes MOSAIC smarter over time.
#
# WHERE PROCEDURES ARE STORED:
#   In Cloud SQL, in the "procedures" table.
#   This table needs to be created — SQL is provided below.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. Agents import and use ProceduralStore automatically.
##############################################################################


import asyncpg
# asyncpg is our async PostgreSQL driver.
# Same library used in episodic_store.py and vector_store.py.
# We use it for all direct database communication.

import json
# json is used for the pgvector VECTOR codec — same as episodic_store.py.
# Converts Python lists to JSON strings for pgvector compatibility.

from datetime import datetime
# datetime.utcnow() gives us the current UTC timestamp.
# We stamp every procedure with when it was created or updated
# so we can track how the agent's reasoning has evolved over time.

from config.settings import settings
# Our centralised settings — DB credentials from .env file.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "memory.procedural_store"
# Every log line from this file shows that module path.


##############################################################################
# BEFORE USING THIS FILE — CREATE THE PROCEDURES TABLE
#
# Connect to Cloud SQL and run this SQL:
#
# CREATE TABLE IF NOT EXISTS procedures (
#     procedure_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
#     agent_name     TEXT NOT NULL,
#     rule_text      TEXT NOT NULL,
#     rule_type      TEXT DEFAULT 'learned',
#     source         TEXT DEFAULT 'default',
#     created_at     TIMESTAMPTZ DEFAULT NOW(),
#     updated_at     TIMESTAMPTZ DEFAULT NOW()
# );
#
# WHAT EACH COLUMN MEANS:
#   procedure_id  → unique ID for this rule
#   agent_name    → which agent owns this rule
#   rule_text     → the actual reasoning rule in plain English
#   rule_type     → "default" (built-in) or "learned" (from HITL)
#   source        → "default" (initial) or "hitl_rejection" (from human)
#   created_at    → when this rule was first created
#   updated_at    → when this rule was last modified
##############################################################################


##############################################################################
# DEFAULT RULES — EVERY AGENT STARTS WITH THESE
#
# These are the built-in reasoning rules that every agent loads
# at the very start of every session — BEFORE doing any analysis.
# Think of these as the agent's "training manual" that it received
# on day 1. They encode the baseline knowledge of what to look for
# and what edge cases to handle.
#
# When a human rejects a signal and explains why, a NEW rule is
# added to the database alongside these defaults. On the next run,
# the agent loads BOTH the defaults AND any learned rules.
##############################################################################

DEFAULT_RULES = {
    # Each key is an agent name.
    # Each value is a LIST of rule strings for that agent.
    # These strings are plain English — the agent reads them directly
    # as part of its system prompt before it starts reasoning.

    "missing_results_agent": [
        "Flag a study as missing results ONLY if status is COMPLETED "
        "and results_posted is False and more than 12 months have "
        "passed since the completion date.",
        # WHY 12 MONTHS?
        # By US law (FDAAA 801), sponsors must post results within
        # 12 months of the primary completion date. We do not flag
        # studies that have not yet hit this deadline.

        "Do NOT flag studies with status TERMINATED as missing results. "
        "Terminated trials are not legally required to post results "
        "in all circumstances — termination often means the study "
        "was stopped early and has incomplete data.",
        # This rule prevents a common false positive —
        # terminated trials look like they have missing results
        # but they are often exempt from the posting requirement.

        "If enrollment was zero or very low (under 10 participants), "
        "note this in the signal but reduce confidence to 0.5. "
        "A study that never really started may not have reportable results.",

        "Always check the sponsor's track record before assigning "
        "a confidence score. A first-time missing result from a "
        "historically compliant sponsor warrants lower confidence "
        "than the same finding from a repeat offender.",
    ],

    "broken_promises_agent": [
        "Flag outcome switching ONLY when the PRIMARY outcome changes "
        "after enrollment has begun. Changes to secondary outcomes "
        "are less concerning and should not trigger a HIGH confidence signal.",
        # The primary outcome is what the study was designed to measure.
        # Changing it after the study starts is the red flag.
        # Changing secondary outcomes is far more common and acceptable.

        "A change in outcome MEASUREMENT METHOD (how it is measured) "
        "is different from a change in the outcome itself. "
        "Method changes may be legitimate protocol improvements — "
        "flag them at MEDIUM confidence, not HIGH.",

        "If a protocol amendment was filed BEFORE enrollment began, "
        "the outcome change is less suspicious — the study had not "
        "yet collected data that could have influenced the change. "
        "Assign MEDIUM confidence in this case.",

        "Always note the date of the change relative to the "
        "enrollment start date — this timing is the most important "
        "factor in assessing whether outcome switching is intentional.",
    ],

    "track_record_agent": [
        "A credibility score below 0.6 should trigger a LOW_CREDIBILITY "
        "signal. Between 0.6 and 0.75 is concerning but not alarming — "
        "note it in the analysis but do not generate a signal.",

        "Weight recent behaviour more heavily than old behaviour. "
        "A sponsor with 5 violations in the last 2 years is more "
        "concerning than one with 10 violations spread over 20 years.",

        "If a sponsor has fewer than 3 studies in our database, "
        "reduce confidence to 0.5. We do not have enough data to "
        "make a reliable judgment about their track record.",

        "Always distinguish between a sponsor's PRIMARY studies "
        "(where they are the lead sponsor) and COLLABORATIVE studies "
        "(where they are a secondary party). Hold them more accountable "
        "for their primary studies.",
    ],

    "pattern_finder_agent": [
        "A cross-study pattern requires at least 3 studies to be "
        "meaningful. Two studies with similar issues may be coincidence. "
        "Three or more is a pattern worth flagging.",

        "When multiple companies are testing the same drug for the "
        "same condition, check whether any of them have hidden "
        "negative results from previous studies in our database.",

        "A drug that failed Phase 2 for condition A but is being "
        "retried in Phase 2 for condition B is worth flagging — "
        "especially if the mechanism of action is the same.",

        "Patterns across the same SPONSOR are more actionable than "
        "patterns across different sponsors. Same-sponsor patterns "
        "suggest systemic issues, not coincidence.",
    ],

    "side_effect_agent": [
        "A safety discrepancy between the official filing and a "
        "published paper is only meaningful if the paper was published "
        "AFTER the trial completed — not during it.",

        "Look specifically for cases where the filing says "
        "'no serious adverse events' but published papers mention "
        "hospitalisations, discontinuations, or deaths. "
        "This is the highest-priority safety signal.",

        "If the discrepancy is minor (e.g. different terminology "
        "for the same event), assign LOW confidence. "
        "If the discrepancy involves severity (mild vs serious), "
        "assign HIGH confidence.",

        "Always note whether the paper's authors are the same as "
        "the trial's investigators. Independent authors are more "
        "credible than sponsor-employed investigators.",
    ],

    "timeline_agent": [
        "Flag a delay ONLY if it exceeds 180 days beyond the "
        "stated completion date AND no amendment was filed explaining "
        "the extension. A silent delay is more suspicious than "
        "a disclosed one.",

        "COVID-19 is a legitimate reason for delays between "
        "March 2020 and December 2022. Do not flag delays in this "
        "period as suspicious without additional evidence.",

        "A study that is recruiting past its stated completion date "
        "may simply have underestimated enrollment time — this is "
        "common and not inherently suspicious. Focus on COMPLETED "
        "studies that are past their results posting deadline.",

        "Always compare the actual completion date against BOTH "
        "the original completion date AND any amended completion "
        "dates. Use the most recent amendment as the baseline.",
    ],
}


class ProceduralStore:
    """
    Stores and retrieves agent reasoning rules (procedures).

    Each agent has its own set of procedures — rules that guide
    how it reasons about clinical trial data. Procedures come
    in two types:
      1. DEFAULT — built-in rules the agent always starts with
      2. LEARNED — rules added when a human reviewer corrects the agent

    Usage:
        store = ProceduralStore()

        # Load all rules for an agent before it starts reasoning
        rules = await store.get_procedures("missing_results_agent")

        # Update rules after a human rejection
        await store.update_from_feedback(
            agent_name="missing_results_agent",
            rejection_reason="Terminated trials should not be flagged"
        )
    """

    def __init__(self):
        # __init__ is the constructor — runs when you create:
        # store = ProceduralStore()

        self._pool: asyncpg.Pool | None = None
        # Connection pool starts as None — created lazily on first use.
        # Same pattern as EpisodicStore — we only open the database
        # connection when we actually need it.

        logger.info("ProceduralStore initialised")

    # ──────────────────────────────────────────────────────────
    # PRIVATE HELPER: _ensure_pool
    # ──────────────────────────────────────────────────────────

    async def _ensure_pool(self) -> None:
        """
        Creates the database connection pool if it does not exist.
        Called at the start of every public method — guarantees
        the database is reachable before we try to use it.
        """

        if self._pool is not None:
            # Pool already exists — nothing to do, return immediately.
            return

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=1,
            # Keep 1 connection always open — agents load procedures
            # at the start of every run so there is always at least
            # one call. Having 1 ready avoids connection setup latency.
            max_size=3,
            # Procedural memory calls are infrequent — one load per
            # agent per run. 3 connections is more than enough.
        )

        logger.info("ProceduralStore pool created")

    # ──────────────────────────────────────────────────────────
    # CORE METHOD: initialise_defaults
    # ──────────────────────────────────────────────────────────

    async def initialise_defaults(self) -> None:
        """
        Inserts the DEFAULT_RULES into the procedures table.

        WHEN IS THIS CALLED?
        Once — when MOSAIC starts up for the very first time.
        After that, the rules are already in the database and
        this method safely skips any that already exist
        (using ON CONFLICT DO NOTHING).

        WHY NOT HARDCODE RULES IN THE AGENT?
        Because we want rules to be:
        1. Stored in the database — persistent across runs
        2. Updateable — humans can add new learned rules
        3. Visible — the API can return current rules on demand
        4. Auditable — we can see how rules changed over time

        If rules were hardcoded in the agent, learned rules would
        disappear every time the application restarted. Storing
        them in the database makes them permanent.
        """

        await self._ensure_pool()
        # Make sure the database connection is open.

        async with self._pool.acquire() as conn:
            # Check out one connection from the pool.
            # "async with" returns it automatically when done.

            for agent_name, rules in DEFAULT_RULES.items():
                # DEFAULT_RULES is our dictionary defined above.
                # .items() returns pairs of (key, value):
                #   ("missing_results_agent", ["rule 1", "rule 2", ...])
                #   ("broken_promises_agent", ["rule 1", "rule 2", ...])
                # We loop through every agent and its rules.

                for rule_text in rules:
                    # rules is the list of rule strings for this agent.
                    # We insert each rule as a separate row in the table.
                    # One row = one rule. An agent with 4 rules → 4 rows.

                    await conn.execute(
                        """
                        INSERT INTO procedures (
                            agent_name,
                            rule_text,
                            rule_type,
                            source
                        )
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                        """,
                        # ON CONFLICT DO NOTHING means:
                        # If this exact combination already exists
                        # in the database, skip the insert silently.
                        # This makes this method SAFE TO CALL MULTIPLE TIMES.
                        # Running initialise_defaults() again will not
                        # duplicate any rules — it just skips existing ones.

                        agent_name,
                        # $1 — which agent this rule belongs to

                        rule_text,
                        # $2 — the actual rule in plain English

                        "default",
                        # $3 — rule_type: "default" means this is a
                        # built-in rule, not one learned from feedback

                        "default",
                        # $4 — source: "default" means it came from
                        # our DEFAULT_RULES dictionary, not from HITL
                    )

        logger.info(
            f"Default procedures initialised | "
            f"agents={list(DEFAULT_RULES.keys())}"
        )
        # list(DEFAULT_RULES.keys()) converts the dictionary keys
        # into a list for clean logging:
        # ["missing_results_agent", "broken_promises_agent", ...]

    # ──────────────────────────────────────────────────────────
    # CORE METHOD: get_procedures
    # ──────────────────────────────────────────────────────────

    async def get_procedures(
        self,
        agent_name: str,
        # Which agent's rules to load.
        # Example: "missing_results_agent"
        # Each agent only loads ITS OWN rules —
        # an agent should not follow another agent's reasoning rules.

    ) -> list[str]:
        # -> list[str] means this returns a list of strings.
        # Each string is one reasoning rule in plain English.
        """
        Returns ALL reasoning rules for a specific agent.

        This is called at the VERY START of every agent session —
        before the agent does any analysis. The agent reads these
        rules and incorporates them into its system prompt so that
        GPT-4o reasons according to them.

        The returned list contains BOTH:
        - Default rules (built-in at startup)
        - Learned rules (added from human feedback over time)

        The agent has no idea which rules are default and which
        are learned — it just receives a list and follows all of them.
        This is intentional — all rules are equally authoritative.

        Args:
            agent_name: Which agent's rules to retrieve.

        Returns:
            List of rule strings, ordered oldest first.
            Example:
            [
              "Flag missing results only if completed AND 12+ months...",
              "Do NOT flag TERMINATED trials as missing results...",
              "When COVID caused the delay, do not flag timeline issues..."
            ]
            Empty list if no rules found (should not happen in practice).
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                # fetch() returns ALL matching rows — not just one.
                # This is correct here because we want ALL rules for
                # this agent, not just the first one.

                """
                SELECT rule_text
                FROM procedures
                WHERE agent_name = $1
                ORDER BY created_at ASC
                """,
                # ORDER BY created_at ASC means:
                # Return rules in the order they were created.
                # ASC = ascending = oldest first, newest last.
                # This means default rules come first (created at startup),
                # followed by learned rules in the order they were added.
                # Agents read rules in this order — defaults first,
                # then any corrections learned from human feedback.

                agent_name,
                # $1 — filter to only this agent's rules
            )

        rules = [row["rule_text"] for row in rows]
        # List comprehension: for each row returned from the database,
        # extract just the "rule_text" field.
        # rows is a list of asyncpg Record objects.
        # row["rule_text"] accesses the rule_text column by name.
        # Result: ["rule 1 text", "rule 2 text", "rule 3 text", ...]

        logger.info(
            f"Procedures loaded | "
            f"agent={agent_name} | "
            f"rules_count={len(rules)}"
        )

        return rules

    # ──────────────────────────────────────────────────────────
    # CORE METHOD: update_from_feedback
    # ──────────────────────────────────────────────────────────

    async def update_from_feedback(
        self,
        agent_name: str,
        # Which agent to add the new rule to.
        # The rule is SPECIFIC to this agent — other agents are
        # not affected by this update.

        rejection_reason: str,
        # The reason the human reviewer gave for rejecting the signal.
        # Example: "This trial was terminated early due to COVID —
        #           terminated trials are exempt from result posting"
        # This plain-English reason becomes a new reasoning rule.
        # The agent will read it at the start of every future session.

    ) -> str:
        # -> str means this returns the procedure_id (a UUID string)
        # of the newly created rule.
        """
        Adds a new learned reasoning rule from a human rejection.

        THIS IS THE LEARNING LOOP IN ACTION.

        When a human reviewer rejects an agent's signal, they explain
        why it was wrong. That explanation is passed to this method.
        This method saves it as a new rule in the procedures table.

        From this point forward, every time this agent runs, it will:
        1. Load its procedures (including this new rule)
        2. Apply the new rule during its reasoning
        3. Avoid making the same mistake again

        One human correction → permanent change in agent behaviour.
        This is what makes MOSAIC genuinely intelligent over time.

        Args:
            agent_name:       Which agent to add the rule to.
            rejection_reason: The human's explanation of what was wrong.

        Returns:
            procedure_id — the unique ID of the new rule.
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            procedure_id = await conn.fetchval(
                # fetchval() returns a SINGLE VALUE — not a list of rows.
                # Perfect here because INSERT ... RETURNING gives us
                # one value back: the auto-generated procedure_id.

                """
                INSERT INTO procedures (
                    agent_name,
                    rule_text,
                    rule_type,
                    source
                )
                VALUES ($1, $2, $3, $4)
                RETURNING procedure_id
                """,
                # RETURNING procedure_id means:
                # After inserting the new row, give us back the
                # procedure_id that was auto-generated for it.
                # This lets us return the ID to the caller without
                # needing to run a separate SELECT query.

                agent_name,
                # $1 — which agent gets this new rule

                rejection_reason,
                # $2 — the human's explanation becomes the rule text.
                # We store it exactly as the human wrote it —
                # no modification, no summarisation.
                # The agent reads these rules as-is in its system prompt.

                "learned",
                # $3 — rule_type: "learned" means this rule came
                # from human feedback, not built-in defaults.
                # Distinguishing learned from default is useful for:
                # - Analytics: how many rules have we learned?
                # - Debugging: which rules changed agent behaviour?
                # - Auditing: when did each correction happen?

                "hitl_rejection",
                # $4 — source: "hitl_rejection" = Human In The Loop.
                # This clearly identifies WHERE this rule came from —
                # a human rejected a signal and gave a reason.
                # Future analytics can filter: "show me all rules
                # that came from human rejections in the last month."
            )

        logger.info(
            f"Procedure learned from feedback | "
            f"agent={agent_name} | "
            f"rule_preview='{rejection_reason[:80]}...' | "
            f"procedure_id={procedure_id}"
            # rejection_reason[:80] takes only the first 80 characters.
            # Prevents very long rejection reasons from flooding the log.
            # "..." visually indicates the text continues beyond 80 chars.
        )

        return str(procedure_id)
        # str() converts the UUID object that PostgreSQL returned
        # into a plain Python string for the caller to use.
        # Example: "b7c9e4f2-1a3d-5e6f-8b9c-0d1e2f3a4b5c"

    # ──────────────────────────────────────────────────────────
    # UTILITY METHOD: get_all_procedures_for_api
    # ──────────────────────────────────────────────────────────

    async def get_all_procedures_for_api(
        self,
        agent_name: str,
    ) -> list[dict]:
        # -> list[dict] means: returns a list of dictionaries.
        # Each dict is one rule with all its metadata.
        # This is richer than get_procedures() which only returns
        # rule text — this returns everything for the API to display.
        """
        Returns all procedures for an agent WITH full metadata.

        Unlike get_procedures() which just returns rule text strings,
        this method returns the full procedure record including
        the rule type, source, and timestamps.

        Used by the API endpoint:
        GET /api/v1/memory/procedures/{agent_name}

        This lets analysts see:
        - What rules the agent currently follows
        - Which are built-in vs learned from feedback
        - When each rule was added

        Args:
            agent_name: Which agent's procedures to return.

        Returns:
            List of procedure dictionaries with full metadata.
        """

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    procedure_id,
                    agent_name,
                    rule_text,
                    rule_type,
                    source,
                    created_at
                FROM procedures
                WHERE agent_name = $1
                ORDER BY created_at ASC
                """,
                agent_name,
            )

        return [
            {
                "procedure_id": str(row["procedure_id"]),
                # str() converts UUID to string for JSON serialisation.
                # JSON does not support UUID objects — only strings.
                # FastAPI needs all values to be JSON-serialisable
                # when returning API responses.

                "agent_name":   row["agent_name"],
                "rule_text":    row["rule_text"],
                "rule_type":    row["rule_type"],
                # "default" or "learned" — tells the API viewer
                # whether this is a built-in rule or a learned one.

                "source":       row["source"],
                # "default" or "hitl_rejection" — where did this come from?

                "created_at":   str(row["created_at"]),
                # str() converts datetime to string for JSON serialisation.
            }
            for row in rows
        ]

    # ──────────────────────────────────────────────────────────
    # CLEANUP METHOD: close
    # ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """
        Closes the connection pool gracefully.
        Call this when the application shuts down.
        """

        if self._pool:
            # Only close if the pool was actually created.
            await self._pool.close()
            # close() waits for active queries to finish,
            # then closes all connections cleanly.

            self._pool = None
            # Reset to None — clean state after closing.

            logger.info("ProceduralStore pool closed")