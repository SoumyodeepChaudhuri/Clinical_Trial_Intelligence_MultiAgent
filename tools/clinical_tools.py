##############################################################################
# tools/clinical_tools.py
#
# PURPOSE:
#   This file defines tools that let agents fetch LIVE data directly
#   from the ClinicalTrials.gov API during an analysis run.
#
# WAIT — DIDN'T WE ALREADY DOWNLOAD STUDY DATA DURING INGESTION?
#   Yes — and that data lives in our Cloud SQL database.
#   So why do we need these tools?
#
#   Three reasons:
#
#   1. FRESHNESS — ClinicalTrials.gov updates constantly.
#      A study that showed "results_posted: False" during ingestion
#      might have posted results since then. These tools fetch
#      the CURRENT state directly from the source.
#
#   2. DETAIL — Our ingestion pipeline stores a subset of fields.
#      The full ClinicalTrials.gov API response has hundreds of fields.
#      When an agent needs a very specific field we did not store
#      (like a specific amendment date or trial arm detail),
#      these tools fetch it directly.
#
#   3. DISCOVERY — Agents may encounter NCT IDs during analysis
#      that were never in our original ingestion corpus.
#      These tools let agents fetch any study on demand.
#
# IMPORTANT — THESE TOOLS MAKE LIVE API CALLS:
#   Unlike search_tools.py which only queries our local database,
#   these tools hit the real ClinicalTrials.gov API over the internet.
#   They add network latency to agent runs (~1-2 seconds per call).
#   Agents should use database tools first and only call these
#   when they specifically need live or detailed data.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. Agents import these tools. Do not run directly.
##############################################################################


import json
# json.dumps() converts Python dicts to JSON strings.
# LangGraph tools must return strings — never dicts or lists.

from langchain_core.tools import tool
# @tool decorator transforms functions into LangGraph-compatible tools.
# GPT-4o reads the function name, docstring, and parameter types
# to decide when and how to call each tool autonomously.

from ingestion.clinical_trials_client import ClinicalTrialsClient
# ClinicalTrialsClient is the class we built in ingestion/
# that knows how to talk to the ClinicalTrials.gov API.
# It handles authentication headers, rate limiting, and retry logic.
# We reuse it here rather than reimplementing API calls from scratch.

from ingestion.document_parser import DocumentParser
# DocumentParser cleans raw API responses into ParsedStudy objects.
# When we fetch fresh data from the API, we clean it the same way
# as during ingestion — consistent data format everywhere.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "tools.clinical_tools"


##############################################################################
# SHARED INSTANCES
#
# One DocumentParser instance shared across all tool calls.
# DocumentParser is stateless — it does not remember anything
# between calls — so one instance can safely serve all tools.
#
# We do NOT create a shared ClinicalTrialsClient here because
# it is an async context manager (used with "async with").
# We create a fresh client inside each tool call instead.
# This is slightly less efficient but much safer — the client
# opens and closes its HTTP connection cleanly for each call.
##############################################################################

_parser = DocumentParser()
# One shared DocumentParser — stateless, safe to reuse.
# Created once at module level to avoid creating a new object
# on every tool call.


##############################################################################
# THE ASYNC BRIDGE — SAME PATTERN AS search_tools.py
#
# LangGraph calls tools synchronously.
# Our ClinicalTrialsClient uses async/await.
# We use asyncio to bridge this gap — same approach as search_tools.py.
##############################################################################

import asyncio
# asyncio lets us run async functions synchronously inside tool calls.

def _run_async(coroutine):
    """
    Runs an async coroutine synchronously.

    LangGraph tools are called synchronously by the framework,
    but our API clients use async/await. This function bridges
    that gap by running the coroutine on the event loop.

    Args:
        coroutine: An unawaited async function call.

    Returns:
        The result of the async function, returned synchronously.
    """

    loop = asyncio.get_event_loop()
    # get_event_loop() returns the currently running event loop.
    # LangGraph always runs inside an event loop so this is safe.

    return loop.run_until_complete(coroutine)
    # run_until_complete() runs the coroutine to completion
    # and returns its result synchronously.


##############################################################################
# TOOL 1: fetch_study_details
##############################################################################

@tool
def fetch_study_details(nct_id: str) -> str:
    """
    Fetch the complete, LIVE record for one specific clinical trial
    directly from ClinicalTrials.gov API.

    Use this tool when:
    - You need the most current version of a study (freshest data)
    - You need fields that may not be in our local database
    - The study was not in our original ingestion corpus

    This makes a LIVE API call — slightly slower than database queries.
    Prefer database search tools when current data is not critical.

    Args:
        nct_id: The study's unique identifier.
                Format: "NCT" followed by 8 digits.
                Example: "NCT04788680"

    Returns:
        JSON string with the complete cleaned study record.
        Returns an error message if the study is not found.
    """

    logger.info(
        f"Tool called: fetch_study_details | nct_id={nct_id}"
    )
    # Log every tool call — we can see in the terminal exactly
    # which NCT IDs the agent is fetching during its reasoning.

    async def _fetch():
        # We define the async logic as an inner function.
        # WHY AN INNER FUNCTION?
        # Because _run_async() needs a coroutine to run.
        # An inner async def gives us that coroutine cleanly
        # without polluting the module namespace with async functions
        # that LangGraph cannot call directly.

        async with ClinicalTrialsClient() as client:
            # "async with" opens the HTTP session when we enter
            # and closes it when we exit — even if an error occurs.
            # This ensures no HTTP connections are left open accidentally.

            raw_study = await client.fetch_study(nct_id=nct_id)
            # fetch_study() makes one GET request to:
            # https://clinicaltrials.gov/api/v2/studies/{nct_id}
            # Returns the raw JSON response as a Python dictionary.
            # Returns None if the study is not found (404 response).

            return raw_study
            # Return the raw dict back to the outer function.

    try:
        raw_study = _run_async(_fetch())
        # Run our async inner function synchronously.
        # raw_study is now a Python dictionary or None.

        if raw_study is None:
            # The API returned 404 — study does not exist.
            # Return a clear message so the agent understands.
            return json.dumps({
                "found":   False,
                "nct_id":  nct_id,
                "message": f"Study {nct_id} was not found on "
                           "ClinicalTrials.gov. The NCT ID may be "
                           "incorrect or the study may have been removed.",
            }, indent=2)

        parsed_study = _parser.parse_study(raw=raw_study)
        # Clean the raw API response into a ParsedStudy object.
        # parse_study() extracts the fields we care about and
        # puts them in a consistent, predictable format.

        if parsed_study is None:
            # parse_study() returns None if something went wrong
            # during parsing — unexpected API response structure.
            return json.dumps({
                "found":   False,
                "nct_id":  nct_id,
                "message": "Study was found but could not be parsed. "
                           "The API response had an unexpected structure.",
            }, indent=2)

        study_dict = parsed_study.model_dump()
        # model_dump() is a Pydantic method that converts our
        # ParsedStudy object into a plain Python dictionary.
        # We need a plain dict to convert to JSON string.
        # Pydantic objects cannot be directly JSON-serialised.

        study_dict.pop("raw_data", None)
        # Remove the raw_data field before returning to the agent.
        # raw_data contains the FULL original API response —
        # it can be thousands of lines of nested JSON.
        # The agent does not need this much detail and it would
        # flood the context window, wasting tokens.
        # .pop("raw_data", None) removes the key if it exists,
        # does nothing if it does not — safe to call always.

        return json.dumps({
            "found":   True,
            "nct_id":  nct_id,
            "study":   study_dict,
            # The cleaned study with all key fields:
            # title, sponsor, phase, status, conditions,
            # primary_outcome, results_posted, completion_date etc.
        }, indent=2, default=str)
        # default=str handles non-JSON-serialisable values
        # like datetime objects — converts them to strings.

    except Exception as e:
        logger.error(
            f"fetch_study_details failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "found": False,
            "error": str(e),
            "nct_id": nct_id,
        })


##############################################################################
# TOOL 2: search_studies_by_condition
##############################################################################

@tool
def search_studies_by_condition(
    condition: str,
    max_results: int = 10,
    status_filter: str = "COMPLETED",
) -> str:
    """
    Search ClinicalTrials.gov LIVE for studies matching a condition.

    Use this tool when you want to find trials for a specific
    medical condition that may not be in our local database.

    Examples of when to use this:
    - "Find all completed diabetes trials from the last 5 years"
    - "Search for cardiovascular trials from Pfizer"
    - "Find recruiting trials for this drug"

    This makes LIVE API calls. Results may differ from our database
    because ClinicalTrials.gov updates continuously.

    Args:
        condition:     Medical condition to search for.
                       Example: "diabetes", "cancer", "heart failure"
        max_results:   Maximum studies to return. Default 10. Max 50.
                       Keep low to avoid slow tool calls.
        status_filter: Filter by study status.
                       "COMPLETED"            → only completed studies
                       "RECRUITING"           → only recruiting studies
                       "ACTIVE_NOT_RECRUITING"→ ongoing but not recruiting
                       Default "COMPLETED" — most relevant for signal detection.

    Returns:
        JSON string with a list of matching studies (cleaned format).
    """

    logger.info(
        f"Tool called: search_studies_by_condition | "
        f"condition={condition} | max_results={max_results}"
    )

    async def _search():
        # Inner async function — same pattern as fetch_study_details.
        # Keeps the async logic contained and clean.

        async with ClinicalTrialsClient() as client:
            raw_studies = await client.search_studies(
                condition=condition,
                # The medical condition to search for.
                # ClinicalTrials.gov v2 uses this for the query.cond parameter.

                status=[status_filter] if status_filter else None,
                # Pass status as a list — the client expects a list.
                # ClinicalTrials.gov accepts multiple statuses:
                # ["COMPLETED", "TERMINATED"] would find both.
                # We pass just one for simplicity.
                # If status_filter is empty string, pass None (no filter).

                max_results=min(max_results, 50),
                # min() ensures we never request more than 50 studies.
                # 50 is our safety cap — more than 50 from a tool call
                # would overwhelm the agent's context window with data.
                # The agent reads every result — keep it manageable.
            )

            return raw_studies

    try:
        raw_studies = _run_async(_search())
        # Run the async search synchronously.

        if not raw_studies:
            return json.dumps({
                "studies": [],
                "count":   0,
                "message": f"No studies found for condition '{condition}' "
                           f"with status '{status_filter}'.",
            }, indent=2)

        parsed_studies = _parser.parse_studies(raw_studies=raw_studies)
        # Clean all raw studies in one batch call.
        # parse_studies() processes the whole list and skips any
        # that fail to parse without crashing the whole batch.

        studies_list = []
        for study in parsed_studies:
            study_dict = study.model_dump()
            # Convert each ParsedStudy object to a plain dictionary.

            study_dict.pop("raw_data", None)
            # Remove raw_data — too large for agent context.

            study_dict.pop("protocol_amendments", None)
            # Remove protocol_amendments too — detailed amendment
            # records are large and rarely needed at this stage.
            # The agent can call fetch_study_details() for a specific
            # study if it needs amendment details.

            studies_list.append(study_dict)

        return json.dumps({
            "studies":      studies_list,
            "count":        len(studies_list),
            "condition":    condition,
            "status_filter": status_filter,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_studies_by_condition failed | "
            f"condition={condition} | error={e}"
        )
        return json.dumps({
            "error":   str(e),
            "studies": [],
            "count":   0,
        })


##############################################################################
# TOOL 3: check_results_posted
##############################################################################

@tool
def check_results_posted(nct_id: str) -> str:
    """
    Check if a specific clinical trial has posted results — RIGHT NOW.

    Use this tool when you need the CURRENT results posting status
    for a study. This gives you the live status from ClinicalTrials.gov,
    not the status stored during ingestion (which may be outdated).

    This is the most important tool for the Missing Results Agent.
    A study that had no results during ingestion may have posted
    results since then — this tool catches that.

    Args:
        nct_id: The study's NCT ID to check.
                Example: "NCT04788680"

    Returns:
        JSON string with:
        - results_posted: True if results are posted, False if not
        - completion_date: When the study completed
        - status: Current study status
        - months_overdue: How many months past the 12-month deadline
                          (only present if results are missing)
    """

    logger.info(
        f"Tool called: check_results_posted | nct_id={nct_id}"
    )

    async def _check():
        async with ClinicalTrialsClient() as client:
            raw_study = await client.fetch_study(nct_id=nct_id)
            return raw_study

    try:
        raw_study = _run_async(_check())

        if raw_study is None:
            return json.dumps({
                "nct_id":  nct_id,
                "found":   False,
                "message": f"Study {nct_id} not found on ClinicalTrials.gov.",
            }, indent=2)

        parsed = _parser.parse_study(raw=raw_study)
        # Parse the raw study into a clean ParsedStudy object.

        if parsed is None:
            return json.dumps({
                "nct_id":  nct_id,
                "found":   False,
                "message": "Could not parse study response.",
            }, indent=2)

        result = {
            "nct_id":          parsed.nct_id,
            "found":           True,
            "results_posted":  parsed.results_posted,
            # True = results are on ClinicalTrials.gov right now.
            # False = no results posted — potential violation.

            "status":          parsed.status,
            # Current study status: COMPLETED, TERMINATED, RECRUITING etc.
            # Missing results only matter for COMPLETED studies.

            "completion_date": parsed.completion_date,
            # When the study finished.
            # The 12-month posting deadline starts from this date.

            "sponsor":         parsed.sponsor,
            # Who is responsible for posting results.
        }

        if not parsed.results_posted and parsed.status == "COMPLETED":
            # This is the core signal: COMPLETED but no results posted.
            # Calculate how overdue this study is.

            if parsed.completion_date:
                try:
                    from datetime import datetime

                    completion = datetime.strptime(
                        parsed.completion_date[:7],
                        "%Y-%m"
                        # Parse just the year and month from the date string.
                        # ClinicalTrials.gov dates come in "YYYY-MM" format.
                        # strptime() converts a string to a datetime object.
                        # "%Y-%m" is the format pattern:
                        #   %Y = 4-digit year (e.g. 2019)
                        #   %m = 2-digit month (e.g. 05)
                    )

                    now = datetime.utcnow()
                    # Current date and time in UTC.

                    months_since_completion = (
                        (now.year - completion.year) * 12
                        + (now.month - completion.month)
                    )
                    # Calculate total months between completion and now.
                    # Formula: year difference in months + month difference
                    # Example: completion=2019-05, now=2024-03
                    #   years diff = 2024-2019 = 5 → 5*12 = 60 months
                    #   months diff = 3-5 = -2 months
                    #   total = 60 + (-2) = 58 months since completion

                    months_overdue = months_since_completion - 12
                    # Subtract 12 because sponsors have 12 months to post.
                    # If months_since_completion = 58 and we subtract 12,
                    # the study is 46 months OVERDUE.
                    # Negative means still within the 12-month window.

                    if months_overdue > 0:
                        result["months_overdue"] = months_overdue
                        result["years_overdue"]  = round(
                            months_overdue / 12, 1
                        )
                        # Also express overdue time in years for readability.
                        # round(..., 1) gives one decimal place: 3.8 years.
                        # This makes the agent's signal summary more impactful:
                        # "Results missing for 3.8 years" is more readable
                        # than "Results missing for 46 months."

                        result["is_violation"] = True
                        # Explicitly flag this as a legal violation.
                        # The agent can include this in its signal summary.

                except ValueError:
                    # strptime() raises ValueError if the date string
                    # does not match the expected format.
                    # Some studies have unusual date formats — we skip
                    # the calculation but still return the basic result.
                    pass

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"check_results_posted failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "found":  False,
        })


##############################################################################
# TOOL 4: get_study_amendments
##############################################################################

@tool
def get_study_amendments(nct_id: str) -> str:
    """
    Fetch the protocol amendment history for a specific clinical trial.

    Use this tool when investigating whether a study changed its
    primary outcomes or design mid-study — the core signal for
    the Broken Promises Agent.

    Protocol amendments are official, time-stamped changes to the
    study design filed with ClinicalTrials.gov. They tell us:
    - When the study design changed
    - What changed
    - Whether the change happened BEFORE or AFTER enrollment began

    Timing matters: changes AFTER enrollment has started are more
    suspicious than changes before enrollment began.

    Args:
        nct_id: The study's NCT ID.
                Example: "NCT04788680"

    Returns:
        JSON string with the amendment history and key timing details.
    """

    logger.info(
        f"Tool called: get_study_amendments | nct_id={nct_id}"
    )

    async def _fetch():
        async with ClinicalTrialsClient() as client:
            raw_study = await client.fetch_study(nct_id=nct_id)
            return raw_study

    try:
        raw_study = _run_async(_fetch())

        if raw_study is None:
            return json.dumps({
                "nct_id":     nct_id,
                "found":      False,
                "amendments": [],
            }, indent=2)

        parsed = _parser.parse_study(raw=raw_study)

        if parsed is None:
            return json.dumps({
                "nct_id":     nct_id,
                "found":      False,
                "amendments": [],
                "message":    "Could not parse study.",
            }, indent=2)

        return json.dumps({
            "nct_id":          parsed.nct_id,
            "found":           True,
            "title":           parsed.title,
            "sponsor":         parsed.sponsor,
            "start_date":      parsed.start_date,
            # When enrollment began — critical for timing analysis.
            # An amendment filed BEFORE this date is less suspicious.
            # An amendment filed AFTER this date is more suspicious.

            "completion_date": parsed.completion_date,
            "primary_outcome": parsed.primary_outcome,
            # What the study is currently measuring as its primary outcome.
            # The Broken Promises agent compares this against what was
            # originally filed to detect outcome switching.

            "amendments":      parsed.protocol_amendments,
            # List of amendment records — each one has a date and description.
            # The agent reads through these to find outcome-related changes.

            "amendment_count": len(parsed.protocol_amendments),
            # Quick summary count — a study with 10 amendments is
            # more concerning than one with 0 or 1.
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"get_study_amendments failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "found":  False,
        })


##############################################################################
# TOOL REGISTRY
#
# A list of all clinical trial tools defined in this file.
# Exported for graph_builder.py to assign to specific agents.
#
# Which agents use which tools:
#   missing_results_agent  → check_results_posted, fetch_study_details
#   broken_promises_agent  → get_study_amendments, fetch_study_details
#   track_record_agent     → fetch_study_details, search_studies_by_condition
#   pattern_finder_agent   → search_studies_by_condition, fetch_study_details
#   timeline_agent         → fetch_study_details, check_results_posted
##############################################################################

ALL_CLINICAL_TOOLS = [
    fetch_study_details,
    search_studies_by_condition,
    check_results_posted,
    get_study_amendments,
]
# Exported list — graph_builder.py imports this and assigns
# specific tools to each agent based on what that agent needs.