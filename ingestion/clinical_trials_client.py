##############################################################################
# ingestion/clinical_trials_client.py
#
# PURPOSE:
#   This is the first file that talks to the outside world.
#   It connects to ClinicalTrials.gov — a free US government database
#   that stores every registered medical research study.
#   It downloads study records and returns them as Python dictionaries.
#
# WHAT IT DOES STEP BY STEP:
#   1. Opens a connection to the ClinicalTrials.gov API
#   2. Searches for studies by condition, sponsor, or intervention
#   3. Handles pagination — the API returns 100 studies at a time
#      so we keep asking for the next page until we have enough
#   4. Handles failures — if the API is slow or drops the connection,
#      we automatically retry instead of crashing
#   5. Returns raw study data exactly as the API gave it to us
#      We never modify the data here — that is document_parser.py's job
#
# IMPORTANT DESIGN DECISION — WHY requests AND NOT httpx:
#   Most modern async Python code uses httpx for HTTP calls.
#   We tried httpx first — ClinicalTrials.gov returned 403 Forbidden.
#   The reason: ClinicalTrials.gov uses bot protection that checks
#   the TLS fingerprint of the HTTP client. httpx has a different
#   fingerprint from a real browser. requests matches closely enough.
#   This is a real production debugging decision — not a textbook choice.
#   We wrap requests inside asyncio.to_thread() to keep it async.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No — do not run this file directly.
#   This file is a CLIENT — it provides a class that other files use.
#   You run run_ingestion.py which imports and uses this client.
#   Think of this file like a car engine — you do not start the engine
#   directly, you turn the ignition key (run_ingestion.py).
#
# HOW OTHER FILES USE THIS:
#   from ingestion.clinical_trials_client import ClinicalTrialsClient
#
#   async with ClinicalTrialsClient() as client:
#       studies = await client.search_studies(
#           condition="diabetes",
#           max_results=50
#       )
##############################################################################


import asyncio
# asyncio is Python's built-in library for writing async code.
# async/await lets our program do other things while waiting for
# the API to respond — instead of just sitting there doing nothing.
# Think of it like placing a food order and doing other things
# while waiting — instead of standing at the counter staring.

import requests
# requests is the most popular Python library for making HTTP calls.
# HTTP is the protocol used to talk to web APIs — the same protocol
# your browser uses when you visit a website.
# We use requests instead of httpx because ClinicalTrials.gov
# blocks httpx at the TLS fingerprint level but accepts requests.
# This was discovered through real debugging — not a textbook decision.

from typing import Any
# Any is a type hint that means "this variable can be any Python type".
# We use it for API responses because the structure varies —
# sometimes it is a dict, sometimes a list, depending on the endpoint.

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
# tenacity is a Python library specifically built for retry logic.
# Instead of writing our own while loop with try/except and sleep(),
# tenacity gives us clean decorators that handle all of that.
#
# retry                    → the main decorator that enables retrying
# stop_after_attempt(3)    → give up after 3 failed attempts
# wait_exponential()       → wait 1s after first fail, 2s after second,
#                            4s after third — this is called exponential
#                            backoff. We give the server more time to
#                            recover before each retry attempt.
# retry_if_exception_type  → only retry on specific error types.
#                            We only retry on network errors —
#                            not on 404 or 403 which are our fault.

from config.settings import settings
# Import the settings singleton we created in config/settings.py.
# This gives us access to all environment variables in one object.
# settings.clinical_trials_base_url → the API base URL from .env
# settings.clinical_trials_page_size → how many results per page

from config.logging_config import setup_logging
# Import our centralised logging setup function.
# Every file in MOSAIC uses this same function.
# Passing __name__ means log lines show "ingestion.clinical_trials_client"
# so we always know exactly which file printed each log line.

logger = setup_logging(__name__)
# Create the logger for this specific file.
# __name__ is a Python built-in that equals the current module path.
# In this file: __name__ = "ingestion.clinical_trials_client"
# Every logger.info() call in this file will show that module path.


# ─────────────────────────────────────────────────────────────
# CONSTANTS
#
# These are fixed values used throughout this file.
# We define them at the top so they are easy to find and change.
# Never scatter magic numbers throughout your code —
# put them here with clear names explaining what they mean.
# ─────────────────────────────────────────────────────────────

BASE_URL = settings.clinical_trials_base_url
# The starting point for all API calls.
# Value: "https://clinicaltrials.gov/api/v2"
# Every endpoint we call is built from this base URL.
# Example: BASE_URL + "/studies" = full endpoint URL

PAGE_SIZE = settings.clinical_trials_page_size
# How many studies to request per API call.
# Value: 100 (set in .env)
# ClinicalTrials.gov maximum is 1000 but we use 100 to be safe.
# If the API is slow or we hit a rate limit, smaller pages
# are easier to retry than large ones.

REQUEST_TIMEOUT = 30
# How many seconds to wait for the API to respond.
# If the API does not respond in 30 seconds we give up on that request.
# Without a timeout, our code could hang forever waiting.
# 30 seconds is generous — ClinicalTrials.gov usually responds in 1-2s.

MAX_RETRIES = 3
# Maximum number of times to retry a failed request.
# First attempt + 2 retries = 3 total attempts before giving up.
# This handles temporary network issues without crashing the pipeline.

HEADERS = {
    "Accept": "application/json",
    # Tell the API we want the response in JSON format.
    # JSON is a text format that Python can easily convert to dicts.
    # Without this header, the API might return XML or HTML instead.

    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    # This tells the API what kind of client is making the request.
    # ClinicalTrials.gov has bot protection that blocks requests
    # without a realistic User-Agent header.
    # This header makes our request look like it came from a
    # real Mac running Chrome — which the bot protection accepts.
    # We confirmed this exact string works through testing.
}
# IMPORTANT: These headers are sent with every single API request.
# The User-Agent header is the key reason requests works and httpx does not.
# httpx sends its own User-Agent that triggers the bot protection.
# requests with this custom User-Agent bypasses it.


# ─────────────────────────────────────────────────────────────
# CLIENT CLASS
#
# A class is a blueprint for creating objects.
# ClinicalTrialsClient is a blueprint for an object that knows
# how to talk to the ClinicalTrials.gov API.
#
# We designed it as an "async context manager" — meaning you
# use it with the "async with" keyword:
#
#   async with ClinicalTrialsClient() as client:
#       studies = await client.search_studies(condition="diabetes")
#
# The "async with" pattern guarantees that the HTTP session
# is properly opened before use and properly closed after —
# even if an error occurs in the middle.
# ─────────────────────────────────────────────────────────────

class ClinicalTrialsClient:
    """
    A client for downloading study records from ClinicalTrials.gov.

    This client handles everything needed to talk to the API:
    - Opening and closing the HTTP session
    - Building the correct URL and parameters for each request
    - Handling pagination (the API gives 100 results at a time)
    - Retrying automatically when the network fails

    Always use it with "async with" to ensure proper cleanup:

        async with ClinicalTrialsClient() as client:
            studies = await client.search_studies(condition="cancer")
    """

    def __init__(self):
        self._session: requests.Session | None = None
        # _session is the HTTP connection to ClinicalTrials.gov.
        # We start it as None — it gets created in __aenter__.
        # Using a Session object means we reuse the same TCP connection
        # for multiple requests — faster than opening a new connection
        # for every single API call.
        # The underscore prefix (_session) is a Python convention
        # meaning "this is internal — do not access it from outside".

    async def __aenter__(self) -> "ClinicalTrialsClient":
        """
        Called automatically when we enter the "async with" block.
        Creates the HTTP session and sets the shared headers.

        The -> "ClinicalTrialsClient" means this method returns
        the client object itself, so we can assign it:
            async with ClinicalTrialsClient() as client:
                           ↑ __aenter__ returns this object as "client"
        """
        self._session = requests.Session()
        # Create a new requests Session.
        # A Session keeps the TCP connection open between requests.
        # This is more efficient than creating a new connection
        # for every single API call we make.

        self._session.headers.update(HEADERS)
        # Apply our custom headers to every request this session makes.
        # This means we never forget to send the User-Agent header —
        # it is set once here and applied automatically every time.

        logger.info("ClinicalTrials client opened")
        # Log that the client is ready.
        # Students will see this in the terminal when the pipeline runs.

        return self
        # Return the client object so the "async with" statement
        # can assign it to the variable after "as":
        #   async with ClinicalTrialsClient() as client:
        #                                         ↑ this is self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Called automatically when we exit the "async with" block.
        Closes the HTTP session and releases the connection.

        This runs even if an error occurred inside the "async with" block.
        That is the whole point of context managers — guaranteed cleanup.

        The three parameters (exc_type, exc_val, exc_tb) contain
        information about any exception that occurred. We do not use
        them here — we just close the session regardless.
        """
        if self._session:
            # Only close if the session was actually created.
            # Avoids errors if __aenter__ failed before creating it.
            self._session.close()
            # Close the HTTP session and release the TCP connection.
            # Without this, connections would leak — staying open
            # indefinitely and wasting server resources.

            logger.info("ClinicalTrials client closed")

    # ── CORE METHOD: SEARCH STUDIES ───────────────────────────

    async def search_studies(
        self,
        condition: str | None = None,
        # What medical condition to search for.
        # Example: "diabetes", "cancer", "heart disease"
        # Maps to the query.cond parameter in the API.
        # None means do not filter by condition.

        intervention: str | None = None,
        # What drug or treatment to search for.
        # Example: "semaglutide", "metformin", "aspirin"
        # Maps to the query.intr parameter in the API.
        # None means do not filter by intervention.

        sponsor: str | None = None,
        # Which organisation is running the study.
        # Example: "Pfizer", "NIH", "Harvard University"
        # Maps to the query.spons parameter in the API.
        # None means do not filter by sponsor.

        status: list[str] | None = None,
        # Filter by study status.
        # Common values: "COMPLETED", "RECRUITING", "ACTIVE_NOT_RECRUITING"
        # Pass a list for multiple: ["COMPLETED", "RECRUITING"]
        # None means return studies of any status.

        max_results: int = 100,
        # Maximum number of studies to return in total.
        # Default is 100 — a safe number for testing.
        # Increase to 500 or more for production runs.
        # The client handles all the pagination automatically —
        # you just say how many you want and it figures out
        # how many pages to request.

    ) -> list[dict[str, Any]]:
        """
        Searches ClinicalTrials.gov and returns a list of study records.

        This is the main method you call to get study data.
        It handles everything internally:
        - Building the search parameters
        - Fetching multiple pages until max_results is reached
        - Returning all results as a flat list of dictionaries

        Args:
            condition:   Medical condition to search for.
            intervention: Drug or treatment to search for.
            sponsor:     Organisation running the study.
            status:      List of study statuses to filter by.
            max_results: Maximum total studies to return.

        Returns:
            List of raw study dictionaries exactly as the API returned them.
            Each dictionary contains all the study fields —
            nct_id, title, sponsor, outcomes, dates, etc.
        """

        all_studies: list[dict[str, Any]] = []
        # This list accumulates studies across all pages.
        # We start empty and add to it with each page we fetch.
        # At the end, this is what we return.

        next_page_token: str | None = None
        # ClinicalTrials.gov uses cursor-based pagination.
        # After each page, the API returns a "nextPageToken" string.
        # We send that token with the next request to get the next page.
        # None means "start from the beginning — first page".
        # Think of it like a bookmark — the token tells the API
        # exactly where we left off.

        page_number = 0
        # Tracks which page we are currently fetching.
        # Only used for logging — so students can see progress.

        logger.info(
            f"Searching studies | "
            f"condition={condition} | "
            f"intervention={intervention} | "
            f"sponsor={sponsor} | "
            f"max_results={max_results}"
        )
        # Log what we are searching for before we start.
        # This appears in the terminal so students can see
        # exactly what parameters are being used.

        # ── PAGINATION LOOP ───────────────────────────────────
        while len(all_studies) < max_results:
            # Keep fetching pages until we have enough studies.
            # len(all_studies) = how many we have collected so far.
            # max_results = how many we want in total.
            # When we have enough, the while condition becomes False
            # and we stop fetching.

            page_number += 1
            # Increment page counter for logging.

            params = self._build_search_params(
                condition=condition,
                intervention=intervention,
                sponsor=sponsor,
                status=status,
                page_token=next_page_token,
            )
            # Build the query parameters for this specific page request.
            # _build_search_params is a private helper method below.
            # It returns a dictionary like:
            # {"pageSize": 100, "format": "json", "query.cond": "diabetes"}

            response_data = await self._fetch_page(params=params)
            # Make the actual API call and get the response.
            # await means: wait here until the API responds.
            # _fetch_page is a private helper method below.
            # It returns the JSON response as a Python dictionary.
            # Returns None if the request failed after all retries.

            if not response_data:
                # If _fetch_page returned None, something went wrong.
                # The error was already logged inside _fetch_page.
                # We break the loop — no point continuing if API is down.
                break

            page_studies = response_data.get("studies", [])
            # Extract the list of studies from the response.
            # The API wraps results inside a "studies" key:
            # {
            #   "studies": [{...}, {...}, {...}],
            #   "nextPageToken": "abc123"
            # }
            # .get("studies", []) safely returns [] if key is missing.

            if not page_studies:
                # If this page has no studies, we have reached the end.
                # No more data available — stop fetching.
                logger.info("No more studies available — pagination complete")
                break

            all_studies.extend(page_studies)
            # Add this page's studies to our running total.
            # extend() adds all items from page_studies into all_studies.
            # This is more efficient than += for lists.

            logger.info(
                f"Page {page_number} | "
                f"fetched={len(page_studies)} | "
                f"total so far={len(all_studies)}"
            )
            # Log progress after each page.
            # Students will see this updating in real time:
            # Page 1 | fetched=100 | total so far=100
            # Page 2 | fetched=100 | total so far=200

            next_page_token = response_data.get("nextPageToken")
            # Get the token for the next page from the response.
            # If "nextPageToken" is in the response → more pages exist.
            # If it is absent → this was the last page.

            if not next_page_token:
                # No next page token means we have reached the last page.
                # Stop the loop even if we have not hit max_results yet.
                logger.info("Last page reached — no nextPageToken in response")
                break

        all_studies = all_studies[:max_results]
        # Trim the list to exactly max_results.
        # The last page might have pushed us slightly over max_results.
        # Example: max_results=150, last page gave us 50 more = 200 total.
        # We trim back to 150.

        logger.info(
            f"Search complete | "
            f"total studies returned={len(all_studies)}"
        )
        # Log the final count — this is what students will see
        # at the end of a search to confirm how many came back.

        return all_studies
        # Return the complete list of study dictionaries.
        # Each dictionary is one study record exactly as the API returned it.
        # The caller (run_ingestion.py) receives this list.

    # ── CORE METHOD: FETCH ONE STUDY BY ID ───────────────────

    async def fetch_study(self, nct_id: str) -> dict[str, Any] | None:
        """
        Fetches the complete record for one specific study by its NCT ID.

        NCT ID is the unique identifier every study gets when it registers
        on ClinicalTrials.gov. Format: NCT followed by 8 digits.
        Example: NCT04788680

        Use this when you already know which specific study you want
        and need its full details — not for searching.

        Args:
            nct_id: The study's unique identifier. Example: "NCT04788680"

        Returns:
            A dictionary with all the study's details.
            None if the study was not found or the request failed.
        """

        logger.info(f"Fetching single study | nct_id={nct_id}")

        def _get_study():
            # This inner function makes the actual HTTP GET request.
            # We define it as a regular (non-async) function because
            # requests is synchronous — it cannot use async/await.
            # We will run this inside asyncio.to_thread() below.
            return self._session.get(
                f"{BASE_URL}/studies/{nct_id}",
                # Build the URL for this specific study.
                # Example: https://clinicaltrials.gov/api/v2/studies/NCT04788680
                timeout=REQUEST_TIMEOUT,
                # Wait maximum 30 seconds for a response.
            )

        try:
            response = await asyncio.to_thread(_get_study)
            # asyncio.to_thread() runs _get_study() in a background thread.
            # This is how we make synchronous requests work in async code.
            # The main program does not freeze while waiting —
            # it can handle other tasks while the thread fetches data.

            response.raise_for_status()
            # Check if the HTTP status code indicates an error.
            # Status 200 = success → no exception raised → continue
            # Status 404 = not found → raises HTTPError → caught below
            # Status 500 = server error → raises HTTPError → caught below

            return response.json()
            # Parse the JSON response body into a Python dictionary.
            # The API returns JSON text — .json() converts it to a dict
            # we can work with in Python.

        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"Study not found | "
                f"nct_id={nct_id} | "
                f"status={e.response.status_code}"
            )
            return None
            # Return None — the caller can handle a missing study.
            # We use WARNING not ERROR because a missing study
            # is not a system failure — it is expected sometimes.

        except Exception as e:
            logger.error(
                f"Failed to fetch study | "
                f"nct_id={nct_id} | "
                f"error={e}"
            )
            return None

    # ── PRIVATE METHOD: BUILD SEARCH PARAMETERS ───────────────

    def _build_search_params(
        self,
        condition: str | None,
        intervention: str | None,
        sponsor: str | None,
        status: list[str] | None,
        page_token: str | None,
    ) -> dict[str, Any]:
        """
        Builds the query parameter dictionary for one API request.

        The ClinicalTrials.gov API expects specific parameter names.
        This method translates our friendly Python arguments into
        the exact parameter names the API understands.

        Only includes parameters that were actually provided —
        if condition is None, we do not add query.cond to the request.

        Args:
            condition:    Medical condition filter.
            intervention: Drug/treatment filter.
            sponsor:      Sponsor organisation filter.
            status:       List of status values to filter by.
            page_token:   Cursor for the next page of results.

        Returns:
            Dictionary of query parameters ready to send to the API.
        """

        params: dict[str, Any] = {
            "pageSize": PAGE_SIZE,
            # How many results per page.
            # We always set this explicitly — never trust API defaults.
            # Default is 10 which would be very slow for 100+ studies.

            "format": "json",
            # Tell the API to return JSON format.
            # Explicit is better than relying on the default.
        }

        if condition:
            params["query.cond"] = condition
            # query.cond is the ClinicalTrials.gov v2 parameter name
            # for searching by medical condition.
            # It searches across condition names and their synonyms.
            # Example: "diabetes" also finds "Type 2 Diabetes Mellitus"

        if intervention:
            params["query.intr"] = intervention
            # query.intr searches by intervention (drug or treatment) name.
            # Example: "semaglutide" finds studies testing that drug.

        if sponsor:
            params["query.spons"] = sponsor
            # query.spons searches by sponsor organisation name.
            # Example: "Pfizer" finds all Pfizer-sponsored studies.

        if status:
            params["filter.overallStatus"] = "|".join(status)
            # filter.overallStatus filters by study status.
            # IMPORTANT: ClinicalTrials.gov v2 uses pipe | as separator.
            # ["COMPLETED", "RECRUITING"] → "COMPLETED|RECRUITING"
            # Using comma here would cause a 403 error — we discovered
            # this through debugging. Pipe is the correct separator.

        if page_token:
            params["pageToken"] = page_token
            # The cursor token from the previous page's response.
            # Sending this tells the API: "give me the page AFTER this token"
            # Without this, every request would return the first page again.

        return params

    # ── PRIVATE METHOD: FETCH ONE PAGE WITH RETRY ─────────────

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        # Stop retrying after MAX_RETRIES attempts (3 total).
        # After 3 failures we give up and return None.

        wait=wait_exponential(multiplier=1, min=1, max=8),
        # Wait between retries using exponential backoff:
        # After 1st failure: wait 1 second
        # After 2nd failure: wait 2 seconds
        # After 3rd failure: wait 4 seconds (capped at 8 seconds)
        # We give the server more time to recover with each retry.

        retry=retry_if_exception_type(
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        ),
        # Only retry on these specific error types:
        # Timeout      → server took too long to respond
        # ConnectionError → could not connect to the server
        # We do NOT retry on HTTPError (404, 403) because those are
        # our fault — retrying will not fix a wrong URL or parameter.
    )
    async def _fetch_page(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """
        Makes one GET request to the /studies endpoint.

        This is decorated with @retry from tenacity, which means
        if it fails with a Timeout or ConnectionError, tenacity
        automatically calls it again — up to MAX_RETRIES times.

        Args:
            params: The query parameters built by _build_search_params.

        Returns:
            The JSON response as a Python dictionary.
            None if all retry attempts failed.
        """

        def _get():
            # Inner function that makes the actual HTTP request.
            # Defined as a regular function because requests is synchronous.
            # We run it in a background thread via asyncio.to_thread().
            return self._session.get(
                f"{BASE_URL}/studies",
                # The endpoint URL for searching studies.
                # Full URL: https://clinicaltrials.gov/api/v2/studies

                params=params,
                # The query parameters — condition, page size, token etc.
                # requests automatically appends these to the URL:
                # /studies?pageSize=100&format=json&query.cond=diabetes

                timeout=REQUEST_TIMEOUT,
                # Wait maximum 30 seconds before giving up on this request.
            )

        try:
            response = await asyncio.to_thread(_get)
            # Run the synchronous _get() function in a background thread.
            # asyncio.to_thread() lets us use synchronous libraries
            # (like requests) inside async code without blocking
            # the entire program.
            # await means: pause here until the thread finishes,
            # but let other async tasks run in the meantime.

            response.raise_for_status()
            # Check the HTTP status code.
            # 200 OK → success, no exception
            # 4xx/5xx → raises requests.exceptions.HTTPError

            return response.json()
            # Convert the JSON response text into a Python dictionary.
            # This is what the caller receives and works with.

        except requests.exceptions.Timeout:
            logger.warning(
                f"Request timed out after {REQUEST_TIMEOUT}s — retrying..."
            )
            raise
            # Re-raise the exception so tenacity knows to retry.
            # If we catch and swallow the exception, tenacity thinks
            # the function succeeded and does not retry.

        except requests.exceptions.ConnectionError:
            logger.warning("Connection error — retrying...")
            raise
            # Re-raise for the same reason — let tenacity retry.

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error from API | "
                f"status={e.response.status_code} | "
                f"url={e.response.url}"
            )
            return None
            # Return None for HTTP errors — do not retry these.
            # A 403 or 404 will not be fixed by retrying.
            # The error is already logged so the caller knows what happened.

        except Exception as e:
            logger.error(
                f"Unexpected error fetching page | "
                f"error={e}"
            )
            return None
            # Catch any other unexpected error.
            # Return None so the pipeline continues with other conditions
            # rather than crashing completely.