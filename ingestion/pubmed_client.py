##############################################################################
# ingestion/pubmed_client.py
#
# PURPOSE:
#   This file talks to PubMed — a free database run by the US government
#   that stores millions of published medical research papers.
#   It finds and downloads research papers that reference specific
#   clinical trials from ClinicalTrials.gov.
#
# WHY WE NEED THIS:
#   ClinicalTrials.gov tells us what a study PROMISED to measure.
#   PubMed tells us what researchers ACTUALLY PUBLISHED about it.
#   The gap between those two things is where signals live.
#   Example: A trial files "no serious side effects observed"
#            but three published papers discuss concerning safety events.
#            That gap is a signal worth flagging.
#
# HOW PUBMED SEARCH WORKS — TWO STEPS:
#   PubMed does not let you search and get full details in one call.
#   It requires two separate API calls:
#
#   Step 1 — esearch:
#     Send a search query → get back a list of paper IDs
#     Example: search for "NCT04788680" → get back ["38234567", "37891234"]
#
#   Step 2 — efetch:
#     Send those paper IDs → get back full paper details
#     Example: send ["38234567", "37891234"] → get title, abstract, authors
#
#   This two-step design is how PubMed's eUtils API works.
#   We cannot skip step 1 and jump straight to step 2.
#
# RATE LIMITING — IMPORTANT:
#   PubMed allows 3 requests per second without an API key.
#   If we send requests faster than that, they get blocked.
#   We add a 400ms sleep between requests to stay safely under the limit.
#   400ms = 2.5 requests per second — safely below the 3/second limit.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No — same as clinical_trials_client.py, this is a client class.
#   It gets imported and used by run_ingestion.py.
#   Do not run it directly.
#
# HOW OTHER FILES USE THIS:
#   from ingestion.pubmed_client import PubMedClient
#
#   async with PubMedClient() as client:
#       papers = await client.fetch_papers_for_trial("NCT04788680")
##############################################################################


import asyncio
import httpx
from typing import Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

BASE_URL         = settings.pubmed_base_url
REQUEST_TIMEOUT  = 30
MAX_RETRIES      = 3
FETCH_BATCH_SIZE = 20
RATE_LIMIT_SLEEP = 0.4


# ─────────────────────────────────────────────────────────────
# CLIENT CLASS
# ─────────────────────────────────────────────────────────────

class PubMedClient:
    """
    A client for downloading research papers from PubMed.

    Works in two steps for every search:
    1. esearch — find paper IDs matching our query
    2. efetch  — get full details for those paper IDs

    Always use with "async with" for proper connection management:

        async with PubMedClient() as client:
            papers = await client.fetch_papers_for_trial("NCT04788680")
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "PubMedClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            headers={"Accept": "application/json"},
        )
        logger.info("PubMed client opened")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            logger.info("PubMed client closed")

    # ── CORE METHOD: FETCH PAPERS FOR ONE TRIAL ───────────────

    async def fetch_papers_for_trial(
        self,
        nct_id: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fetches all PubMed papers that reference a specific clinical trial.

        Handles the two-step process internally:
        Step 1: esearch — find paper IDs for this NCT ID
        Step 2: efetch  — get full details for those paper IDs

        Args:
            nct_id:      The clinical trial ID to search for.
            max_results: Maximum papers to return.

        Returns:
            List of paper dictionaries with title, abstract, authors etc.
            Empty list if no papers found or request failed.
        """

        logger.info(f"Fetching PubMed papers | nct_id={nct_id}")

        paper_ids = await self._search_paper_ids(
            nct_id=nct_id,
            max_results=max_results,
        )

        if not paper_ids:
            logger.info(f"No PubMed papers found | nct_id={nct_id}")
            return []

        logger.info(f"Found {len(paper_ids)} paper IDs | nct_id={nct_id}")

        papers = await self._fetch_paper_details(paper_ids=paper_ids)

        logger.info(
            f"PubMed fetch complete | "
            f"nct_id={nct_id} | "
            f"papers_returned={len(papers)}"
        )

        return papers

    # ── CORE METHOD: FETCH PAPERS FOR MULTIPLE TRIALS ─────────

    async def fetch_papers_for_trials(
        self,
        nct_ids: list[str],
        max_per_trial: int = 20,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetches papers for a list of clinical trials.
        Processes trials one at a time with a sleep between each
        to respect PubMed's rate limits.

        Args:
            nct_ids:       List of NCT IDs to process.
            max_per_trial: Maximum papers per trial.

        Returns:
            Dictionary mapping each NCT ID to its list of papers.
        """

        results: dict[str, list[dict[str, Any]]] = {}

        for i, nct_id in enumerate(nct_ids):
            logger.info(
                f"Processing trial {i + 1}/{len(nct_ids)} | "
                f"nct_id={nct_id}"
            )

            papers = await self.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=max_per_trial,
            )

            results[nct_id] = papers

            if i < len(nct_ids) - 1:
                await asyncio.sleep(RATE_LIMIT_SLEEP)

        return results

    # ── PRIVATE METHOD: SEARCH FOR PAPER IDs (esearch) ────────

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.ConnectError)
        ),
    )
    async def _search_paper_ids(
        self,
        nct_id: str,
        max_results: int,
    ) -> list[str]:
        """
        Calls PubMed esearch to get paper IDs for a given NCT ID.

        Args:
            nct_id:      The NCT ID to search for in PubMed.
            max_results: Maximum number of paper IDs to return.

        Returns:
            List of PubMed paper ID strings.
            Empty list if no papers found or request failed.
        """

        try:
            response = await self._client.get(
                f"{BASE_URL}/esearch.fcgi",
                params={
                    "db":         "pubmed",
                    "term":       f"{nct_id}[si]",
                    "retmax":     max_results,
                    "retmode":    "json",
                    "usehistory": "n",
                },
            )

            response.raise_for_status()

            data    = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            await asyncio.sleep(RATE_LIMIT_SLEEP)

            return id_list

        except httpx.TimeoutException:
            logger.warning(
                f"Timeout searching PubMed | nct_id={nct_id} — retrying..."
            )
            raise

        except httpx.ConnectError:
            logger.warning(
                f"Connection error searching PubMed | nct_id={nct_id} — retrying..."
            )
            raise

        except Exception as e:
            logger.error(
                f"Failed to search PubMed | nct_id={nct_id} | error={e}"
            )
            return []

    # ── PRIVATE METHOD: FETCH PAPER DETAILS (efetch) ──────────

    async def _fetch_paper_details(
        self,
        paper_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Calls PubMed efetch to get full paper details for a list of IDs.
        Processes IDs in batches of FETCH_BATCH_SIZE (20).

        Args:
            paper_ids: List of PubMed paper IDs from _search_paper_ids.

        Returns:
            List of parsed paper dictionaries.
        """

        all_papers: list[dict[str, Any]] = []

        batches = [
            paper_ids[i: i + FETCH_BATCH_SIZE]
            for i in range(0, len(paper_ids), FETCH_BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Fetching paper details | "
                f"batch={batch_num + 1}/{len(batches)} | "
                f"papers_in_batch={len(batch)}"
            )

            batch_papers = await self._fetch_batch(paper_ids=batch)
            all_papers.extend(batch_papers)

            if batch_num < len(batches) - 1:
                await asyncio.sleep(RATE_LIMIT_SLEEP)

        return all_papers

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.ConnectError)
        ),
    )
    async def _fetch_batch(
        self,
        paper_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Fetches full details for one batch of paper IDs using efetch.
        efetch returns XML — we parse it into Python dictionaries here.

        Args:
            paper_ids: One batch of PubMed IDs to fetch.

        Returns:
            List of parsed paper dictionaries for this batch.
        """

        try:
            response = await self._client.get(
                f"{BASE_URL}/efetch.fcgi",
                params={
                    "db":      "pubmed",
                    "id":      ",".join(paper_ids),
                    "retmode": "xml",
                    "rettype": "abstract",
                },
            )

            response.raise_for_status()

            papers = self._parse_xml_response(xml_text=response.text)

            await asyncio.sleep(RATE_LIMIT_SLEEP)

            return papers

        except httpx.TimeoutException:
            logger.warning("Timeout fetching paper batch — retrying...")
            raise

        except httpx.ConnectError:
            logger.warning("Connection error fetching paper batch — retrying...")
            raise

        except Exception as e:
            logger.error(f"Failed to fetch paper batch | error={e}")
            return []

    # ── PRIVATE METHOD: PARSE XML RESPONSE ────────────────────

    def _parse_xml_response(self, xml_text: str) -> list[dict[str, Any]]:
        """
        Parses the XML response from PubMed efetch into a list of
        clean Python dictionaries.

        Args:
            xml_text: The raw XML string from the efetch response.

        Returns:
            List of paper dictionaries with standardised fields.
        """

        import xml.etree.ElementTree as ET

        papers: list[dict[str, Any]] = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                paper = self._extract_paper_fields(article)
                if paper:
                    papers.append(paper)

        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML response | error={e}")

        return papers

    def _extract_paper_fields(self, article_element: Any) -> dict[str, Any] | None:
        """
        Extracts the fields we need from one PubmedArticle XML element.

        Args:
            article_element: One PubmedArticle XML element.

        Returns:
            Dictionary with the paper's key fields.
            None if extraction failed completely.
        """

        import xml.etree.ElementTree as ET

        def get_text(element: Any, path: str, default: str = "") -> str:
            node = element.find(path)
            return node.text.strip() if node is not None and node.text else default

        try:
            pmid  = get_text(article_element, ".//PMID")
            title = get_text(article_element, ".//ArticleTitle")

            abstract_texts = article_element.findall(".//AbstractText")
            abstract = " ".join(
                node.text.strip()
                for node in abstract_texts
                if node.text
            )

            pub_year  = get_text(article_element, ".//PubDate/Year")
            pub_month = get_text(article_element, ".//PubDate/Month", "01")
            pub_date  = f"{pub_year}-{pub_month}" if pub_year else ""

            journal = get_text(article_element, ".//Journal/Title")

            author_elements = article_element.findall(".//Author")
            authors = []
            for author in author_elements:
                last  = get_text(author, "LastName")
                first = get_text(author, "ForeName")
                if last:
                    authors.append(f"{last}, {first}".strip(", "))

            nct_ids_referenced = [
                id_elem.text.strip()
                for id_elem in article_element.findall(
                    ".//DataBankList/DataBank/AccessionNumberList/AccessionNumber"
                )
                if id_elem.text and id_elem.text.strip().startswith("NCT")
            ]

            return {
                "pmid":               pmid,
                "title":              title,
                "abstract":           abstract,
                "journal":            journal,
                "pub_date":           pub_date,
                "authors":            authors,
                "nct_ids_referenced": nct_ids_referenced,
                "source":             "pubmed",
            }

        except Exception as e:
            logger.error(
                f"Failed to extract paper fields | pmid=UNKNOWN | error={e}"
            )
            return None