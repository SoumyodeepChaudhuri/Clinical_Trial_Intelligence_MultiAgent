##############################################################################
# tools/pubmed_tools.py
#
# PURPOSE:
#   This file defines tools that let agents fetch LIVE research papers
#   directly from PubMed during an analysis run.
#
# WHY DO WE NEED PUBMED TOOLS SEPARATELY FROM search_tools.py?
#   search_tools.py searches papers that are ALREADY in our database —
#   papers we downloaded during ingestion and stored as chunks.
#
#   pubmed_tools.py fetches papers LIVE from PubMed on demand.
#
#   The difference matters for the Side Effect Checker agent:
#   It needs to find papers published AFTER our ingestion run.
#   A paper published last week about a safety concern would not
#   be in our database — but PubMed has it right now.
#
# WHICH AGENTS USE THESE TOOLS?
#   Side Effect Checker   → fetch_papers_for_trial (primary user)
#                           Compares official filings vs published papers
#   Pattern Finder        → search_pubmed_by_query
#                           Finds papers that discuss multiple trials
#   Missing Results Agent → fetch_papers_for_trial
#                           Checks if results were published in papers
#                           even though they were not posted officially
#
# IMPORTANT NOTE ON RATE LIMITS:
#   PubMed allows 3 requests per second without an API key.
#   Our PubMedClient already handles rate limiting with a 400ms sleep.
#   These tools respect that — do not call PubMed tools in rapid loops.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. Agents import these tools. Do not run directly.
##############################################################################


import json
# json.dumps() converts Python dicts to JSON strings.
# LangGraph tools must always return strings — never dicts or lists.
# GPT-4o reads the JSON string as text and extracts what it needs.

import asyncio
# asyncio bridges the sync/async gap — same pattern as clinical_tools.py.
# LangGraph calls tools synchronously, our PubMedClient is async.
# asyncio.get_event_loop().run_until_complete() solves this.

from langchain_core.tools import tool
# @tool decorator transforms a regular Python function into a tool
# that GPT-4o can call autonomously during agent reasoning.
# GPT-4o reads the function name, docstring, and parameter types
# to understand what the tool does and when to use it.

from ingestion.pubmed_client import PubMedClient
# PubMedClient is the class we built in ingestion/ that handles
# all PubMed API communication — the two-step esearch → efetch pattern,
# rate limiting, XML parsing, and retry logic.
# We reuse it here rather than reimplementing PubMed API calls.

from ingestion.document_parser import DocumentParser
# DocumentParser cleans raw PubMed responses into ParsedPaper objects.
# Same parser used during ingestion — consistent format everywhere.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "tools.pubmed_tools"


##############################################################################
# SHARED INSTANCES
#
# One DocumentParser shared across all tool calls.
# DocumentParser is stateless — safe to reuse many times.
# Created once at module level to avoid repeated object creation.
#
# PubMedClient is NOT created here — it is an async context manager
# (used with "async with") so we create it fresh inside each tool call.
##############################################################################

_parser = DocumentParser()
# One shared DocumentParser — stateless, safe to reuse.


##############################################################################
# THE ASYNC BRIDGE — IDENTICAL PATTERN TO clinical_tools.py
##############################################################################

def _run_async(coroutine):
    """
    Runs an async coroutine synchronously inside a LangGraph tool.

    WHY THIS IS NEEDED:
    LangGraph tools are called synchronously by the framework.
    PubMedClient uses async/await for non-blocking HTTP calls.
    This function bridges those two worlds using asyncio.

    Args:
        coroutine: An unawaited async function call.

    Returns:
        The result of the async function, returned synchronously.
    """

    loop = asyncio.get_event_loop()
    # get_event_loop() returns the currently running event loop.
    # There is always an event loop when running inside LangGraph.

    return loop.run_until_complete(coroutine)
    # run_until_complete() drives the coroutine to completion
    # and returns the result, all within the current thread.


##############################################################################
# TOOL 1: fetch_papers_for_trial
##############################################################################

@tool
def fetch_papers_for_trial(
    nct_id: str,
    max_papers: int = 10,
) -> str:
    """
    Fetch all published research papers that reference a specific
    clinical trial — LIVE from PubMed right now.

    This is the primary tool for the Side Effect Checker agent.
    It finds papers where authors reported their findings about
    a specific trial — then the agent compares those findings
    against what the official trial filing says.

    WHY THIS IS POWERFUL:
    Official filings are written by the sponsor.
    Published papers are written by independent researchers.
    When these two sources DISAGREE about safety or outcomes,
    that disagreement is a signal worth investigating.

    Example scenario:
    Official filing says: "No serious adverse events observed"
    Published paper says: "Three patients were hospitalised"
    → Side Effect Checker flags this as a safety gap signal.

    Args:
        nct_id:     The trial's NCT ID to search for in PubMed.
                    Example: "NCT04788680"
        max_papers: Maximum papers to fetch. Default 10.
                    Keep low — each paper adds to agent context window.
                    More than 15 papers can overwhelm the agent.

    Returns:
        JSON string with all papers found, including:
        - title, abstract, journal, authors, publication date
        - word_count (very short abstracts may indicate limited detail)
        Empty list if no papers reference this trial in PubMed.
    """

    logger.info(
        f"Tool called: fetch_papers_for_trial | "
        f"nct_id={nct_id} | max_papers={max_papers}"
    )

    async def _fetch():
        # Inner async function — contains all the async logic.
        # We define it here and run it with _run_async() below.
        # This pattern keeps async code cleanly separated from
        # the synchronous tool interface that LangGraph expects.

        async with PubMedClient() as client:
            # "async with" opens the httpx session when we enter
            # and closes it when we exit — guaranteed cleanup.

            papers = await client.fetch_papers_for_trial(
                nct_id=nct_id,
                # The NCT ID to search for in PubMed's Secondary
                # Identifier field — where authors register trials.

                max_results=max_papers,
                # Maximum papers to fetch for this trial.
            )

            return papers
            # papers is a list of raw paper dictionaries.
            # Each dict has: pmid, title, abstract, journal,
            # pub_date, authors, nct_ids_referenced.

    try:
        raw_papers = _run_async(_fetch())
        # Run the async fetch synchronously.
        # raw_papers is a list of raw dicts from PubMed.

        if not raw_papers:
            # PubMed has no papers referencing this trial.
            # This is common — most trials are never directly cited.
            # The agent reads this and notes "no published research found."
            return json.dumps({
                "nct_id":  nct_id,
                "papers":  [],
                "count":   0,
                "message": f"No published papers found on PubMed that "
                           f"reference trial {nct_id}. The trial may not "
                           "have published results in academic journals, "
                           "or results may only exist as grey literature.",
            }, indent=2)

        parsed_papers = _parser.parse_papers(raw_papers=raw_papers)
        # Clean the raw paper dictionaries into ParsedPaper objects.
        # parse_papers() processes the full list and skips any
        # that fail to parse without crashing the whole batch.

        papers_list = []
        for paper in parsed_papers:

            paper_dict = paper.model_dump()
            # model_dump() converts the Pydantic ParsedPaper object
            # to a plain Python dictionary — needed for JSON serialisation.

            papers_list.append({
                "pmid":               paper_dict["pmid"],
                # PubMed's unique paper identifier. Example: "38234567"

                "title":              paper_dict["title"],
                # The paper's full title.

                "abstract":           paper_dict["abstract"],
                # The abstract is the most important field —
                # it summarises the paper's findings.
                # The Side Effect Checker reads abstracts to find
                # safety events not reported in the official filing.

                "journal":            paper_dict["journal"],
                # Where the paper was published.
                # High-impact journals (NEJM, Lancet, JAMA) carry
                # more weight than lower-tier publications.

                "pub_date":           paper_dict["pub_date"],
                # When the paper was published.
                # Papers published AFTER trial completion are most
                # relevant — they report final results.

                "authors":            paper_dict["authors"][:5],
                # First 5 authors only — prevents long author lists
                # from consuming agent context window unnecessarily.
                # [:5] slices the list to maximum 5 items.

                "word_count":         paper_dict["word_count"],
                # Approximate word count of the abstract.
                # Very short abstracts (under 50 words) may indicate
                # limited information — agent can weight these lower.

                "nct_ids_referenced": paper_dict["nct_ids_referenced"],
                # Which clinical trials this paper mentions.
                # Useful for Pattern Finder — a paper referencing
                # multiple NCT IDs may reveal cross-trial connections.
            })

        return json.dumps({
            "nct_id": nct_id,
            "papers": papers_list,
            "count":  len(papers_list),
            # count tells the agent how many papers were found —
            # useful for reasoning: "3 papers found, all agree on safety"
            # is different from "1 paper found with safety concerns."
        }, indent=2, default=str)
        # default=str handles datetime objects and other
        # non-JSON-serialisable values by converting to strings.

    except Exception as e:
        logger.error(
            f"fetch_papers_for_trial failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "papers": [],
            "count":  0,
        })


##############################################################################
# TOOL 2: search_pubmed_by_query
##############################################################################

@tool
def search_pubmed_by_query(
    query: str,
    max_papers: int = 5,
) -> str:
    """
    Search PubMed with a free-text query and fetch matching papers.

    Use this tool when you want to find papers about a topic,
    drug, or condition — not just papers about one specific trial.

    Different from fetch_papers_for_trial which searches by NCT ID.
    This tool accepts any PubMed search query.

    Examples:
    - "semaglutide cardiovascular outcomes 2023"
    - "metformin diabetes safety adverse events"
    - "Novo Nordisk clinical trial results transparency"

    The Pattern Finder agent uses this to find papers that discuss
    multiple trials from the same sponsor — revealing systemic patterns.

    Args:
        query:      Any PubMed-compatible search query.
                    Can include drug names, conditions, author names,
                    journal names, or any combination.
        max_papers: Maximum papers to return. Default 5.
                    Keep low — each paper adds to agent context.

    Returns:
        JSON string with matching papers from PubMed.
    """

    logger.info(
        f"Tool called: search_pubmed_by_query | "
        f"query='{query[:60]}' | max_papers={max_papers}"
    )

    async def _search():
        # Inner async function for the PubMed API call.
        # Two steps as always with PubMed:
        # Step 1: esearch → get paper IDs matching the query
        # Step 2: efetch  → get full paper details for those IDs

        async with PubMedClient() as client:

            paper_ids = await client._search_paper_ids(
                nct_id=query,
                # We reuse _search_paper_ids() here with the query
                # as the search term instead of an NCT ID.
                # PubMed's esearch accepts any search string —
                # not just NCT IDs.
                # Note: We are calling a "private" method (underscore prefix).
                # This is acceptable here because we are in the tools layer
                # which is a peer module — not external code.
                # In a stricter design, PubMedClient would have a
                # public search_by_query() method. For this build,
                # reusing _search_paper_ids() avoids code duplication.

                max_results=max_papers,
            )

            if not paper_ids:
                return []
                # No papers found for this query — return empty list.

            papers = await client._fetch_paper_details(
                paper_ids=paper_ids
            )
            # Fetch full paper details for the IDs we found.
            # _fetch_paper_details() handles batching and rate limiting.

            return papers

    try:
        raw_papers = _run_async(_search())

        if not raw_papers:
            return json.dumps({
                "query":   query,
                "papers":  [],
                "count":   0,
                "message": f"No papers found on PubMed for query: '{query}'. "
                           "Try a broader search term or different keywords.",
            }, indent=2)

        parsed_papers = _parser.parse_papers(raw_papers=raw_papers)
        # Clean raw paper dicts into ParsedPaper objects.

        papers_list = []
        for paper in parsed_papers:
            paper_dict = paper.model_dump()

            papers_list.append({
                "pmid":               paper_dict["pmid"],
                "title":              paper_dict["title"],
                "abstract":           paper_dict["abstract"],
                "journal":            paper_dict["journal"],
                "pub_date":           paper_dict["pub_date"],
                "authors":            paper_dict["authors"][:5],
                "word_count":         paper_dict["word_count"],
                "nct_ids_referenced": paper_dict["nct_ids_referenced"],
                # nct_ids_referenced is especially valuable here —
                # when searching by topic rather than NCT ID,
                # knowing which trials each paper mentions lets the
                # Pattern Finder agent discover cross-trial connections.
            })

        return json.dumps({
            "query":  query,
            "papers": papers_list,
            "count":  len(papers_list),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_pubmed_by_query failed | "
            f"query={query} | error={e}"
        )
        return json.dumps({
            "query":  query,
            "error":  str(e),
            "papers": [],
            "count":  0,
        })


##############################################################################
# TOOL 3: compare_filing_vs_papers
##############################################################################

@tool
def compare_filing_vs_papers(
    nct_id: str,
    filing_summary: str,
) -> str:
    """
    Fetch papers for a trial and compare them against the official filing.

    This is the Side Effect Checker's most powerful tool.
    It fetches all published papers about a trial, then returns
    both the official filing summary AND the papers — so the agent
    can compare them and identify discrepancies.

    The agent looks for:
    - Safety events mentioned in papers but not in the filing
    - Different severity levels (filing says "mild", paper says "serious")
    - Outcomes reported in papers that differ from the primary outcome
    - Results published in papers when no official results were posted

    Args:
        nct_id:          The trial's NCT ID.
        filing_summary:  A summary of what the official filing says.
                         The agent provides this from earlier tool calls.
                         Example: "Filing reports no serious adverse events.
                                   Primary outcome was HbA1c reduction.
                                   Results posted: No."

    Returns:
        JSON string with:
        - filing_summary: what the agent passed in (echoed back)
        - papers: all published papers found on PubMed
        - comparison_note: guidance on what to look for
        - papers_count: how many papers were found
    """

    logger.info(
        f"Tool called: compare_filing_vs_papers | nct_id={nct_id}"
    )

    async def _fetch():
        async with PubMedClient() as client:
            papers = await client.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=15,
                # Fetch up to 15 papers for comparison.
                # More papers = more evidence for the agent to compare.
                # 15 is the upper limit — beyond this, the agent's
                # context window gets overwhelmed.
            )
            return papers

    try:
        raw_papers = _run_async(_fetch())
        parsed_papers = _parser.parse_papers(raw_papers=raw_papers or [])

        papers_list = []
        for paper in parsed_papers:
            paper_dict = paper.model_dump()
            papers_list.append({
                "pmid":     paper_dict["pmid"],
                "title":    paper_dict["title"],
                "abstract": paper_dict["abstract"],
                # Abstract is the key field for comparison.
                # The agent reads each abstract looking for mentions
                # of adverse events, safety concerns, or outcome data
                # that contradict or supplement the official filing.

                "journal":  paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors":  paper_dict["authors"][:3],
                # 3 authors is enough for citation context.
            })

        comparison_note = (
            "Compare the filing_summary above against each paper's abstract. "
            "Look specifically for: "
            "(1) adverse events mentioned in papers but absent from filing, "
            "(2) different severity descriptions for the same event, "
            "(3) outcome results that contradict the filing's claims, "
            "(4) results data in papers when filing shows results_posted=False."
        )
        # This note is guidance for GPT-4o — it tells the agent
        # exactly what to look for when comparing the two sources.
        # Clear instructions in tool output = better agent reasoning.
        # Without this note, the agent might just summarise the papers
        # instead of actively looking for discrepancies.

        return json.dumps({
            "nct_id":          nct_id,
            "filing_summary":  filing_summary,
            # Echo the filing summary back — the agent sees both
            # the official claims and the paper evidence in one response.
            # Having both in one JSON response means the agent does not
            # need to remember what it found in a previous tool call.

            "papers":          papers_list,
            "papers_count":    len(papers_list),
            "comparison_note": comparison_note,
            # The guidance on what discrepancies to look for.

            "has_papers": len(papers_list) > 0,
            # Simple boolean — True if any papers were found.
            # Agents can branch: if has_papers is False, no comparison
            # is possible and confidence should be lower.
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"compare_filing_vs_papers failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "papers": [],
        })


##############################################################################
# TOOL REGISTRY
#
# List of all PubMed tools defined in this file.
# Exported for graph_builder.py to assign to specific agents.
#
# Which agents use which PubMed tools:
#   side_effect_agent    → compare_filing_vs_papers (primary)
#                          fetch_papers_for_trial
#   pattern_finder_agent → search_pubmed_by_query
#                          fetch_papers_for_trial
#   missing_results_agent→ fetch_papers_for_trial
#                          (checks if results exist in papers
#                           even when not officially posted)
##############################################################################

ALL_PUBMED_TOOLS = [
    fetch_papers_for_trial,
    search_pubmed_by_query,
    compare_filing_vs_papers,
]
# graph_builder.py imports this list and assigns specific
# tools to each agent based on what that agent needs.