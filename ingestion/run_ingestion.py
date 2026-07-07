##############################################################################
# ingestion/run_ingestion.py
#
# PURPOSE:
#   This is the file students actually RUN. Everything we built so far —
#   clinical_trials_client.py, pubmed_client.py, document_parser.py,
#   gcs_store.py — was just preparation. None of those files did
#   anything by themselves. This file is where it all comes together
#   and real data actually starts flowing.
#
# THINK OF THIS FILE LIKE A KITCHEN:
#   clinical_trials_client.py = the person who goes shopping for ingredients
#   pubmed_client.py          = the person who goes shopping at a different store
#   document_parser.py        = the person who washes and chops everything
#   gcs_store.py               = the person who puts everything in the fridge
#   run_ingestion.py           = YOU, the head chef, who tells everyone
#                                 what to do and in what order
#
# WHAT THIS FILE ACTUALLY DOES, STEP BY STEP:
#   1. Pick a list of medical conditions to search for
#      (example: "diabetes", "cancer", "heart disease")
#   2. For each condition, ask ClinicalTrials.gov for matching studies
#   3. Save the RAW version of every study to Google Cloud Storage
#   4. Clean up each study using document_parser.py
#   5. Save the CLEANED version to Google Cloud Storage too
#   6. For each study, ask PubMed if any research papers mention it
#   7. Save those papers (raw and cleaned) the same way
#   8. Print a final summary of everything that happened
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   YES. This is the one file in the ingestion folder you actually run.
#   Run it from your terminal like this:
#
#     python3 ingestion/run_ingestion.py
#
# WHAT OUTPUT TO EXPECT WHEN YOU RUN IT:
#   You will see a stream of log lines in your terminal showing:
#   - Which condition is currently being searched
#   - How many studies were found
#   - Each study being saved (raw, then cleaned)
#   - PubMed papers being searched for each study
#   - A final summary at the bottom with total counts
#
#   A full run with a handful of conditions usually takes a few
#   minutes — most of the time is spent waiting on the PubMed API,
#   since we deliberately slow down requests to respect their
#   rate limit (see RATE_LIMIT_SLEEP in pubmed_client.py).
##############################################################################


import asyncio
# We need asyncio because our entire pipeline is built using
# async/await. asyncio.run() is the command that actually STARTS
# an async program — everything we built so far has been async
# functions waiting to be called, and this is where we finally
# call the very first one.

from ingestion.clinical_trials_client import ClinicalTrialsClient
from ingestion.pubmed_client import PubMedClient
from ingestion.document_parser import DocumentParser
from ingestion.gcs_store import GCSStore
# Import all four classes we built earlier.
# This file does not contain any new logic of its own for talking
# to APIs or saving files — it simply ORCHESTRATES these four
# classes, calling them in the right order.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "ingestion.run_ingestion"
# Every log line printed by THIS file will show that module path,
# so when you watch the terminal you know exactly which part
# of the pipeline is currently talking.


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — WHAT TO SEARCH FOR
#
# These are the "control knobs" for this run.
# Change these values to control how much data gets downloaded.
# We keep them at the top of the file, in plain sight, so you
# never have to dig through the code to find them.
# ─────────────────────────────────────────────────────────────

SEARCH_CONDITIONS = [
    "diabetes",
    "cancer",
    "cardiovascular disease",
]
# The list of medical conditions we will search for.
# For each condition in this list, we ask ClinicalTrials.gov
# for matching studies. Start small with just 2-3 conditions
# while testing — you can always add more once you trust the
# pipeline works correctly.

MAX_STUDIES_PER_CONDITION = 50
# How many studies to fetch PER CONDITION above.
# With 3 conditions and 50 each, we get up to 150 studies total.
# Keep this LOW (like 10-20) the very first time you ever run
# this pipeline — just to make sure everything works before
# you commit to a longer run.

MAX_PAPERS_PER_STUDY = 10
# How many PubMed papers to fetch per study.
# Remember: most studies have ZERO papers referencing them —
# this number is just a CAP, not a guarantee every study will
# have this many papers.


# ─────────────────────────────────────────────────────────────
# THE MAIN PIPELINE FUNCTION
#
# This is the "head chef" function — it calls everything else
# in the correct order and keeps track of the running totals.
# ─────────────────────────────────────────────────────────────

async def run_ingestion():
    """
    Runs the complete ingestion pipeline from start to finish.

    Downloads studies from ClinicalTrials.gov for every condition
    in SEARCH_CONDITIONS, saves them to GCS (both raw and cleaned),
    then does the same for any related PubMed papers.
    """

    logger.info("=" * 60)
    logger.info("Starting MOSAIC ingestion pipeline")
    logger.info(f"Conditions to search : {', '.join(SEARCH_CONDITIONS)}")
    logger.info(f"Max studies per condition : {MAX_STUDIES_PER_CONDITION}")
    logger.info("=" * 60)
    # "=" * 60 just prints a line of 60 equals signs.
    # This is purely cosmetic — it makes the start and end of
    # each pipeline run easy to spot when scrolling through logs.

    # ── CREATE ONE INSTANCE OF EACH HELPER CLASS ──────────────
    parser = DocumentParser()
    store  = GCSStore()
    # We create ONE parser and ONE store object that gets reused
    # for every single study and paper in this run.
    # No need to create a new one each time — these classes do
    # not "remember" anything between calls, so reusing them is
    # both safe and more efficient (avoids unnecessary setup).

    # ── RUNNING TOTALS — TO REPORT AT THE END ─────────────────
    total_studies = 0
    total_papers  = 0
    # We start both counters at zero and increase them as we go,
    # so we can print an honest final summary at the end —
    # showing exactly how much real data was downloaded.

    # ── OPEN BOTH API CLIENTS ──────────────────────────────────
    async with ClinicalTrialsClient() as ct_client:
        async with PubMedClient() as pubmed_client:
            # "async with" opens BOTH clients and guarantees they
            # get closed properly afterwards — even if something
            # goes wrong partway through. We nest them because
            # we need BOTH open at the same time throughout this run.

            # ── LOOP THROUGH EVERY CONDITION ───────────────────
            for condition in SEARCH_CONDITIONS:

                # ── STEP 1: SEARCH CLINICALTRIALS.GOV ──────────
                logger.info(f"Fetching studies | condition={condition}")

                raw_studies = await ct_client.search_studies(
                    condition=condition,
                    max_results=MAX_STUDIES_PER_CONDITION,
                )
                # This single line is doing a LOT of work behind
                # the scenes — paginating through multiple pages,
                # retrying on network failures, applying our
                # custom headers to bypass bot protection.
                # All of that complexity is HIDDEN inside the
                # ClinicalTrialsClient class — this file just
                # calls one simple method and gets a clean list back.

                logger.info(
                    f"Fetched {len(raw_studies)} studies | "
                    f"condition={condition}"
                )

                # ── STEP 2: CLEAN EVERY STUDY WE GOT BACK ──────
                parsed_studies = parser.parse_studies(raw_studies)
                # Send the whole batch of raw studies to our parser.
                # It returns only the ones that parsed successfully —
                # any broken records are silently skipped (and logged).

                # ── STEP 3: SAVE EACH STUDY — RAW, THEN CLEANED ─
                for study in parsed_studies:

                    # First, find the matching RAW version of this
                    # study so we can save it untouched.
                    raw_match = next(
                        (r for r in raw_studies
                         if r.get("protocolSection", {})
                         .get("identificationModule", {})
                         .get("nctId") == study.nct_id),
                        None
                    )
                    # This line looks scary but it is simply asking:
                    # "out of all the raw studies, find the ONE whose
                    #  nctId matches this cleaned study's nct_id."
                    # next(..., None) returns the first match found,
                    # or None if somehow nothing matched.

                    if raw_match:
                        await store.save_raw_study(
                            nct_id=study.nct_id,
                            data=raw_match,
                        )
                        # Save the untouched original — our safety copy.

                    await store.save_parsed_study(study)
                    # Save the cleaned, structured version too.

                    total_studies += 1
                    # Increment our running counter by one.

                    # ── STEP 4: SEARCH PUBMED FOR THIS STUDY ───
                    papers = await pubmed_client.fetch_papers_for_trial(
                        nct_id=study.nct_id,
                        max_results=MAX_PAPERS_PER_STUDY,
                    )
                    # For THIS specific study, ask PubMed: "has anyone
                    # published a paper that references this trial?"
                    # Most of the time the answer will be "no papers
                    # found" — and that is completely normal.

                    parsed_papers = parser.parse_papers(papers)
                    # Clean up whatever papers came back, same as
                    # we did for studies above.

                    # ── STEP 5: SAVE EACH PAPER — RAW, THEN CLEANED ─
                    for paper in parsed_papers:
                        await store.save_raw_paper(
                            pmid=paper.pmid,
                            data=paper.model_dump(),
                        )
                        # Note: for papers, the "raw" version we save
                        # here is actually the already-somewhat-cleaned
                        # dict from pubmed_client.py (since that file
                        # already had to parse XML into a dict).
                        # It is still "raw" relative to our final
                        # ParsedPaper Pydantic object below.

                        await store.save_parsed_paper(paper)

                        total_papers += 1

    # ── FINAL SUMMARY ──────────────────────────────────────────
    # This code runs AFTER both "async with" blocks above have
    # closed — meaning both API clients have been safely shut down
    # before we print our final report.

    logger.info("=" * 60)
    logger.info("Ingestion complete")
    logger.info(f"Studies saved : {total_studies}")
    logger.info(f"Papers saved  : {total_papers}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
#
# This is the part of the file that actually RUNS when you type:
#   python3 ingestion/run_ingestion.py
#
# Everything above this point just DEFINES functions and classes —
# none of it executes until Python reaches this final block.
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This is a standard Python pattern.
    # __name__ is automatically set to "__main__" ONLY when this
    # file is run directly (python3 run_ingestion.py).
    # If this file is ever IMPORTED by another file instead of run
    # directly, __name__ would be "ingestion.run_ingestion" instead,
    # and this block would be skipped — preventing the pipeline
    # from accidentally running just because something imported it.

    asyncio.run(run_ingestion())
    # asyncio.run() is what actually KICKS OFF our async pipeline.
    # Everything we built — every "async def" function, every
    # "await" keyword — has been waiting for this single line.
    # This is the ignition key that starts the whole engine.