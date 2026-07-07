##############################################################################
# ingestion/document_parser.py
#
# PURPOSE:
#   This file cleans up the raw, messy data that comes back from the
#   ClinicalTrials.gov and PubMed APIs.
#   The APIs return deeply nested, inconsistent JSON. This file
#   extracts only the fields we actually need and puts them into
#   clean, predictable, typed Python objects.
#
# WHY THIS FILE EXISTS — THE BOUNDARY PRINCIPLE:
#   Think of this file as a border checkpoint.
#   Messy data comes in from outside (the APIs).
#   Clean data goes out to the rest of our system.
#   Everything AFTER this file — chunking, embedding, agents — only
#   ever sees the clean version. They never have to deal with the
#   API's confusing nested structure.
#
#   This matters because if ClinicalTrials.gov changes their API
#   tomorrow, we only need to fix THIS ONE FILE. Nothing else in
#   the entire system needs to change.
#
# WHAT IS PYDANTIC AND WHY WE USE IT HERE:
#   Pydantic lets us define a "shape" for our data using a class.
#   Once defined, Pydantic automatically checks that every piece
#   of data matches that shape — right types, right fields.
#   If something is wrong, Pydantic raises a clear error immediately
#   instead of causing a confusing crash somewhere else later.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No — this file defines classes and a parser. It gets imported
#   and used by run_ingestion.py. Do not run it directly.
#
# HOW OTHER FILES USE THIS:
#   from ingestion.document_parser import DocumentParser
#
#   parser = DocumentParser()
#   clean_study = parser.parse_study(raw_study_dict)
#   clean_paper = parser.parse_paper(raw_paper_dict)
##############################################################################


from datetime import datetime
# datetime gives us the current date and time.
# We use it to stamp every parsed record with when it was processed.

from typing import Any
# Any is a type hint meaning "this can be any Python type".
# We use it for the raw incoming data since its structure varies.

from pydantic import BaseModel, Field
# BaseModel is Pydantic's base class for defining a data shape.
# Every class we build below inherits from BaseModel.
# Field lets us add a default value or extra metadata to a field.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "ingestion.document_parser"


# ─────────────────────────────────────────────────────────────
# INTERNAL DATA SHAPES (SCHEMAS)
#
# These two classes define exactly what a "study" and a "paper"
# look like INSIDE our system — after cleaning.
# Every other file in MOSAIC works with these clean shapes,
# never with the raw API data directly.
# ─────────────────────────────────────────────────────────────

class ParsedStudy(BaseModel):
    """
    A clinical trial study, cleaned and structured.

    This is what a "study" means everywhere else in our codebase.
    The chunker reads this. The vector store reads this.
    The agents reason about this. Nobody touches raw API data
    except this one file.
    """

    nct_id: str
    # The unique ID ClinicalTrials.gov assigns to every study.
    # Format: "NCT" followed by 8 digits. Example: "NCT04788680"
    # This is our primary key — every study has exactly one.

    title: str
    # The official name of the study.
    # Example: "A Study of Semaglutide in Adults With Type 2 Diabetes"

    sponsor: str
    # Who is running this study — a pharma company, university,
    # hospital, or government agency.
    # Example: "Novo Nordisk A/S"

    phase: str
    # Which stage of testing this study represents.
    # PHASE1 = small group, mainly testing safety
    # PHASE2 = larger group, testing if it actually works
    # PHASE3 = large scale, final check before approval
    # PHASE4 = monitoring after the drug is already approved
    # NA     = not applicable, e.g. observational studies

    status: str
    # The current state of the study.
    # RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, TERMINATED, etc.

    conditions: list[str]
    # The medical conditions this study is investigating.
    # Example: ["Type 2 Diabetes", "Obesity"]

    interventions: list[str]
    # The drugs, devices, or procedures being tested.
    # Example: ["Semaglutide 2.4mg", "Placebo"]

    primary_outcome: str
    # The single most important thing this study promised to measure
    # before it started. This is the field our Broken Promises agent
    # cares about most — did the study still measure this at the end?
    # Example: "Change in HbA1c at 26 weeks"

    secondary_outcomes: list[str]
    # Additional things the study measured beyond the primary outcome.

    start_date: str
    # When the study began enrolling participants. Format: "YYYY-MM"

    completion_date: str
    # When the study finished or is expected to finish. Format: "YYYY-MM"

    results_posted: bool
    # Whether the sponsor has posted results to ClinicalTrials.gov.
    # If this is False AND status is COMPLETED, that is a signal
    # our Missing Results agent looks for.

    enrollment: int
    # How many participants were enrolled, or planned to be enrolled.

    protocol_amendments: list[dict[str, Any]]
    # Every time the study design was officially changed mid-study.
    # Multiple amendments can hint at instability in the study design.

    raw_data: dict[str, Any]
    # The complete original API response, kept exactly as received.
    # We keep this "just in case" — if we ever need a field we did
    # not extract above, it is still here. Nothing is ever thrown away.

    parsed_at: str
    # The timestamp of when this record was cleaned.
    # Useful later for knowing how fresh the data is.


class ParsedPaper(BaseModel):
    """
    A PubMed research paper, cleaned and structured.

    Same idea as ParsedStudy — this is what a "paper" means
    everywhere else in our codebase.
    """

    pmid: str
    # PubMed's unique ID for this paper. Example: "38234567"

    title: str
    abstract: str
    journal: str
    pub_date: str
    authors: list[str]

    nct_ids_referenced: list[str]
    # Which clinical trials this paper mentions.
    # Used to link a paper back to the study it discusses.

    source: str = "pubmed"
    # Always "pubmed" — tags this record so downstream code knows
    # it came from a research paper, not a clinical trial filing.

    word_count: int
    # Roughly how many words are in the abstract.
    # A very short abstract (under 50 words) often signals that
    # the paper does not have much useful detail for our agents.

    parsed_at: str


# ─────────────────────────────────────────────────────────────
# THE PARSER CLASS
#
# This class does the actual work of turning messy raw data
# into the clean ParsedStudy and ParsedPaper shapes above.
# It is "stateless" — it does not remember anything between calls.
# You can create one DocumentParser and reuse it for everything.
# ─────────────────────────────────────────────────────────────

class DocumentParser:
    """
    Converts raw API data into clean ParsedStudy and ParsedPaper objects.

    Usage:
        parser = DocumentParser()
        study = parser.parse_study(raw_study_dict)
        paper = parser.parse_paper(raw_paper_dict)
    """

    # ── PARSE ONE STUDY ───────────────────────────────────────

    def parse_study(self, raw: dict[str, Any]) -> ParsedStudy | None:
        """
        Cleans one raw ClinicalTrials.gov study record.

        The ClinicalTrials.gov API nests everything very deeply —
        a field like "title" might be 4 or 5 levels deep inside
        the raw dictionary. This method digs through that nesting
        and pulls out only what we need.

        Args:
            raw: One raw study dictionary, exactly as the API returned it.

        Returns:
            A ParsedStudy if everything worked.
            None if something essential was missing or broken.
        """

        try:
            # ── NAVIGATE THE NESTED STRUCTURE ──────────────────
            # The API wraps almost everything inside "protocolSection".
            # Think of this as opening a series of boxes inside boxes.
            protocol = raw.get("protocolSection", {})

            id_module          = protocol.get("identificationModule", {})
            status_module      = protocol.get("statusModule", {})
            sponsor_module     = protocol.get("sponsorCollaboratorsModule", {})
            conditions_module  = protocol.get("conditionsModule", {})
            design_module      = protocol.get("designModule", {})
            outcomes_module    = protocol.get("outcomesModule", {})
            interventions_mod  = protocol.get("armsInterventionsModule", {})
            # Each "module" is one box inside the bigger box.
            # We open each one we need and store it in a short variable
            # so the rest of this method stays readable.

            results_section = raw.get("resultsSection", {})
            has_results     = bool(results_section)
            # resultsSection is a SEPARATE top-level box, outside
            # protocolSection. If it exists and is not empty,
            # the sponsor has posted results for this study.
            # bool({}) is False, bool({"some": "data"}) is True.

            # ── EXTRACT THE NCT ID FIRST ───────────────────────
            nct_id = id_module.get("nctId", "")
            if not nct_id:
                # Every study MUST have an NCT ID — it is the primary key.
                # If it is missing, this record is useless to us.
                # We log it and skip this study rather than crashing.
                logger.warning("Study is missing its NCT ID — skipping")
                return None

            # ── EXTRACT THE TITLE ──────────────────────────────
            title = (
                id_module.get("officialTitle")
                or id_module.get("briefTitle")
                or ""
            )
            # Try the full official title first.
            # Some studies only have a short "brief" title — use that
            # as a fallback. The "or" chain tries each option in order
            # until one of them is not empty.

            # ── EXTRACT THE SPONSOR ────────────────────────────
            sponsor = (
                sponsor_module
                .get("leadSponsor", {})
                .get("name", "Unknown Sponsor")
            )
            # Chain of .get() calls — each one safely returns an empty
            # dict {} if the key is missing, so the NEXT .get() never
            # crashes trying to call .get() on something that is None.

            # ── EXTRACT THE PHASE ──────────────────────────────
            phase = design_module.get("phases", ["NA"])
            phase = phase[0] if phase else "NA"
            # "phases" comes back as a list — sometimes a study lists
            # two phases together like ["PHASE1", "PHASE2"].
            # We take the first one as our single phase value.

            # ── EXTRACT STATUS, CONDITIONS, INTERVENTIONS ──────
            status = status_module.get("overallStatus", "UNKNOWN")

            conditions = conditions_module.get("conditions", [])

            interventions = [
                i.get("name", "")
                for i in interventions_mod.get("interventions", [])
                if i.get("name")
            ]
            # List comprehension: go through every intervention entry,
            # pull out its "name" field, and only keep ones that
            # actually have a name (skip empty/broken entries).

            # ── EXTRACT THE PRIMARY OUTCOME ────────────────────
            primary_outcomes_list = outcomes_module.get("primaryOutcomes", [])
            primary_outcome = (
                primary_outcomes_list[0].get("measure", "")
                if primary_outcomes_list
                else ""
            )
            # A study can technically list more than one primary outcome,
            # but in practice the first one is the main one.
            # The "if primary_outcomes_list else" guards against an
            # empty list — calling [0] on an empty list would crash.

            secondary_outcomes = [
                o.get("measure", "")
                for o in outcomes_module.get("secondaryOutcomes", [])
                if o.get("measure")
            ]

            # ── EXTRACT THE DATES ───────────────────────────────
            start_date = (
                status_module
                .get("startDateStruct", {})
                .get("date", "")
            )

            completion_date = (
                status_module
                .get("primaryCompletionDateStruct", {})
                .get("date", "")
                or status_module
                .get("completionDateStruct", {})
                .get("date", "")
            )
            # Try "primary completion date" first — this is the date
            # the main outcome was actually measured.
            # Fall back to the general "completion date" if the
            # primary one is not available.

            # ── EXTRACT ENROLLMENT NUMBER ──────────────────────
            enrollment_info = design_module.get("enrollmentInfo", {})
            enrollment = enrollment_info.get("count", 0)
            try:
                enrollment = int(enrollment)
            except (ValueError, TypeError):
                enrollment = 0
            # Sometimes "count" comes back as a string like "1200"
            # instead of a number. We convert it to int safely —
            # if conversion fails for any reason, default to 0
            # rather than crashing the whole pipeline.

            # ── EXTRACT PROTOCOL AMENDMENTS ────────────────────
            annotations      = raw.get("annotationSection", {})
            amendment_module = annotations.get("annotationModule", {})
            amendments       = amendment_module.get("unpostedAnnotation", {})

            protocol_amendments = []
            if amendments:
                protocol_amendments = [
                    {
                        "date":        amendments.get("unpostedResponsibleParty", ""),
                        "description": str(amendments),
                    }
                ]
            # The amendment data in the API is structured inconsistently
            # across different studies. We capture whatever is there
            # in a simple form — our agents can still reason about it
            # even in this rough shape.

            # ── BUILD THE FINAL CLEAN OBJECT ───────────────────
            return ParsedStudy(
                nct_id=nct_id,
                title=title,
                sponsor=sponsor,
                phase=phase,
                status=status,
                conditions=conditions,
                interventions=interventions,
                primary_outcome=primary_outcome,
                secondary_outcomes=secondary_outcomes,
                start_date=start_date,
                completion_date=completion_date,
                results_posted=has_results,
                enrollment=enrollment,
                protocol_amendments=protocol_amendments,
                raw_data=raw,
                parsed_at=datetime.utcnow().isoformat(),
            )
            # Pydantic checks every field here matches the type
            # we declared in ParsedStudy above. If something is
            # the wrong type, Pydantic raises a clear error right now —
            # not silently somewhere downstream.

        except Exception as e:
            # Catch ANY unexpected error during parsing.
            # We try to get the NCT ID for the error log even though
            # parsing failed, so we know WHICH study had the problem.
            nct_id = raw.get("protocolSection", {}).get(
                "identificationModule", {}
            ).get("nctId", "UNKNOWN")
            logger.error(
                f"Failed to parse study | nct_id={nct_id} | error={e}"
            )
            return None
            # Return None — the caller skips this one study and
            # continues processing the rest. One bad record should
            # never stop the entire ingestion run.

    # ── PARSE MANY STUDIES AT ONCE ────────────────────────────

    def parse_studies(
        self,
        raw_studies: list[dict[str, Any]],
    ) -> list[ParsedStudy]:
        """
        Parses a whole list of raw studies in one call.
        Any study that fails to parse is skipped — not fatal.

        Args:
            raw_studies: List of raw study dicts from the API.

        Returns:
            List of successfully parsed ParsedStudy objects.
            Failed studies are simply not included in the result.
        """

        parsed = []
        failed = 0

        for raw in raw_studies:
            study = self.parse_study(raw)
            if study:
                parsed.append(study)
            else:
                failed += 1
            # We loop through every raw study, try to parse it,
            # and either keep it or count it as failed.
            # This way one broken study record never stops the
            # other 143 from being processed successfully.

        logger.info(
            f"Parsed studies | "
            f"success={len(parsed)} | "
            f"failed={failed} | "
            f"total={len(raw_studies)}"
        )

        return parsed

    # ── PARSE ONE PAPER ───────────────────────────────────────

    def parse_paper(self, raw: dict[str, Any]) -> ParsedPaper | None:
        """
        Cleans one raw PubMed paper record.

        PubMed papers are simpler than studies — the pubmed_client.py
        file already flattened the XML into a reasonably clean dict.
        This method does the final cleanup and builds the typed object.

        Args:
            raw: One raw paper dictionary from pubmed_client.py.

        Returns:
            A ParsedPaper if everything worked.
            None if something went wrong.
        """

        try:
            abstract = raw.get("abstract", "")
            word_count = len(abstract.split()) if abstract else 0
            # split() breaks the abstract into words by whitespace.
            # len() counts how many words there are.
            # This is an approximate count — good enough for our purpose,
            # which is just spotting unusually short/empty abstracts.

            return ParsedPaper(
                pmid=raw.get("pmid", ""),
                title=raw.get("title", ""),
                abstract=abstract,
                journal=raw.get("journal", ""),
                pub_date=raw.get("pub_date", ""),
                authors=raw.get("authors", []),
                nct_ids_referenced=raw.get("nct_ids_referenced", []),
                source="pubmed",
                word_count=word_count,
                parsed_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error(
                f"Failed to parse paper | "
                f"pmid={raw.get('pmid', 'UNKNOWN')} | "
                f"error={e}"
            )
            return None

    # ── PARSE MANY PAPERS AT ONCE ──────────────────────────────

    def parse_papers(
        self,
        raw_papers: list[dict[str, Any]],
    ) -> list[ParsedPaper]:
        """
        Parses a whole list of raw papers in one call.
        Same pattern as parse_studies — failures are skipped, not fatal.

        Args:
            raw_papers: List of raw paper dicts from pubmed_client.py.

        Returns:
            List of successfully parsed ParsedPaper objects.
        """

        parsed = []
        failed = 0

        for raw in raw_papers:
            paper = self.parse_paper(raw)
            if paper:
                parsed.append(paper)
            else:
                failed += 1

        logger.info(
            f"Parsed papers | "
            f"success={len(parsed)} | "
            f"failed={failed} | "
            f"total={len(raw_papers)}"
        )

        return parsed