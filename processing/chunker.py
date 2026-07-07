##############################################################################
# processing/chunker.py
#
# PURPOSE:
#   This file takes a long study document and breaks it into smaller,
#   overlapping pieces of text called "chunks".
#   Each chunk is then sent to OpenAI in the next step (embedder.py)
#   to get its vector embedding — a mathematical representation of
#   what that piece of text MEANS.
#
# WHY DO WE NEED TO CHUNK AT ALL?
#   OpenAI's embedding model has a token limit — it cannot process
#   an entire study document in one shot. A single clinical trial
#   record can easily be 2000-5000 words long. We must break it
#   into smaller pieces first.
#
#   But there is a deeper reason too. If we embed the WHOLE document
#   as one giant chunk, the embedding becomes a blurry average of
#   everything in it. When an agent searches for "sponsor never posted
#   results", a whole-document embedding might miss that signal because
#   it is diluted by all the other content.
#
#   Smaller, focused chunks → sharper, more precise embeddings →
#   agents find exactly what they are looking for.
#
# WHY DO CHUNKS OVERLAP?
#   Imagine cutting a book into pages. If an important sentence
#   happens to fall RIGHT at the cut point — half on page 10,
#   half on page 11 — both pages miss the complete thought.
#
#   Overlapping solves this. Each chunk shares 50 words with the
#   next chunk. So no sentence ever gets cut in half and lost.
#   The overlap is like a safety net at every boundary.
#
# WHAT IS A TextChunk?
#   It is a small Python dataclass — a lightweight container
#   that holds one chunk of text plus metadata about it:
#   which study it came from, which position it is in the
#   document, and what type of content it contains.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. This file defines a class. It gets imported and used
#   by run_processing.py. Do not run it directly.
#
# HOW OTHER FILES USE THIS:
#   from processing.chunker import Chunker
#
#   chunker = Chunker()
#   chunks = chunker.chunk_study(parsed_study)
##############################################################################


from dataclasses import dataclass
# dataclass is a Python decorator that automatically generates
# common methods like __init__ and __repr__ for a class.
# Instead of writing a long __init__ method ourselves,
# we just decorate the class with @dataclass and Python
# handles all the boilerplate automatically.
# Think of it as a shortcut for creating simple data containers.

from typing import Any
from ingestion.document_parser import ParsedStudy, ParsedPaper
# We import our clean data models from document_parser.py.
# The chunker only ever works with CLEAN data — it never
# touches raw API responses directly.

from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "processing.chunker"


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — THE CHUNKING RULES
#
# These two numbers control how chunking works.
# They are defined at the top so they are easy to find and tune.
# ─────────────────────────────────────────────────────────────

CHUNK_SIZE = 500
# Maximum number of WORDS per chunk.
# We measure in words (not characters or tokens) because words
# are easier to reason about — "this chunk is about 500 words"
# is intuitive. Characters and tokens are less human-friendly.
# 500 words ≈ one or two paragraphs — a focused, coherent piece.

OVERLAP_SIZE = 50
# How many words to REPEAT between consecutive chunks.
# The last 50 words of chunk 1 become the first 50 words of chunk 2.
# This ensures no important sentence gets split across a boundary.
# 50 words is roughly 2-3 sentences — enough context to preserve
# meaning at the edges without wasting too much space on repetition.


# ─────────────────────────────────────────────────────────────
# THE TextChunk DATACLASS
#
# This is the output format of the chunker.
# Every chunk produced by this file is a TextChunk object.
# The embedder.py file receives a list of TextChunks and
# adds an embedding to each one.
# ─────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    One chunk of text from a study or paper, ready to be embedded.

    Think of this as a labelled envelope containing a piece of text.
    The label tells us everything about where this text came from
    and where it sits in the original document.

    Fields:
        chunk_id:    Unique identifier for this specific chunk.
                     Format: NCT_ID_chunk_0, NCT_ID_chunk_1, etc.
        nct_id:      Which study this chunk belongs to.
        chunk_text:  The actual text content of this chunk.
        chunk_index: Position of this chunk in the document.
                     0 = first chunk, 1 = second chunk, etc.
        source:      Where this chunk came from.
                     "study"  = from a ClinicalTrials.gov record
                     "paper"  = from a PubMed research paper
        word_count:  How many words are in this chunk.
                     Useful for debugging and quality checks.
    """

    chunk_id:    str
    nct_id:      str
    chunk_text:  str
    chunk_index: int
    source:      str
    word_count:  int


# ─────────────────────────────────────────────────────────────
# THE CHUNKER CLASS
# ─────────────────────────────────────────────────────────────

class Chunker:
    """
    Splits study and paper documents into overlapping text chunks.

    Usage:
        chunker = Chunker()
        chunks = chunker.chunk_study(parsed_study)
        chunks = chunker.chunk_paper(parsed_paper)
    """

    # ── CHUNK ONE STUDY ───────────────────────────────────────

    def chunk_study(self, study: ParsedStudy) -> list[TextChunk]:
        """
        Takes one ParsedStudy and produces a list of TextChunks.

        First we BUILD the full text by combining all the study's
        important fields into one long string, with clear labels
        so the embedding model knows what each section means.

        Then we SPLIT that long string into overlapping chunks.

        Args:
            study: A clean ParsedStudy object from document_parser.py

        Returns:
            A list of TextChunk objects, ready for embedding.
        """

        full_text = self._build_study_text(study)
        # Step 1: Combine all the study's fields into one labelled
        # text block. The _build_study_text method handles this.

        chunks = self._split_into_chunks(
            text=full_text,
            nct_id=study.nct_id,
            source="study",
        )
        # Step 2: Split that text block into overlapping chunks.
        # The _split_into_chunks method handles this.

        logger.info(
            f"Chunked study | "
            f"nct_id={study.nct_id} | "
            f"chunks_produced={len(chunks)}"
        )

        return chunks

    # ── CHUNK ONE PAPER ───────────────────────────────────────

    def chunk_paper(self, paper: ParsedPaper) -> list[TextChunk]:
        """
        Takes one ParsedPaper and produces a list of TextChunks.

        Same two-step process as chunk_study:
        1. Build the full text from all the paper's fields.
        2. Split into overlapping chunks.

        Args:
            paper: A clean ParsedPaper object from document_parser.py

        Returns:
            A list of TextChunk objects, ready for embedding.
        """

        full_text = self._build_paper_text(paper)
        chunks    = self._split_into_chunks(
            text=full_text,
            nct_id=paper.pmid,
            # For papers we use pmid as the identifier.
            # nct_id is a slightly misleading field name here —
            # it just means "the ID of the source document".
            source="paper",
        )

        logger.info(
            f"Chunked paper | "
            f"pmid={paper.pmid} | "
            f"chunks_produced={len(chunks)}"
        )

        return chunks

    # ── CHUNK MANY STUDIES AT ONCE ────────────────────────────

    def chunk_studies(self, studies: list[ParsedStudy]) -> list[TextChunk]:
        """
        Chunks a whole list of studies in one call.
        Returns ALL chunks from ALL studies as one flat list.

        Args:
            studies: List of ParsedStudy objects to chunk.

        Returns:
            Flat list of all TextChunks from all studies combined.
        """

        all_chunks: list[TextChunk] = []

        for study in studies:
            chunks = self.chunk_study(study)
            all_chunks.extend(chunks)
            # extend() adds all chunks from this study to the
            # running total list. After processing all studies,
            # all_chunks contains every single chunk from every study.

        logger.info(
            f"Chunked all studies | "
            f"studies={len(studies)} | "
            f"total_chunks={len(all_chunks)}"
        )

        return all_chunks

    # ── CHUNK MANY PAPERS AT ONCE ─────────────────────────────

    def chunk_papers(self, papers: list[ParsedPaper]) -> list[TextChunk]:
        """
        Chunks a whole list of papers in one call.
        Same pattern as chunk_studies — returns one flat list.

        Args:
            papers: List of ParsedPaper objects to chunk.

        Returns:
            Flat list of all TextChunks from all papers combined.
        """

        all_chunks: list[TextChunk] = []

        for paper in papers:
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked all papers | "
            f"papers={len(papers)} | "
            f"total_chunks={len(all_chunks)}"
        )

        return all_chunks

    # ── PRIVATE METHOD: BUILD STUDY TEXT ──────────────────────

    def _build_study_text(self, study: ParsedStudy) -> str:
        """
        Combines all a study's fields into one labelled text block.

        WHY LABEL EACH FIELD?
        When OpenAI embeds this text, the labels help the model
        understand what it is reading. "SPONSOR: Novo Nordisk"
        is much more informative than just "Novo Nordisk" sitting
        there with no context. Labels make embeddings more precise.

        WHY NOT JUST EMBED THE RAW JSON?
        Raw JSON has lots of noise — curly braces, quotes, nested
        keys that mean nothing to a language model. Plain labelled
        text is cleaner and produces better embeddings.

        Args:
            study: The ParsedStudy to convert to text.

        Returns:
            One long string containing all the study's key fields,
            each clearly labelled on its own line.
        """

        sections = []
        # We build sections as a list first, then join them at the end.
        # This is cleaner than string concatenation with +=
        # and avoids creating many intermediate string objects in memory.

        # ── TITLE AND IDENTIFICATION ───────────────────────────
        sections.append(f"NCT ID: {study.nct_id}")
        sections.append(f"TITLE: {study.title}")
        # These two fields are the most important identifiers.
        # Every chunk from this study will start with these
        # so the embedding always knows which study it came from.

        # ── SPONSOR ────────────────────────────────────────────
        sections.append(f"SPONSOR: {study.sponsor}")
        # The sponsor is critically important for our Track Record
        # and Pattern Finder agents — they reason about sponsors
        # across many studies. Labelling it clearly helps.

        # ── PHASE AND STATUS ───────────────────────────────────
        sections.append(f"PHASE: {study.phase}")
        sections.append(f"STATUS: {study.status}")
        # Status (COMPLETED, RECRUITING etc.) is what our Missing
        # Results agent cares about most — "COMPLETED + no results"
        # is the signal it looks for.

        # ── CONDITIONS AND INTERVENTIONS ───────────────────────
        if study.conditions:
            sections.append(
                f"CONDITIONS: {', '.join(study.conditions)}"
            )
            # join() turns the list into a readable comma-separated
            # string. ["Diabetes", "Obesity"] → "Diabetes, Obesity"

        if study.interventions:
            sections.append(
                f"INTERVENTIONS: {', '.join(study.interventions)}"
            )

        # ── OUTCOMES — THE MOST IMPORTANT FIELDS ───────────────
        if study.primary_outcome:
            sections.append(
                f"PRIMARY OUTCOME: {study.primary_outcome}"
            )
            # This is what the Broken Promises agent compares against
            # the actual results. If what was promised here does not
            # match what was measured, that is outcome switching.

        if study.secondary_outcomes:
            sections.append(
                f"SECONDARY OUTCOMES: {'; '.join(study.secondary_outcomes)}"
                # Using semicolon to separate secondary outcomes
                # because they can be long sentences themselves —
                # commas inside them would cause confusion.
            )

        # ── DATES ──────────────────────────────────────────────
        if study.start_date:
            sections.append(f"START DATE: {study.start_date}")

        if study.completion_date:
            sections.append(f"COMPLETION DATE: {study.completion_date}")

        # ── RESULTS POSTED — CRITICAL FIELD ────────────────────
        sections.append(
            f"RESULTS POSTED: {'YES' if study.results_posted else 'NO'}"
        )
        # This is a boolean in our data model but we convert it
        # to plain English here — "YES" or "NO" — because language
        # models understand natural language better than True/False.
        # "RESULTS POSTED: NO" is a very strong signal for our
        # Missing Results agent to pick up on.

        # ── ENROLLMENT ─────────────────────────────────────────
        if study.enrollment:
            sections.append(f"ENROLLMENT: {study.enrollment} participants")
            # Adding "participants" makes the number more meaningful
            # in the embedding — "1200 participants" is clearer
            # than just "1200".

        # ── AMENDMENTS ─────────────────────────────────────────
        if study.protocol_amendments:
            sections.append(
                f"PROTOCOL AMENDMENTS: "
                f"{len(study.protocol_amendments)} amendment(s) filed"
            )
            # We just note the COUNT of amendments here rather than
            # the full details. Too much amendment detail would bloat
            # the text and dilute the more important outcome signals.

        return "\n".join(sections)
        # Join all sections with newlines between them.
        # The result looks like:
        # NCT ID: NCT04788680
        # TITLE: A Study of Semaglutide in Adults With Type 2 Diabetes
        # SPONSOR: Novo Nordisk A/S
        # PHASE: PHASE3
        # STATUS: COMPLETED
        # CONDITIONS: Type 2 Diabetes, Obesity
        # PRIMARY OUTCOME: Change in HbA1c at 26 weeks
        # RESULTS POSTED: NO
        # ENROLLMENT: 1200 participants

    # ── PRIVATE METHOD: BUILD PAPER TEXT ──────────────────────

    def _build_paper_text(self, paper: ParsedPaper) -> str:
        """
        Combines all a paper's fields into one labelled text block.
        Same idea as _build_study_text but for PubMed papers.

        Args:
            paper: The ParsedPaper to convert to text.

        Returns:
            One long string with all the paper's key fields labelled.
        """

        sections = []

        sections.append(f"PMID: {paper.pmid}")
        sections.append(f"TITLE: {paper.title}")

        if paper.journal:
            sections.append(f"JOURNAL: {paper.journal}")

        if paper.pub_date:
            sections.append(f"PUBLICATION DATE: {paper.pub_date}")

        if paper.authors:
            sections.append(
                f"AUTHORS: {', '.join(paper.authors[:5])}"
                # We only include the first 5 authors to keep
                # the text focused. A paper with 30 authors does
                # not need all 30 listed — the first 5 are enough
                # to identify it.
            )

        if paper.nct_ids_referenced:
            sections.append(
                f"CLINICAL TRIALS REFERENCED: "
                f"{', '.join(paper.nct_ids_referenced)}"
            )
            # Which clinical trials this paper discusses.
            # Our Side Effect Checker agent uses this to link
            # papers back to their corresponding trial filings.

        if paper.abstract:
            sections.append(f"ABSTRACT: {paper.abstract}")
            # The abstract is the most important field — it contains
            # the actual scientific content. We add it last because
            # it is by far the longest field.

        return "\n".join(sections)

    # ── PRIVATE METHOD: SPLIT TEXT INTO CHUNKS ────────────────

    def _split_into_chunks(
        self,
        text: str,
        # The full labelled text to split up.

        nct_id: str,
        # The ID of the source document — used to name each chunk.

        source: str,
        # "study" or "paper" — used to tag each chunk.

    ) -> list[TextChunk]:
        """
        The core splitting algorithm. Splits one long text into
        overlapping chunks of CHUNK_SIZE words each.

        HOW IT WORKS:
        1. Split the full text into individual words
        2. Use a sliding window of CHUNK_SIZE words
        3. Slide forward by (CHUNK_SIZE - OVERLAP_SIZE) words each time
        4. This creates the overlap — each chunk shares OVERLAP_SIZE
           words with the next one

        Args:
            text:   The full text to split.
            nct_id: Source document ID for naming chunks.
            source: "study" or "paper" for tagging chunks.

        Returns:
            List of TextChunk objects.
        """

        words = text.split()
        # Split the text into a list of individual words.
        # "Hello world foo" → ["Hello", "world", "foo"]
        # Python's split() splits on any whitespace —
        # spaces, tabs, newlines — which is exactly what we want.

        if not words:
            # If the text was empty or only whitespace,
            # there is nothing to chunk. Return empty list.
            logger.warning(
                f"Empty text — no chunks produced | "
                f"nct_id={nct_id} | source={source}"
            )
            return []

        chunks:     list[TextChunk] = []
        chunk_index = 0
        # chunk_index tracks the position of the current chunk
        # in the document. First chunk = 0, second = 1, etc.

        step = CHUNK_SIZE - OVERLAP_SIZE
        # How many words to advance the window each iteration.
        # CHUNK_SIZE=500, OVERLAP_SIZE=50 → step=450
        # This means chunk 1 covers words 0-499,
        # chunk 2 covers words 450-949 (sharing 50 words with chunk 1),
        # chunk 3 covers words 900-1399, and so on.

        for start in range(0, len(words), step):
            # range(0, total_words, step) generates starting positions:
            # 0, 450, 900, 1350, ... until we run out of words.
            # Each number is where the next chunk begins.

            end = start + CHUNK_SIZE
            # The ending position for this chunk.
            # Python list slicing is safe even if end > len(words)
            # — it just returns whatever words are left.

            chunk_words = words[start:end]
            # Extract the words for this specific chunk.
            # words[0:500]   → first chunk
            # words[450:950] → second chunk (shares words 450-499)
            # words[900:1400] → third chunk

            if not chunk_words:
                # Safety check — should never happen with our range()
                # logic, but better safe than crashing.
                break

            chunk_text = " ".join(chunk_words)
            # Rejoin the words back into a string.
            # ["Hello", "world"] → "Hello world"

            chunk = TextChunk(
                chunk_id=f"{nct_id}_chunk_{chunk_index}",
                # Example: "NCT04788680_chunk_0", "NCT04788680_chunk_1"
                # This gives every chunk a globally unique ID
                # because NCT IDs are globally unique themselves.

                nct_id=nct_id,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                source=source,
                word_count=len(chunk_words),
            )

            chunks.append(chunk)
            chunk_index += 1

        logger.info(
            f"Split complete | "
            f"nct_id={nct_id} | "
            f"total_words={len(words)} | "
            f"chunks={len(chunks)} | "
            f"chunk_size={CHUNK_SIZE} | "
            f"overlap={OVERLAP_SIZE}"
        )

        return chunks