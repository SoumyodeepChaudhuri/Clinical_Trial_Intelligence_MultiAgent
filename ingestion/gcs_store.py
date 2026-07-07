##############################################################################
# ingestion/gcs_store.py
#
# PURPOSE:
#   This file saves our data to Google Cloud Storage (GCS) — and
#   loads it back when we need it.
#   Think of GCS as a giant hard drive in the cloud that never
#   runs out of space and is always online.
#
# WHY DO WE SAVE TO GCS AT ALL — WHY NOT JUST KEEP DATA IN MEMORY:
#   If our Python program crashes halfway through downloading
#   144 studies, we lose everything if we only kept it in memory.
#   By saving to GCS as we go, even if something crashes,
#   the studies we already downloaded are safe.
#
# THE "RAW vs PROCESSED" PATTERN — WHY WE SAVE DATA TWICE:
#   We save EVERY piece of data in two different forms:
#
#   1. RAW   — exactly what the API gave us, completely untouched.
#              Think of this as a photocopy of the original document.
#   2. PROCESSED — the cleaned-up version after document_parser.py
#              has done its work. Think of this as a neatly typed-up
#              summary of that document.
#
#   WHY BOTH? If our parser (document_parser.py) has a bug and
#   cleans the data incorrectly, we have NOT lost anything —
#   the raw original is still sitting safely in GCS. We can simply
#   fix the parser and re-process the raw data again.
#   This is a real production safety pattern — never throw away
#   your original source data.
#
# THE FOLDER STRUCTURE INSIDE OUR BUCKET:
#   raw/studies/NCT04788680.json        ← exactly what ClinicalTrials.gov sent us
#   raw/papers/38234567.json            ← exactly what PubMed sent us
#   processed/studies/NCT04788680.json  ← the cleaned ParsedStudy object
#   processed/papers/38234567.json      ← the cleaned ParsedPaper object
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. This file defines a class that run_ingestion.py uses.
#   Do not run it by itself.
#
# HOW OTHER FILES USE THIS:
#   from ingestion.gcs_store import GCSStore
#
#   store = GCSStore()
#   await store.save_raw_study(nct_id="NCT04788680", data=raw_dict)
#   await store.save_parsed_study(study=parsed_study_object)
##############################################################################


import json
# json is Python's built-in library for converting between
# Python dictionaries and JSON text (the format APIs and files use).
# json.dumps() turns a Python dict INTO a JSON string.
# json.loads() turns a JSON string BACK INTO a Python dict.

import asyncio
# We use this for asyncio.to_thread() — explained in detail below.

from typing import Any
# Any = "this value can be of any type". Used for flexible dict data.

from google.cloud import storage
# This is Google's official Python library for talking to
# Cloud Storage. It is installed via:
#   pip install google-cloud-storage
# Once we are logged in via "gcloud auth application-default login",
# this library automatically knows how to authenticate — no API
# key needed anywhere in our code.

from config.settings import settings
# Gives us settings.gcp_project_id and settings.gcs_bucket_name
# which we read from our .env file.

from config.logging_config import setup_logging
from ingestion.document_parser import ParsedStudy, ParsedPaper
# We import these two classes from document_parser.py because
# this file needs to know their "shape" in order to save them.

logger = setup_logging(__name__)
# __name__ here = "ingestion.gcs_store"


# ─────────────────────────────────────────────────────────────
# FOLDER PATHS INSIDE OUR BUCKET
#
# A GCS "bucket" does not really have folders the way your
# computer does — but it LOOKS like it has folders because
# every file we save has a path-like name with slashes in it.
# Example: "raw/studies/NCT04788680.json" looks like a folder
# structure even though GCS technically just sees one long name.
# Defining these as constants means if we ever want to change
# the folder layout, we only change it in ONE place.
# ─────────────────────────────────────────────────────────────

PREFIX_RAW_STUDIES       = "raw/studies"
PREFIX_RAW_PAPERS        = "raw/papers"
PREFIX_PROCESSED_STUDIES = "processed/studies"
PREFIX_PROCESSED_PAPERS  = "processed/papers"


# ─────────────────────────────────────────────────────────────
# THE GCS STORE CLASS
# ─────────────────────────────────────────────────────────────

class GCSStore:
    """
    Handles saving data to and loading data from Google Cloud Storage.

    IMPORTANT — WHY WE USE asyncio.to_thread() THROUGHOUT THIS FILE:

    Google's official storage library is "synchronous" — meaning
    when you ask it to upload a file, your whole program freezes
    and waits until the upload is done before doing anything else.

    But our entire MOSAIC system is built to be "asynchronous" —
    meaning we want our program to be able to do OTHER things
    while waiting for slow operations like uploads to finish.

    asyncio.to_thread() is the bridge between these two worlds.
    It takes a synchronous function (like a GCS upload) and runs
    it in a separate background thread, while letting our main
    program keep working on other tasks in the meantime.
    Think of it like handing a task to an assistant in another
    room, instead of standing there waiting yourself.
    """

    def __init__(self):
        self._client = storage.Client(project=settings.gcp_project_id)
        # Create a connection to Google Cloud Storage.
        # This does NOT immediately connect to the internet —
        # it just sets up the object that knows HOW to connect
        # when we actually ask it to do something.

        self._bucket = self._client.bucket(settings.gcs_bucket_name)
        # Get a reference to our specific bucket — the one we
        # created earlier in the GCP Console / terminal.
        # Again, this does not make a network call yet — it is
        # just pointing to the right "folder" we will work inside.

        logger.info(
            f"GCSStore initialised | "
            f"bucket={settings.gcs_bucket_name} | "
            f"project={settings.gcp_project_id}"
        )

    # ── SAVE A RAW STUDY ───────────────────────────────────────

    async def save_raw_study(
        self,
        nct_id: str,
        # The study's unique ID — we use this to name the file.

        data: dict[str, Any],
        # The raw dictionary exactly as ClinicalTrials.gov sent it.
        # Nothing has been cleaned or modified yet.

    ) -> str:
        """
        Saves the EXACT, untouched API response for one study.

        Call this the moment you receive data from the API —
        BEFORE any cleaning or parsing happens. This way, even
        if the parser has a bug, the original is always safe.

        Args:
            nct_id: The study's ID, used as the filename.
            data:   The raw study dictionary to save.

        Returns:
            The path inside GCS where the file was saved.
            Example: "raw/studies/NCT04788680.json"
        """

        gcs_path = f"{PREFIX_RAW_STUDIES}/{nct_id}.json"
        # Build the full file path/name.
        # Example: "raw/studies/NCT04788680.json"

        await self._upload_json(path=gcs_path, data=data)
        # Call our private helper method (defined further down)
        # which does the actual uploading work.

        logger.info(f"Saved raw study | nct_id={nct_id} | path={gcs_path}")
        return gcs_path

    # ── SAVE A RAW PAPER ───────────────────────────────────────

    async def save_raw_paper(
        self,
        pmid: str,
        # PubMed's unique paper ID — used as the filename.

        data: dict[str, Any],
        # The raw paper dictionary, untouched.

    ) -> str:
        """
        Saves the EXACT, untouched data for one PubMed paper.
        Same idea as save_raw_study — but for papers.

        Args:
            pmid: The paper's PubMed ID, used as the filename.
            data: The raw paper dictionary to save.

        Returns:
            The path inside GCS where the file was saved.
        """

        gcs_path = f"{PREFIX_RAW_PAPERS}/{pmid}.json"
        await self._upload_json(path=gcs_path, data=data)

        logger.info(f"Saved raw paper | pmid={pmid} | path={gcs_path}")
        return gcs_path

    # ── SAVE A CLEANED (PARSED) STUDY ─────────────────────────

    async def save_parsed_study(self, study: ParsedStudy) -> str:
        """
        Saves the CLEANED version of a study — after document_parser.py
        has already processed it into a ParsedStudy object.

        Args:
            study: A ParsedStudy object (the clean, typed version).

        Returns:
            The path inside GCS where the file was saved.
        """

        gcs_path = f"{PREFIX_PROCESSED_STUDIES}/{study.nct_id}.json"

        await self._upload_json(
            path=gcs_path,
            data=study.model_dump(),
            # study.model_dump() is a Pydantic method that converts
            # our ParsedStudy OBJECT back into a plain Python dictionary.
            # We need a plain dict because that is what gets turned
            # into JSON text for saving — Pydantic objects cannot be
            # saved directly, only their dict version can.
        )

        logger.info(
            f"Saved parsed study | nct_id={study.nct_id} | path={gcs_path}"
        )
        return gcs_path

    # ── SAVE A CLEANED (PARSED) PAPER ─────────────────────────

    async def save_parsed_paper(self, paper: ParsedPaper) -> str:
        """
        Saves the CLEANED version of a paper.

        Args:
            paper: A ParsedPaper object (the clean, typed version).

        Returns:
            The path inside GCS where the file was saved.
        """

        gcs_path = f"{PREFIX_PROCESSED_PAPERS}/{paper.pmid}.json"

        await self._upload_json(
            path=gcs_path,
            data=paper.model_dump(),
        )

        logger.info(
            f"Saved parsed paper | pmid={paper.pmid} | path={gcs_path}"
        )
        return gcs_path

    # ── LOAD A CLEANED STUDY BACK FROM GCS ────────────────────

    async def load_parsed_study(self, nct_id: str) -> ParsedStudy | None:
        """
        Loads a previously saved, cleaned study back from GCS.

        This is the REVERSE of save_parsed_study — we use this
        in the processing layer when we need to read studies
        back in to chunk and embed them.

        Args:
            nct_id: Which study to load, by its NCT ID.

        Returns:
            A ParsedStudy object if found.
            None if no study with that ID exists in GCS.
        """

        gcs_path = f"{PREFIX_PROCESSED_STUDIES}/{nct_id}.json"
        data = await self._download_json(path=gcs_path)
        # Download the raw JSON text and convert it back to a
        # Python dictionary (our private helper does this).

        if not data:
            return None
            # If nothing came back, the file probably does not exist.

        return ParsedStudy(**data)
        # **data "unpacks" the dictionary into keyword arguments.
        # Example: if data = {"nct_id": "NCT123", "title": "..."}
        # then ParsedStudy(**data) is the same as writing:
        # ParsedStudy(nct_id="NCT123", title="...")
        # This rebuilds our typed Pydantic object from the saved dict.

    # ── LIST ALL STUDIES WE HAVE ALREADY PROCESSED ────────────

    async def list_processed_studies(self) -> list[str]:
        """
        Returns a list of every study's NCT ID currently saved
        in the "processed" folder of our bucket.

        We use this later in the processing layer to know exactly
        which studies are available to chunk and embed, without
        needing to ask the database first.

        Returns:
            A list of NCT ID strings.
            Example: ["NCT04788680", "NCT03232294", "NCT06796322"]
        """

        blobs = await asyncio.to_thread(
            self._bucket.list_blobs,
            prefix=PREFIX_PROCESSED_STUDIES,
        )
        # "Blob" is GCS terminology for a single saved file.
        # list_blobs() is synchronous, so we wrap it in
        # asyncio.to_thread() as explained at the top of this class.
        # prefix= means: only show files whose path starts with
        # "processed/studies" — i.e. only the files we want.

        nct_ids = []
        for blob in blobs:
            filename = blob.name.split("/")[-1]
            # blob.name looks like: "processed/studies/NCT04788680.json"
            # .split("/") breaks it into: ["processed", "studies", "NCT04788680.json"]
            # [-1] grabs the LAST piece — just the filename itself.

            nct_id = filename.replace(".json", "")
            # Remove the ".json" ending to get just the NCT ID.
            # "NCT04788680.json" → "NCT04788680"

            if nct_id:
                nct_ids.append(nct_id)

        logger.info(f"Listed processed studies | count={len(nct_ids)}")
        return nct_ids

    # ── PRIVATE HELPER: UPLOAD ANY DICT AS A JSON FILE ────────

    async def _upload_json(
        self,
        path: str,
        # Where inside the bucket to save this file.

        data: dict[str, Any],
        # The Python dictionary we want to save.

    ) -> None:
        """
        The shared internal method that ACTUALLY does the uploading.
        Every save_* method above eventually calls this one.

        Args:
            path: The destination path inside the GCS bucket.
            data: The dictionary to save as JSON.
        """

        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        # Step 1: json.dumps(data, indent=2) turns our Python dict
        #         into a nicely-formatted JSON text string.
        #         indent=2 just makes it readable with spacing —
        #         purely cosmetic, helps when you open the file later.
        # Step 2: default=str — if there is ANY value that json.dumps
        #         does not know how to convert (like a datetime object),
        #         it converts it to plain text instead of crashing.
        # Step 3: .encode("utf-8") converts that text string into raw
        #         bytes. GCS uploads need bytes, not a Python string.

        blob = self._bucket.blob(path)
        # Create a reference to where this file will live in GCS.
        # This does NOT upload anything yet — it is just a pointer
        # to the destination, like writing an address on an envelope
        # before you actually mail it.

        await asyncio.to_thread(
            blob.upload_from_string,
            json_bytes,
            content_type="application/json",
        )
        # NOW we actually upload. upload_from_string is synchronous
        # (it would normally freeze our program), so we run it inside
        # asyncio.to_thread() to keep everything else moving smoothly.
        # content_type="application/json" tells GCS what kind of file
        # this is — helpful when browsing files in the GCP Console.

    # ── PRIVATE HELPER: DOWNLOAD A JSON FILE BACK AS A DICT ───

    async def _download_json(
        self,
        path: str,
    ) -> dict[str, Any] | None:
        """
        The shared internal method that downloads a file from GCS
        and converts it back into a Python dictionary.

        Args:
            path: The path inside the GCS bucket to download from.

        Returns:
            A Python dictionary if the file was found.
            None if the file does not exist or something went wrong.
        """

        try:
            blob = self._bucket.blob(path)
            # Point to the file we want to download.

            json_bytes = await asyncio.to_thread(blob.download_as_bytes)
            # Download the file's raw content as bytes.
            # Again wrapped in asyncio.to_thread() since this
            # Google library call is synchronous by default.

            return json.loads(json_bytes.decode("utf-8"))
            # Step 1: .decode("utf-8") turns the raw bytes back
            #         into readable text.
            # Step 2: json.loads(...) turns that JSON text back
            #         into a Python dictionary we can work with.

        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                # A 404 simply means "this file does not exist".
                # This is an EXPECTED situation sometimes — not a bug.
                logger.warning(f"File not found in GCS | path={path}")
            else:
                # Anything else is a real, unexpected problem —
                # log it as an error so we can investigate.
                logger.error(
                    f"Failed to download from GCS | path={path} | error={e}"
                )
            return None