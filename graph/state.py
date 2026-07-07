##############################################################################
# graph/state.py
##############################################################################

from typing import Any, TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


def _merge_lists(a: list, b: list) -> list:
    """
    Reducer that APPENDS lists instead of replacing them.
    When multiple agents write to the same list field simultaneously,
    LangGraph uses this function to merge their outputs together.
    Without this, parallel agents overwrite each other's results.
    """
    return (a or []) + (b or [])


class SignalOutput(TypedDict):
    agent:       str
    signal_type: str
    nct_id:      str
    summary:     str
    evidence:    list
    confidence:  float


class MosaicState(TypedDict):

    # ── TASK INPUT ─────────────────────────────────────────────
    task:        str
    nct_ids:     list[str]
    max_studies: int

    # ── CONVERSATION HISTORY ───────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── AGENT OUTPUTS — Annotated with _merge_lists ────────────
    # These fields are written by multiple agents IN PARALLEL.
    # Without Annotated + reducer, LangGraph crashes with:
    # "Can receive only one value per step"
    # _merge_lists tells LangGraph: APPEND results, don't replace.
    signals:          Annotated[list[SignalOutput], _merge_lists]
    agents_activated: Annotated[list[str], _merge_lists]
    error_log:        Annotated[list, _merge_lists]

    # ── SUPERVISOR OUTPUTS ─────────────────────────────────────
    final_brief:  str
    run_complete: bool

    # ── METADATA ───────────────────────────────────────────────
    run_id: str