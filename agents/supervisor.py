##############################################################################
# agents/supervisor.py
#
# PURPOSE:
#   The Supervisor is the ORCHESTRATOR of the entire MOSAIC agent graph.
#   It has two jobs and two jobs only:
#
#   Job 1 — ROUTE (supervisor_route):
#     Read the incoming task, prepare the shared state, and hand off
#     to all six specialist agents simultaneously.
#     Think of it as a project manager who reads a client brief,
#     understands what needs investigating, and assigns work to
#     the right specialists.
#
#   Job 2 — COMPILE (supervisor_compile):
#     After all six specialists finish, read every signal they found,
#     rank them by priority, and write one final intelligence brief.
#     Think of it as the project manager collecting everyone's findings
#     and writing the executive summary for the client.
#
# WHAT THE SUPERVISOR DOES NOT DO:
#   The supervisor does NOT analyse any studies itself.
#   It does NOT call any tools.
#   It does NOT search the database.
#   Its only job is coordination — routing work and compiling output.
#   This separation keeps each component focused and testable.
#
# TWO FUNCTIONS — NOT A CLASS:
#   We define the supervisor as two plain async functions rather than
#   a class because LangGraph nodes are just functions — they receive
#   state, do work, and return updated state. No object needed.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. graph_builder.py imports these functions and wires them
#   into the LangGraph StateGraph as nodes.
##############################################################################


import uuid
# uuid.uuid4() generates a unique run ID for each analysis session.
# Every time someone calls POST /api/v1/analyze, a new run_id is
# generated so we can track all signals from that specific run.

from langchain_openai import ChatOpenAI
# ChatOpenAI is LangChain's wrapper around OpenAI's GPT models.
# It handles authentication, request formatting, and response parsing.
# We use it for the compile step — GPT-4o writes the final brief.

from langchain_core.messages import HumanMessage, SystemMessage
# HumanMessage = a message from the user (or in our case, the task)
# SystemMessage = instructions we give to GPT-4o before the conversation
# These are the standard message types in LangChain's message system.

from graph.state import MosaicState, SignalOutput
# MosaicState is the shared state dictionary that flows through all nodes.
# SignalOutput is the TypedDict that defines what one signal looks like.

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)
# __name__ here = "agents.supervisor"


# ── LLM INSTANCE ──────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=settings.openai_chat_model,
    # Which GPT model to use. Value from .env: "gpt-4o"
    # GPT-4o is the best reasoning model — it writes high-quality
    # intelligence briefs that sound like a real analyst wrote them.

    temperature=0.1,
    # temperature controls how creative vs deterministic the output is.
    # 0.0 = completely deterministic (same input → same output always)
    # 1.0 = very creative, random, unpredictable
    # 0.1 = mostly deterministic with slight variation
    # We use 0.1 because intelligence briefs should be factual and
    # consistent — not creative. We want the same quality every time.

    api_key=settings.openai_api_key,
    # The OpenAI API key from our .env file.
    # Never hardcode API keys — always read from environment variables.
)


##############################################################################
# FUNCTION 1: supervisor_route
#
# This is the FIRST node in the LangGraph graph.
# It runs before any specialist agent.
# Its job is to initialise the run and prepare state for the specialists.
##############################################################################

async def supervisor_route(state: MosaicState) -> dict:
    """
    The entry point of every MOSAIC analysis run.

    WHAT THIS FUNCTION DOES:
    1. Generates a unique run ID for this analysis session
    2. Logs what task is being investigated
    3. Returns an updated state that all specialists will receive

    WHY SO SIMPLE?
    The supervisor does NOT need to read the task and decide which
    specialists to activate — we always run ALL six specialists in
    parallel for every task. This is by design:
    - Different agents may find different signals in the same task
    - Running all six costs the same time as running one (parallel)
    - We never miss a signal type by selectively routing

    LANGGRAPH NODE CONTRACT:
    Every LangGraph node must:
    - Accept: the current MosaicState
    - Return: a dict of ONLY the fields that changed
    LangGraph automatically merges the returned dict into the full state.
    You do not return the entire state — just your changes.

    Args:
        state: The current MosaicState from LangGraph.

    Returns:
        Dict with updated run_id, agents_activated, and signals fields.
    """

    run_id = str(uuid.uuid4())
    # Generate a unique ID for this specific analysis run.
    # Every signal generated during this run will be tagged with this ID.
    # This lets us later query: "show me all signals from run X."

    logger.info(
        f"Supervisor routing | "
        f"run_id={run_id} | "
        f"task='{state.get('task', '')[:80]}'"
        # state.get('task', '') safely reads the task from state.
        # [:80] takes the first 80 characters — prevents very long
        # task descriptions from flooding the log.
    )

    return {
        "run_id":           run_id,
        # Set the run ID — all specialist agents will see this in state.

        "agents_activated": [],
        # Start with empty list — each specialist will ADD its name
        # to this list when it runs. By the end, this list shows
        # exactly which agents were activated.

        "signals":          [],
        # Start with empty signals list — each specialist APPENDS
        # its found signals to this list.
        # Because MosaicState uses add_messages pattern for signals,
        # LangGraph merges rather than replaces.

        "run_complete":     False,
        # False = run is in progress.
        # supervisor_compile sets this to True when done.

        "error_log":        [],
        # Empty error log at start — agents write errors here
        # if something goes wrong during their run.
    }


##############################################################################
# FUNCTION 2: supervisor_compile
#
# This is the LAST node before END in the LangGraph graph.
# It runs AFTER all six specialists have finished.
# Its job is to read every signal and write the final brief.
##############################################################################

async def supervisor_compile(state: MosaicState) -> dict:
    """
    Reads all agent signals and compiles the final intelligence brief.

    WHEN THIS RUNS:
    LangGraph calls this node only after ALL six specialist nodes
    have completed. This is guaranteed by the graph structure in
    graph_builder.py — all specialists connect to this node.

    WHAT THIS FUNCTION DOES:
    1. Collects all signals from state (from all 6 agents)
    2. Separates high-confidence signals from those needing review
    3. Uses GPT-4o to write a professional intelligence brief
    4. Returns the completed state

    WHY USE GPT-4o TO WRITE THE BRIEF?
    The raw signals are structured data — JSON with fields like
    summary, confidence, nct_id. They are accurate but not readable.
    GPT-4o transforms them into a professional narrative brief that
    a human analyst can read and act on immediately.
    The signals provide the FACTS. GPT-4o provides the WRITING.

    Args:
        state: The full MosaicState — now populated with all agent signals.

    Returns:
        Dict with final_brief, run_complete=True, and summary stats.
    """

    signals          = state.get("signals", [])
    agents_activated = state.get("agents_activated", [])
    task             = state.get("task", "")

    logger.info(
        f"Supervisor compiling brief | "
        f"signals={len(signals)} | "
        f"agents_activated={len(agents_activated)}"
    )

    if not signals:
        # No signals found by any agent — return a clean summary.
        # This can happen when the task finds no issues in the data.
        logger.info("No signals found — returning clean brief")
        return {
            "final_brief":  (
                "**EXECUTIVE SUMMARY:** Analysis complete. "
                "No significant research integrity signals were detected "
                "for the specified task and study set."
            ),
            "run_complete":     True,
            "agents_activated": agents_activated,
        }

    # ── SEPARATE SIGNALS BY REVIEW STATUS ────────────────────────────────
    high_confidence_signals = [
        s for s in signals
        if s.get("confidence", 0) >= 0.6
        # High confidence signals go directly into the brief.
        # They were either saved directly by HITLGate or are
        # being reported for inclusion in the summary.
    ]

    review_signals = [
        s for s in signals
        if s.get("confidence", 0) < 0.6
        # Low confidence signals went to the review queue.
        # We still mention them in the brief but flag them
        # as "pending human review" — the analyst knows to check
        # the review queue for these.
    ]

    # ── FORMAT SIGNALS FOR GPT-4o ─────────────────────────────────────────
    signals_text = _format_signals_for_llm(signals)
    # Convert the list of signal dicts into a clean text block
    # that GPT-4o can read and summarise effectively.
    # Explained in detail below in _format_signals_for_llm.

    # ── BUILD THE PROMPT FOR GPT-4o ───────────────────────────────────────
    system_prompt = """You are the Chief Intelligence Officer of MOSAIC —
a clinical trial research integrity system. Your job is to compile
a professional executive intelligence brief from the signals generated
by specialist AI agents.

BRIEF FORMAT:
1. EXECUTIVE SUMMARY — 2-3 sentences summarising the most critical findings
2. SIGNALS BY PRIORITY — each signal as a numbered item with:
   - What was found
   - Why it matters
   - What action to take
3. SIGNALS REQUIRING HUMAN REVIEW — list any low-confidence signals
4. PIPELINE HEALTH — note any errors or issues during the run

TONE: Professional, factual, actionable. Write as if briefing a
senior compliance officer or investigative journalist.
Be specific — include NCT IDs, sponsor names, and exact timeframes.
"""
    # The system prompt establishes GPT-4o's role and gives it
    # a clear format to follow. Structured prompts produce
    # structured, consistent outputs — critical for a production system.

    human_prompt = f"""
ANALYSIS TASK: {task}

SIGNALS FOUND BY AGENTS:
{signals_text}

HIGH CONFIDENCE SIGNALS: {len(high_confidence_signals)}
SIGNALS REQUIRING REVIEW: {len(review_signals)}
AGENTS ACTIVATED: {', '.join(agents_activated)}

Please compile the final intelligence brief now.
"""
    # The human prompt provides the actual content — the task and
    # all the signals. GPT-4o uses this to write the brief.

    # ── CALL GPT-4o ───────────────────────────────────────────────────────
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
            # ainvoke() is the ASYNC version of invoke().
            # "a" prefix = async in LangChain's naming convention.
            # We await it because it makes a network call to OpenAI.
            # The list contains our two messages — system first, then human.
            # GPT-4o reads both and generates the brief.
        )

        final_brief = response.content
        # response.content is the text GPT-4o generated.
        # This is the complete intelligence brief — ready to return.

        logger.info(
            f"Brief compiled successfully | "
            f"signals_included={len(signals)} | "
            f"brief_length={len(final_brief)} chars"
        )

    except Exception as e:
        # If GPT-4o call fails, fall back to a structured plain-text brief.
        # The run should not crash just because the LLM had an issue.
        logger.error(f"LLM brief compilation failed | error={e}")
        final_brief = _fallback_brief(signals, agents_activated, task)
        # _fallback_brief() generates a basic brief from the raw signal
        # data without using GPT-4o — explained below.

    return {
        "final_brief":              final_brief,
        # The complete intelligence brief — returned to the API caller.

        "run_complete":             True,
        # Mark the run as finished.

        "total_signals":            len(signals),
        # Total signals across all agents.

        "signals_requiring_review": len(review_signals),
        # How many signals went to the human review queue.

        "agents_activated":         agents_activated,
        # Which agents ran during this analysis.
    }


##############################################################################
# PRIVATE HELPER: _format_signals_for_llm
##############################################################################

def _format_signals_for_llm(signals: list[SignalOutput]) -> str:
    """
    Converts a list of signal dicts into a clean, readable text block
    that GPT-4o can effectively summarise into the final brief.

    WHY FORMAT BEFORE SENDING TO GPT-4o?
    Raw signal dicts are JSON — full of curly braces and quotes.
    GPT-4o works better with plain, labelled text than raw JSON.
    Formatting the signals into clear sections produces better briefs.

    Args:
        signals: List of SignalOutput dicts from specialist agents.

    Returns:
        A formatted string with all signals clearly laid out.
    """

    if not signals:
        return "No signals generated."

    lines = []
    # Build the output as a list of strings, then join at the end.
    # This is more efficient than string concatenation with +=
    # because each += creates a new string object in memory.

    for i, signal in enumerate(signals, start=1):
        # enumerate(signals, start=1) gives us:
        # i=1, signal=signals[0]
        # i=2, signal=signals[1]
        # etc. Starting at 1 makes the output "Signal 1", "Signal 2"
        # rather than "Signal 0" which looks wrong to humans.

        lines.append(f"SIGNAL {i}:")
        lines.append(f"  Agent:       {signal.get('agent', 'unknown')}")
        lines.append(f"  Type:        {signal.get('signal_type', 'unknown')}")
        lines.append(f"  NCT ID:      {signal.get('nct_id', 'N/A')}")
        lines.append(f"  Confidence:  {signal.get('confidence', 0.0):.2f}")
        # :.2f formats the confidence score to 2 decimal places.
        # 0.847 → 0.85. Clean and readable.

        lines.append(f"  Summary:     {signal.get('summary', '')}")
        lines.append("")
        # Empty string adds a blank line between signals —
        # makes the text block much easier for GPT-4o to parse.

    return "\n".join(lines)
    # Join all lines with newline characters.
    # "\n".join(["a", "b", "c"]) → "a\nb\nc"


##############################################################################
# PRIVATE HELPER: _fallback_brief
##############################################################################

def _fallback_brief(
    signals:          list,
    agents_activated: list,
    task:             str,
) -> str:
    """
    Generates a basic structured brief WITHOUT using GPT-4o.

    Called when the LLM call fails — ensures the API always returns
    something useful even if OpenAI is down or rate-limited.
    The output is less polished than the GPT-4o brief but contains
    all the factual information the caller needs.

    Args:
        signals:          All signals from the run.
        agents_activated: Which agents ran.
        task:             The original analysis task.

    Returns:
        A plain text brief built directly from signal data.
    """

    lines = [
        "**EXECUTIVE SUMMARY:**",
        f"Analysis complete. {len(signals)} signal(s) detected.",
        "",
        "**SIGNALS BY PRIORITY:**",
        "",
    ]

    for i, signal in enumerate(signals, start=1):
        lines.append(
            f"{i}. **{signal.get('nct_id', 'Unknown')} "
            f"- {signal.get('signal_type', 'Unknown')}:**"
        )
        lines.append(f"   {signal.get('summary', 'No summary available.')}")
        lines.append(
            f"   Confidence: {signal.get('confidence', 0.0):.2f} | "
            f"Agent: {signal.get('agent', 'unknown')}"
        )
        lines.append("")

    lines.append(f"**AGENTS ACTIVATED:** {', '.join(agents_activated)}")
    lines.append(
        "\n*Note: This brief was generated without LLM assistance "
        "due to a temporary error. Please review raw signals directly.*"
    )

    return "\n".join(lines)