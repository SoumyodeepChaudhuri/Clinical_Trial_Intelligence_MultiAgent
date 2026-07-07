##############################################################################
# agents/missing_results_agent.py
#
# PURPOSE:
#   Finds completed clinical trials that never posted their results.
#   By US law (FDAAA 801), sponsors must post results within 12 months
#   of the primary completion date. About 30% of completed trials
#   never do — this agent finds every single one in our corpus.
#
# WHAT THIS AGENT DOES STEP BY STEP:
#   1. Loads its reasoning procedures from procedural memory
#      (the rules it has learned from human feedback over time)
#   2. Searches past episodes — has it seen similar cases before?
#   3. Searches the database for completed studies with no results
#   4. For each suspicious study, verifies with a live API call
#   5. Checks the sponsor's track record
#   6. Generates a signal with confidence score
#   7. Updates the sponsor's profile in semantic memory
#   8. Saves this session as an episode in episodic memory
#
# CONFIDENCE THRESHOLDS:
#   >= 0.65 → saved directly (high confidence, clear violation)
#   <  0.65 → sent to human review queue
##############################################################################


import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from graph.state import MosaicState, SignalOutput
from memory.procedural_store import ProceduralStore
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from tools.search_tools import (
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
)
from tools.clinical_tools import check_results_posted, fetch_study_details
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


# ── AGENT IDENTITY ─────────────────────────────────────────────────────────
AGENT_NAME = "missing_results_agent"
# Every agent has a fixed name string used across:
# - Logging (every log line shows which agent it came from)
# - Episodic memory (episodes are tagged with agent_name)
# - Procedural memory (rules are stored per agent_name)
# - Signal output (the signal says which agent generated it)
# - HITL (the threshold lookup uses agent_name as the key)

SIGNAL_TYPE = "missing_results"
# The signal type label that appears in the signals table.
# Analysts filter signals by type — this label must be consistent
# across every run so filters always work correctly.

AGENT_TOOLS = [
    search_studies_by_meaning,  # search database by meaning
    search_past_episodes,       # check what this agent found before
    save_episode,               # save this session to memory
    get_sponsor_profile,        # check sponsor's track record
    update_sponsor_profile,     # update after analysis
    check_results_posted,       # verify live from ClinicalTrials.gov
    fetch_study_details,        # get full study details if needed
]
# Each agent gets ONLY the tools it actually needs.
# Giving an agent too many tools confuses GPT-4o — it spends
# reasoning capacity deciding which tool to use instead of
# focusing on the actual analysis task.
# This focused toolset keeps the agent sharp and efficient.


# ── STORES ─────────────────────────────────────────────────────────────────
_procedural = ProceduralStore()
# Loads reasoning rules for this agent at the start of every run.

_episodic   = EpisodicStore()
# Saves and searches this agent's past sessions.

_semantic   = SemanticStore()
# Reads and updates sponsor credibility profiles.


# ── LLM WITH TOOLS ─────────────────────────────────────────────────────────
_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    # "gpt-4o" — the most capable reasoning model.
    temperature=0.1,
    # Low temperature for consistent, factual analysis.
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)
# .bind_tools() attaches our tool list to the LLM.
# Now when GPT-4o reasons and decides it needs to search the database,
# it can emit a "tool_call" in its response.
# LangGraph sees the tool_call, executes the right function,
# and feeds the result back to the agent automatically.
# Without .bind_tools(), the agent has no way to access data —
# it would only reason from what is in its system prompt.


async def missing_results_node(state: MosaicState) -> dict:
    """
    The Missing Results Agent node — called by LangGraph during graph execution.

    Finds completed clinical trials that never posted results.
    Returns updated state with any signals found and this agent's name
    added to agents_activated.

    Args:
        state: The current MosaicState from LangGraph.
               Contains the task, nct_ids to analyse, and prior signals.

    Returns:
        Dict with updated signals list and agents_activated list.
    """

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        # ── STEP 1: LOAD REASONING PROCEDURES ──────────────────────────
        procedures = await _procedural.get_procedures(AGENT_NAME)
        # Load all reasoning rules for this agent from Cloud SQL.
        # This includes both default rules AND any rules learned
        # from human feedback via the HITL rejection loop.
        # Returns a list of plain English rule strings.

        procedures_text = "\n".join(f"- {r}" for r in procedures)
        # Format as a bulleted list for the system prompt.
        # "\n".join() puts each rule on its own line.
        # f"- {r}" adds a dash before each rule for readability.
        # GPT-4o reads these rules and applies them during analysis.

        # ── STEP 2: BUILD THE SYSTEM PROMPT ────────────────────────────
        system_prompt = f"""You are the Missing Results Agent for MOSAIC —
a clinical trial research integrity intelligence system.

YOUR MISSION:
Find completed clinical trials that have never posted their results
to ClinicalTrials.gov, violating federal law (FDAAA 801).
By law, results must be posted within 12 months of primary completion.

YOUR REASONING RULES (follow these exactly):
{procedures_text}

YOUR WORKFLOW:
1. Search past episodes to see if you have investigated similar cases
2. Search the database for completed studies with missing results
3. For each suspicious study, verify the current status with a live API call
4. Check the sponsor's track record using get_sponsor_profile
5. Generate a signal with confidence score based on evidence strength
6. Update the sponsor profile with your findings
7. Save this session as an episode before finishing

CONFIDENCE SCORING GUIDE:
- 0.9+ : Completed 5+ years ago, zero results, repeat offender sponsor
- 0.8  : Completed 2-5 years ago, zero results, known non-compliant sponsor
- 0.7  : Completed 1-2 years ago, zero results, average sponsor
- 0.6  : Completed 1 year ago exactly, borderline timing
- Below 0.6: Uncertain — send to human review

OUTPUT FORMAT for each signal found:
Return a JSON block exactly like this:
{{
  "nct_id": "NCT_ID_HERE",
  "signal_type": "missing_results",
  "summary": "Plain English description of what you found",
  "evidence": ["key fact 1", "key fact 2", "key fact 3"],
  "confidence": 0.85
}}

If you find no signals, say "NO_SIGNALS_FOUND" clearly.
"""
        # The system prompt is GPT-4o's entire instruction set.
        # It defines the agent's role, its rules, its workflow,
        # and the exact output format we need.
        # The more precise and structured this prompt is,
        # the more reliable the agent's output will be.

        task = state.get("task", "Find completed trials with missing results")
        # Read the task from state — what analysis was requested.
        # Falls back to a default task if none was provided.

        nct_ids = state.get("nct_ids", [])
        # Specific studies to analyse — empty list means analyse broadly.

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES TO CHECK: {nct_ids if nct_ids else "Search broadly — no specific studies provided"}

Begin your investigation now. Use your tools to search for completed
studies with missing results. Generate signals for every violation you find.
"""

        # ── STEP 3: RUN THE AGENT REASONING LOOP ───────────────────────
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]
        # Start the conversation with system instructions and the task.
        # LangGraph will extend this list as the agent calls tools
        # and receives tool results — building up the full conversation.

        signals_found = []
        # Accumulates all signals this agent generates during this run.

        max_iterations = 10
        # Safety cap — prevents infinite loops if the agent keeps
        # calling tools without reaching a conclusion.
        # 10 iterations is more than enough for thorough analysis.

        for iteration in range(max_iterations):
            # The agent reasoning loop — each iteration is one LLM call.
            # The agent reads the conversation, decides what to do,
            # either calls a tool or writes its final answer.

            response = await _llm.ainvoke(messages)
            # ainvoke() sends the full conversation to GPT-4o and
            # gets back the agent's next action.
            # response.content = text the agent wrote
            # response.tool_calls = tools the agent wants to call

            messages.append(AIMessage(content=response.content or ""))
            # Add the agent's response to the conversation history.
            # This is how LangGraph maintains conversation context —
            # every message is appended so the agent always sees
            # the full history of what has been said and done.

            if not response.tool_calls:
                # Agent wrote a final answer without calling any more tools.
                # This means it has finished its analysis.
                # Parse the response for signal JSON blocks.
                logger.info(
                    f"{AGENT_NAME} | Analysis complete | iteration={iteration+1}"
                )
                signals_found = _parse_signals(response.content, AGENT_NAME)
                break
                # Exit the loop — agent is done.

            # ── EXECUTE TOOL CALLS ──────────────────────────────────────
            for tool_call in response.tool_calls:
                tool_result = await _execute_tool(tool_call, AGENT_TOOLS)
                # Run the tool the agent requested and get the result.
                # _execute_tool() finds the right tool function by name
                # and calls it with the arguments GPT-4o specified.

                messages.append(
                    HumanMessage(
                        content=f"Tool result for {tool_call['name']}:\n{tool_result}"
                    )
                )
                # Add the tool result to the conversation.
                # Now GPT-4o can read what the tool returned and
                # decide what to do next — call another tool or
                # write its final analysis.

        # ── STEP 4: SAVE THIS SESSION AS AN EPISODE ────────────────────
        episode_content = (
            f"Task: {task}. "
            f"Found {len(signals_found)} missing results signals. "
            f"Signals: {[s.get('nct_id') for s in signals_found]}"
        )
        # Build a plain text description of what happened this session.
        # This gets embedded and stored for future searches.

        await _episodic.save_episode(
            agent_name=AGENT_NAME,
            content=episode_content,
            outcome="signal_generated" if signals_found else "no_signal",
        )
        # Save to episodic memory so future runs can search:
        # "have I found missing results from this type of sponsor before?"

        logger.info(
            f"{AGENT_NAME} | Complete | signals_found={len(signals_found)}"
        )

        # ── STEP 5: RETURN UPDATED STATE ───────────────────────────────
        current_signals   = state.get("signals", [])
        current_activated = state.get("agents_activated", [])

        return {
            "signals":          current_signals + signals_found,
            # Append this agent's signals to whatever signals
            # other agents have already found.
            # We read the existing list and add to it — not replace it.
            # This is critical for parallel execution — all 6 agents
            # write to the same signals list and we must not overwrite
            # each other's work.

            "agents_activated": current_activated + [AGENT_NAME],
            # Add this agent's name to the activated list.
        }

    except Exception as e:
        logger.error(f"{AGENT_NAME} | Error | {e}")
        error_log = state.get("error_log", [])
        return {
            "error_log":        error_log + [f"{AGENT_NAME}: {str(e)}"],
            "agents_activated": state.get("agents_activated", []) + [AGENT_NAME],
        }
        # Even on error — add this agent to activated list
        # and log the error. Never silently fail.


def _parse_signals(response_text: str, agent_name: str) -> list[SignalOutput]:
    """
    Extracts signal JSON blocks from the agent's text response.

    GPT-4o writes signals as JSON blocks in its response text.
    This function finds and parses every JSON block.

    WHY PARSE FROM TEXT?
    We could ask GPT-4o to return structured JSON directly.
    But agents need to explain their reasoning in plain text TOO —
    the text before and after the JSON contains valuable context
    for debugging and audit trails.
    Parsing JSON from mixed text gives us both.

    Args:
        response_text: The full text response from GPT-4o.
        agent_name:    Used to tag each signal with its source agent.

    Returns:
        List of SignalOutput dicts ready to add to state.
    """

    signals = []

    if not response_text or "NO_SIGNALS_FOUND" in response_text:
        return signals
        # Agent explicitly said no signals — return empty list.

    import re
    # re is Python's regular expression library.
    # We use it to find JSON blocks in the agent's text response.

    json_pattern = re.compile(r'\{[^{}]*"signal_type"[^{}]*\}', re.DOTALL)
    # This regex finds JSON objects that contain "signal_type".
    # r'\{...\}' matches curly braces — the boundaries of a JSON object.
    # [^{}]* means "any characters except curly braces" — the content.
    # "signal_type" ensures we only match signal JSON, not other JSON.
    # re.DOTALL means . matches newlines too — handles multi-line JSON.

    matches = json_pattern.findall(response_text)
    # findall() returns ALL matches in the text as a list of strings.

    for match in matches:
        try:
            signal_data = json.loads(match)
            # json.loads() converts the JSON string to a Python dict.
            # If the JSON is malformed (incomplete, wrong quotes),
            # this raises json.JSONDecodeError — caught below.

            signal: SignalOutput = {
                "agent":       agent_name,
                "signal_type": signal_data.get("signal_type", SIGNAL_TYPE),
                "nct_id":      signal_data.get("nct_id", ""),
                "summary":     signal_data.get("summary", ""),
                "evidence":    signal_data.get("evidence", []),
                "confidence":  float(signal_data.get("confidence", 0.5)),
                # float() ensures confidence is always a number.
                # If GPT-4o somehow returned a string like "0.85",
                # float("0.85") = 0.85 — safe conversion.
            }
            signals.append(signal)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse signal JSON | error={e}")
            # Skip malformed JSON — do not crash the whole run.
            continue

    return signals


async def _execute_tool(tool_call: dict, available_tools: list) -> str:
    """
    Finds and executes the tool that GPT-4o requested.

    LangGraph agents emit tool_calls in their responses — these
    contain the tool name and arguments GPT-4o wants to use.
    This function looks up the right tool by name and calls it.

    Args:
        tool_call:       The tool call from GPT-4o response.
                         Contains: name (string) and args (dict).
        available_tools: List of tool functions available to this agent.

    Returns:
        The tool's output as a string — fed back to the agent.
    """

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    # Extract the tool name and arguments from the tool_call dict.
    # tool_name: which function GPT-4o wants to call
    # tool_args: the arguments it wants to pass to that function

    tool_func = None
    for t in available_tools:
        if t.name == tool_name:
            tool_func = t
            break
    # Search through available_tools to find the one with matching name.
    # t.name is the tool's registered name from the @tool decorator.
    # If no match found, tool_func stays None — handled below.

    if tool_func is None:
        return f"Error: Tool '{tool_name}' not found in agent's toolset."

    try:
        result = tool_func.invoke(tool_args)
        # tool_func.invoke() calls the tool with the provided arguments.
        # This is LangChain's standard way to call a tool —
        # it handles argument validation and error wrapping.
        # Returns a string (all our tools return JSON strings).

        return str(result)

    except Exception as e:
        logger.error(f"Tool execution failed | tool={tool_name} | error={e}")
        return f"Error executing tool '{tool_name}': {str(e)}"
        # Return error as string — the agent reads this and decides
        # whether to retry with different arguments or move on.