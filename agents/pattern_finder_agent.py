##############################################################################
# agents/pattern_finder_agent.py
#
# PURPOSE:
#   Finds cross-study patterns that no single human reading one study
#   at a time would ever notice.
#
# WHAT PATTERNS IT LOOKS FOR:
#   - Same drug being tested by multiple sponsors for the same condition
#     (why are so many companies testing this? what are they hiding?)
#   - A drug that failed Phase 2 being retried in a new disease area
#     (same mechanism, different disease — hoping no one notices)
#   - A sponsor consistently running trials in the same condition
#     with consistently poor compliance
#   - Multiple trials from the same sponsor all missing results
#     in the same year (coordinated non-compliance?)
#
# WHY THIS IS UNIQUE:
#   Pattern detection requires SIMULTANEOUS awareness of hundreds of studies.
#   A human analyst can deeply follow 10-15 studies at a time.
#   This agent holds all 286 studies in its search context at once.
##############################################################################

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from graph.state import MosaicState, SignalOutput
from memory.procedural_store import ProceduralStore
from memory.episodic_store import EpisodicStore
from tools.search_tools import (
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    get_low_credibility_sponsors,
)
from tools.clinical_tools import search_studies_by_condition, fetch_study_details
from tools.pubmed_tools import search_pubmed_by_query
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

AGENT_NAME  = "pattern_finder_agent"
SIGNAL_TYPE = "cross_study_pattern"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    get_low_credibility_sponsors,
    search_studies_by_condition,
    fetch_study_details,
    search_pubmed_by_query,
    # search_pubmed_by_query is key for this agent —
    # papers that reference MULTIPLE trials reveal cross-study patterns
    # that would be invisible from the trial database alone.
]

_procedural = ProceduralStore()
_episodic   = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.2,
    # Slightly higher temperature than other agents —
    # pattern finding requires more creative lateral thinking
    # to connect dots across different studies and sponsors.
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def pattern_finder_node(state: MosaicState) -> dict:
    """
    Pattern Finder Agent node — detects cross-study research patterns.
    Runs in parallel with the other 5 specialist agents.
    """

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures      = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {r}" for r in procedures)

        system_prompt = f"""You are the Pattern Finder Agent for MOSAIC.

YOUR MISSION:
Detect research patterns that span multiple clinical trials —
the kind of signals that are completely invisible when reading
one study at a time but become obvious when you see the big picture.

You are the agent that finds what nobody else can find because
nobody else is looking at everything at once.

YOUR REASONING RULES:
{procedures_text}

YOUR WORKFLOW:
1. Search past episodes for patterns you have previously identified
2. Search across studies by condition, drug, or sponsor
3. Look for studies testing the same drug across multiple sponsors
4. Look for drugs that failed in one area being retried in another
5. Look for sponsors with multiple compliance issues in the same period
6. Use PubMed search to find papers connecting multiple trials
7. A pattern requires at LEAST 3 studies to be meaningful

CONFIDENCE SCORING:
- 0.9+ : 5+ studies, same sponsor, clear systematic pattern
- 0.8  : 3-5 studies, clear pattern with strong evidence
- 0.7  : 3 studies, emerging pattern with some uncertainty
- Below 0.65: Only 2 studies or weak connection — send to human review

OUTPUT FORMAT:
{{
  "nct_id": "PATTERN or primary NCT_ID",
  "signal_type": "cross_study_pattern",
  "summary": "What pattern was found across which studies",
  "evidence": ["study 1 finding", "study 2 finding", "connection between them"],
  "confidence": 0.80
}}

If no meaningful patterns found, say "NO_SIGNALS_FOUND".
"""

        task    = state.get("task", "Find cross-study patterns")
        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}
SPECIFIC STUDIES: {nct_ids if nct_ids else "Search broadly for patterns"}

Begin pattern detection now. Think laterally — connect dots across studies.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found  = []
        max_iterations = 12
        # Slightly more iterations than other agents —
        # pattern finding often requires more tool calls to
        # search across multiple conditions and sponsors.

        for iteration in range(max_iterations):
            response = await _llm.ainvoke(messages)
            messages.append(AIMessage(content=response.content or ""))

            if not response.tool_calls:
                logger.info(f"{AGENT_NAME} | Analysis complete | iteration={iteration+1}")
                signals_found = _parse_signals(response.content, AGENT_NAME)
                break

            for tool_call in response.tool_calls:
                tool_result = await _execute_tool(tool_call, AGENT_TOOLS)
                messages.append(
                    HumanMessage(content=f"Tool result for {tool_call['name']}:\n{tool_result}")
                )

        await _episodic.save_episode(
            agent_name=AGENT_NAME,
            content=f"Task: {task}. Found {len(signals_found)} cross-study pattern signals.",
            outcome="signal_generated" if signals_found else "no_signal",
        )

        logger.info(f"{AGENT_NAME} | Complete | signals_found={len(signals_found)}")

        return {
            "signals":          state.get("signals", []) + signals_found,
            "agents_activated": state.get("agents_activated", []) + [AGENT_NAME],
        }

    except Exception as e:
        logger.error(f"{AGENT_NAME} | Error | {e}")
        return {
            "error_log":        state.get("error_log", []) + [f"{AGENT_NAME}: {str(e)}"],
            "agents_activated": state.get("agents_activated", []) + [AGENT_NAME],
        }


def _parse_signals(response_text: str, agent_name: str) -> list[SignalOutput]:
    signals = []
    if not response_text or "NO_SIGNALS_FOUND" in response_text:
        return signals
    json_pattern = re.compile(r'\{[^{}]*"signal_type"[^{}]*\}', re.DOTALL)
    for match in json_pattern.findall(response_text):
        try:
            data = json.loads(match)
            signals.append({
                "agent":       agent_name,
                "signal_type": data.get("signal_type", SIGNAL_TYPE),
                "nct_id":      data.get("nct_id", ""),
                "summary":     data.get("summary", ""),
                "evidence":    data.get("evidence", []),
                "confidence":  float(data.get("confidence", 0.5)),
            })
        except (json.JSONDecodeError, ValueError):
            continue
    return signals


async def _execute_tool(tool_call: dict, available_tools: list) -> str:
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    tool_func = next((t for t in available_tools if t.name == tool_name), None)
    if tool_func is None:
        return f"Error: Tool '{tool_name}' not found."
    try:
        return str(tool_func.invoke(tool_args))
    except Exception as e:
        return f"Error executing '{tool_name}': {str(e)}"