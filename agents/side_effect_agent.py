##############################################################################
# agents/side_effect_agent.py
#
# PURPOSE:
#   Compares what official trial filings say about safety against what
#   published research papers report about the same trial.
#
# THE CORE INSIGHT:
#   Official filings are written by the sponsor — the entity with the
#   most to lose if safety problems are disclosed.
#   Published papers are written by independent researchers who have
#   no financial incentive to hide safety concerns.
#
#   When these two sources DISAGREE, that disagreement is a signal.
#
#   Example:
#   Official filing: "No serious adverse events observed"
#   Published paper: "Three patients were hospitalised with cardiac events"
#   → This gap is what this agent hunts for.
#
# LOWEST CONFIDENCE THRESHOLD (0.55):
#   We set the lowest threshold of all agents here.
#   Safety concerns should err on the side of caution —
#   we would rather send a borderline safety signal for human review
#   than miss a genuine patient safety issue.
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
)
from tools.clinical_tools import fetch_study_details
from tools.pubmed_tools import fetch_papers_for_trial, compare_filing_vs_papers
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

AGENT_NAME  = "side_effect_agent"
SIGNAL_TYPE = "safety_gap"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    fetch_study_details,
    fetch_papers_for_trial,
    compare_filing_vs_papers,
    # compare_filing_vs_papers is the signature tool of this agent.
    # It fetches both the official filing summary AND all published papers
    # in one call — giving the agent everything it needs to compare.
]

_procedural = ProceduralStore()
_episodic   = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def side_effect_node(state: MosaicState) -> dict:
    """
    Side Effect Checker Agent node — finds safety gaps between filings and papers.
    Runs in parallel with the other 5 specialist agents.
    """

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures      = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {r}" for r in procedures)

        system_prompt = f"""You are the Side Effect Checker Agent for MOSAIC.

YOUR MISSION:
Find cases where official clinical trial safety reports DISAGREE with
what independent researchers published in peer-reviewed journals.

Official filings are written by sponsors. Papers are written by independent
scientists. When they tell different stories about the same trial,
patients and regulators deserve to know.

YOUR REASONING RULES:
{procedures_text}

YOUR WORKFLOW:
1. Check past episodes for previous safety gap investigations
2. Search for studies where safety is a concern
3. Use compare_filing_vs_papers to get both official data and published papers
4. Compare what the filing says about safety vs what papers report
5. Flag cases where papers mention serious events absent from filings
6. Be extra careful — safety signals should always err toward caution

WHAT TO LOOK FOR:
- Filing says "no serious adverse events" but papers mention hospitalisations
- Filing reports mild side effects, papers report the same events as severe
- Papers report deaths or discontinuations not mentioned in filing
- Results published in papers when official results were never posted

CONFIDENCE SCORING (lowest threshold of all agents — 0.55):
- 0.9+ : Filing says "no SAEs", paper explicitly describes hospitalisations/deaths
- 0.8  : Clear severity difference for the same event (mild vs serious)
- 0.7  : Additional side effects in papers not mentioned in filing
- 0.6  : Minor terminology differences that suggest downplaying
- 0.55 : Possible discrepancy — needs human expert review

OUTPUT FORMAT:
{{
  "nct_id": "NCT_ID",
  "signal_type": "safety_gap",
  "summary": "What the filing said vs what papers reported",
  "evidence": ["filing claim", "paper contradicts with X"],
  "confidence": 0.80
}}

If no gaps found, say "NO_SIGNALS_FOUND".
"""

        task    = state.get("task", "Find safety gaps between filings and papers")
        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}
SPECIFIC STUDIES: {nct_ids if nct_ids else "Search for studies with published papers"}

Begin safety comparison now. Prioritise patient safety — when in doubt, flag it.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found  = []
        max_iterations = 10

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
            content=f"Task: {task}. Found {len(signals_found)} safety gap signals.",
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