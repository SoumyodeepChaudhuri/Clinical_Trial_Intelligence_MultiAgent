##############################################################################
# agents/timeline_agent.py
#
# PURPOSE:
#   Detects studies that are significantly delayed past their own
#   stated completion date WITHOUT filing an explanation.
#
# THE DISTINCTION THAT MATTERS:
#   A disclosed delay = sponsor filed an amendment explaining the extension
#   → Acceptable. Trials get delayed. Transparency is the key.
#
#   A silent delay = trial is months/years past completion date,
#                    no amendment filed, no explanation given
#   → Suspicious. Why the silence?
#
#   This agent specifically hunts for SILENT delays — the ones
#   where something may have gone wrong but the sponsor went quiet.
#
# COVID EXCEPTION:
#   Delays between March 2020 and December 2022 often had legitimate
#   COVID-related causes. The procedural memory rules account for this.
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
    update_sponsor_profile,
)
from tools.clinical_tools import (
    fetch_study_details,
    check_results_posted,
    get_study_amendments,
)
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

AGENT_NAME  = "timeline_agent"
SIGNAL_TYPE = "timeline_delay"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    fetch_study_details,
    check_results_posted,
    get_study_amendments,
    # get_study_amendments is critical here —
    # it tells us whether a delay was DISCLOSED (amendment filed)
    # or SILENT (no amendment, no explanation). Silent is the red flag.
]

_procedural = ProceduralStore()
_episodic   = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def timeline_node(state: MosaicState) -> dict:
    """
    Timeline Agent node — detects silent delays in clinical trials.
    Runs in parallel with the other 5 specialist agents.
    """

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures      = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {r}" for r in procedures)

        system_prompt = f"""You are the Timeline Analyst Agent for MOSAIC.

YOUR MISSION:
Find clinical trials that are significantly past their stated completion date
WITHOUT having filed any amendment or explanation for the delay.

A disclosed delay is acceptable. A silent delay — where a trial goes
quiet past its deadline — is a red flag that warrants investigation.

YOUR REASONING RULES:
{procedures_text}

YOUR WORKFLOW:
1. Check past episodes for previous timeline investigations
2. Search for studies that appear delayed or past their completion date
3. Fetch full study details to confirm completion date and current status
4. Check amendments — was an extension officially filed?
5. If no amendment and significantly delayed → generate signal
6. Check sponsor profile — is this a pattern for this sponsor?
7. Update sponsor profile with delay information

SILENCE THRESHOLD:
Only flag delays greater than 180 days (6 months) past completion date
with NO amendment filed. Minor delays are expected and acceptable.

CONFIDENCE SCORING:
- 0.9+ : 2+ years past completion, no amendment, no results, repeat offender
- 0.8  : 1-2 years past completion, no amendment, sponsor has prior delays
- 0.7  : 6-12 months past completion, no amendment, first-time delay
- 0.6  : 6 months past, no amendment, COVID period overlap possible
- Below 0.60: Borderline timing — send to human review

OUTPUT FORMAT:
{{
  "nct_id": "NCT_ID",
  "signal_type": "timeline_delay",
  "summary": "How many months/years delayed, whether amendment was filed",
  "evidence": ["completion date was X", "today is Y", "no amendment found"],
  "confidence": 0.80
}}

If no timeline violations found, say "NO_SIGNALS_FOUND".
"""

        task    = state.get("task", "Find trials with unexplained timeline delays")
        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}
SPECIFIC STUDIES: {nct_ids if nct_ids else "Search broadly for delayed trials"}

Begin timeline investigation now. Focus on silent delays — no amendment filed.
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
            content=f"Task: {task}. Found {len(signals_found)} timeline delay signals.",
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