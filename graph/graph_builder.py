##############################################################################
# graph/graph_builder.py
#
# Wires all agents into a LangGraph StateGraph.
# Defines which agents run, in what order, and how state flows.
#
# GRAPH STRUCTURE:
#   START
#     → supervisor_route  (decides which specialists to run)
#     → [all 6 specialists run in PARALLEL]
#     → supervisor_compile (reads all signals, writes final brief)
#   END
##############################################################################

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from graph.state import MosaicState
from agents.supervisor import supervisor_route, supervisor_compile
from agents.broken_promises_agent import broken_promises_node
from agents.missing_results_agent import missing_results_node
from agents.track_record_agent import track_record_node
from agents.pattern_finder_agent import pattern_finder_node
from agents.side_effect_agent import side_effect_node
from agents.timeline_agent import timeline_node

from tools.search_tools import ALL_SEARCH_TOOLS
from tools.clinical_tools import ALL_CLINICAL_TOOLS
from tools.pubmed_tools import ALL_PUBMED_TOOLS

from config.logging_config import setup_logging

logger = setup_logging(__name__)

# All tools combined — passed to ToolNode so LangGraph can
# execute tool calls made by agents automatically.
ALL_TOOLS = ALL_SEARCH_TOOLS + ALL_CLINICAL_TOOLS + ALL_PUBMED_TOOLS


def build_mosaic_graph():
    """
    Builds and compiles the complete MOSAIC agent graph.

    Returns a compiled LangGraph graph ready to invoke.
    Call this once at startup and reuse the compiled graph.
    """

    logger.info("Building MOSAIC agent graph...")

    # ── CREATE THE GRAPH ───────────────────────────────────────
    graph = StateGraph(MosaicState)
    # StateGraph takes our MosaicState TypedDict.
    # Every node in the graph receives and returns MosaicState.

    # ── TOOL NODE ──────────────────────────────────────────────
    # ToolNode is a special LangGraph node that executes tool calls.
    # When an agent returns a message with tool_calls, LangGraph
    # routes to this node which runs the actual tool functions.
    tool_node = ToolNode(ALL_TOOLS)

    # ── ADD ALL NODES ──────────────────────────────────────────
    graph.add_node("supervisor_route",   supervisor_route)
    graph.add_node("broken_promises",    broken_promises_node)
    graph.add_node("missing_results",    missing_results_node)
    graph.add_node("track_record",       track_record_node)
    graph.add_node("pattern_finder",     pattern_finder_node)
    graph.add_node("side_effect",        side_effect_node)
    graph.add_node("timeline",           timeline_node)
    graph.add_node("tools",              tool_node)
    graph.add_node("supervisor_compile", supervisor_compile)

    logger.info("Node added: broken_promises_agent")
    logger.info("Node added: missing_results_agent")
    logger.info("Node added: track_record_agent")
    logger.info("Node added: pattern_finder_agent")
    logger.info("Node added: side_effect_agent")
    logger.info("Node added: timeline_agent")

    # ── ENTRY POINT ────────────────────────────────────────────
    graph.add_edge(START, "supervisor_route")
    # Every run starts at supervisor_route — the supervisor reads
    # the task and decides what to do next.

    # ── SUPERVISOR → ALL SPECIALISTS (PARALLEL) ────────────────
    # All 6 specialists receive the same state simultaneously.
    # LangGraph runs them in parallel — not sequentially.
    # This means a 6-agent run takes as long as the SLOWEST agent,
    # not the SUM of all agent times. Massive performance gain.
    graph.add_edge("supervisor_route", "broken_promises")
    graph.add_edge("supervisor_route", "missing_results")
    graph.add_edge("supervisor_route", "track_record")
    graph.add_edge("supervisor_route", "pattern_finder")
    graph.add_edge("supervisor_route", "side_effect")
    graph.add_edge("supervisor_route", "timeline")

    # ── ALL SPECIALISTS → COMPILE ──────────────────────────────
    # After all specialists finish, supervisor_compile runs.
    # It reads all signals from state and writes the final brief.
    graph.add_edge("broken_promises",    "supervisor_compile")
    graph.add_edge("missing_results",    "supervisor_compile")
    graph.add_edge("track_record",       "supervisor_compile")
    graph.add_edge("pattern_finder",     "supervisor_compile")
    graph.add_edge("side_effect",        "supervisor_compile")
    graph.add_edge("timeline",           "supervisor_compile")

    # ── COMPILE → END ──────────────────────────────────────────
    graph.add_edge("supervisor_compile", END)

    # ── COMPILE THE GRAPH ──────────────────────────────────────
    compiled = graph.compile()
    # compile() validates all edges, checks all nodes are reachable,
    # and returns a runnable graph object.

    logger.info(
        f"MOSAIC graph compiled successfully | nodes=8"
    )

    return compiled


# Build the graph once at module import time.
# All API requests share this single compiled graph instance.
# Building it once at startup avoids rebuild overhead per request.
mosaic_graph = build_mosaic_graph()