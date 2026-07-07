##############################################################################
# api/dependencies.py
#
# PURPOSE:
#   Creates and manages shared resources that API endpoints need.
#
# WHAT IS A FASTAPI DEPENDENCY?
#   A dependency is a function that FastAPI calls AUTOMATICALLY before
#   running an endpoint. The endpoint receives the dependency's return
#   value as a parameter.
#
#   Example:
#     @app.get("/signals")
#     async def get_signals(store: VectorStore = Depends(get_vector_store)):
#         ...
#
#   FastAPI sees "Depends(get_vector_store)", calls get_vector_store(),
#   and passes the result as the "store" parameter automatically.
#   The endpoint never creates the store itself — it just uses it.
#
# WHY USE @lru_cache?
#   lru_cache = Least Recently Used Cache.
#   It makes a function return the SAME object every time it is called
#   instead of creating a new one each time.
#
#   Without lru_cache:
#     get_hitl_gate() creates a new HITLGate() on every API request.
#     Every request opens a new database connection pool. Very wasteful.
#
#   With lru_cache:
#     get_hitl_gate() creates ONE HITLGate() on the first call.
#     Every subsequent call returns that same object.
#     One connection pool shared across all requests. Efficient.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. api/main.py and routers import from this file.
##############################################################################


from functools import lru_cache
# lru_cache is Python's built-in memoisation decorator.
# Applied to a function, it caches the result of the first call
# and returns that cached result on all subsequent calls.
# Perfect for expensive-to-create objects like database connection pools.

from graph.hitl import HITLGate
from memory.episodic_store import EpisodicStore
from memory.procedural_store import ProceduralStore
from memory.semantic_store import SemanticStore
from graph.graph_builder import mosaic_graph
# Import the pre-compiled MOSAIC graph from graph_builder.py.
# mosaic_graph was built ONCE at module import time.
# We expose it through a dependency so endpoints can access it cleanly.


@lru_cache(maxsize=1)
def get_hitl_gate() -> HITLGate:
    """
    Returns the shared HITLGate instance.

    maxsize=1 means: cache the result of the first call.
    Every subsequent call returns the same HITLGate object.
    The HITLGate holds a database connection pool — we want
    exactly ONE pool, not a new one per request.

    Returns:
        The singleton HITLGate instance.
    """
    return HITLGate()


@lru_cache(maxsize=1)
def get_episodic_store() -> EpisodicStore:
    """
    Returns the shared EpisodicStore instance.
    One instance = one connection pool = efficient resource use.
    """
    return EpisodicStore()


@lru_cache(maxsize=1)
def get_procedural_store() -> ProceduralStore:
    """
    Returns the shared ProceduralStore instance.
    """
    return ProceduralStore()


@lru_cache(maxsize=1)
def get_semantic_store() -> SemanticStore:
    """
    Returns the shared SemanticStore instance.
    """
    return SemanticStore()


def get_graph():
    """
    Returns the compiled MOSAIC LangGraph graph.

    NOT cached with lru_cache because mosaic_graph is already
    a module-level singleton — it was compiled once in graph_builder.py
    and importing it here gives us the same object every time.

    Returns:
        The compiled LangGraph graph ready to invoke.
    """
    return mosaic_graph