"""Core building blocks for HydraDB++."""

__all__ = [
    "KnowledgeGraph",
    "GraphStats",
    "RelationshipState",
    "TemporalEngine",
    "TemporalCommit",
    "MemoryStore",
    "MemoryResult",
    "MemoryStats",
]


def __getattr__(name: str):
    """Lazily expose core exports to avoid optional heavy imports during collection."""

    if name in {"KnowledgeGraph", "GraphStats", "RelationshipState"}:
        from .graph_engine import GraphStats, KnowledgeGraph, RelationshipState

        return {
            "KnowledgeGraph": KnowledgeGraph,
            "GraphStats": GraphStats,
            "RelationshipState": RelationshipState,
        }[name]
    if name in {"TemporalEngine", "TemporalCommit"}:
        from .temporal_engine import TemporalCommit, TemporalEngine

        return {"TemporalEngine": TemporalEngine, "TemporalCommit": TemporalCommit}[name]
    if name in {"MemoryStore", "MemoryResult", "MemoryStats"}:
        from .memory_store import MemoryResult, MemoryStats, MemoryStore

        return {
            "MemoryStore": MemoryStore,
            "MemoryResult": MemoryResult,
            "MemoryStats": MemoryStats,
        }[name]
    raise AttributeError(name)

