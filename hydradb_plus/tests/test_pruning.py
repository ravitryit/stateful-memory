import datetime

from hydradb_plus.core.graph_engine import KnowledgeGraph
from hydradb_plus.contributions.graph_pruning.pruner import GraphPruner


def test_edge_deduplication_merges_duplicates() -> None:
    """GraphPruner should merge duplicate edges with same from/to/relation/value."""

    g = KnowledgeGraph()
    g.add_entity("A", "entity", metadata={"description": "prefer hate"})
    g.add_entity("object", "entity", metadata={})

    # Add duplicates: same from/to/relation/value but different contexts.
    for i in range(3):
        g.add_relationship("A", "object", "MENTIONS", "X", context=f"context_{i}")

    before_edges = g.graph.number_of_edges()
    assert before_edges == 3

    pruner = GraphPruner()
    report = pruner.run_pruning_cycle(g, memory_store=None)

    after_edges = g.graph.number_of_edges()
    assert after_edges == 1
    assert report["merged_count"] >= 2


def test_never_forget_pinning_prevents_archival() -> None:
    """Pinned nodes (never_forget) should not be archived even if score is low."""

    g = KnowledgeGraph()
    g.add_entity("C", "entity", metadata={"description": "casual mention"})
    # Force old created_at to make score decay extremely low.
    old_time = (datetime.datetime.now() - datetime.timedelta(days=365)).isoformat()
    g.graph.nodes["C"]["created_at"] = old_time
    g.graph.nodes["C"]["never_forget"] = True

    g.add_entity("object", "entity", metadata={})
    g.add_relationship("C", "object", "MENTIONS", "value", context="context")

    pruner = GraphPruner()
    _ = pruner.run_pruning_cycle(g, memory_store=None)

    assert g.graph.nodes["C"]["archived"] is False
    assert str(g.graph.nodes["C"].get("tier")) == "HOT"

