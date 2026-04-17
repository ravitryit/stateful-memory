from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.traceback import install as rich_install

from core.graph_engine import KnowledgeGraph

rich_install(show_locals=False)
console = Console()


@dataclass(frozen=True)
class PruningReport:
    """Structured output for a pruning cycle."""

    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    archived_count: int
    merged_count: int
    size_reduction_percent: float
    tier_stats: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a dict."""

        return {
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "edges_before": self.edges_before,
            "edges_after": self.edges_after,
            "archived_count": self.archived_count,
            "merged_count": self.merged_count,
            "size_reduction_percent": self.size_reduction_percent,
            "tier_stats": dict(self.tier_stats),
        }


class GraphPruner:
    """Graph pruning engine combining confidence scoring, tiering and deduplication."""

    def __init__(self) -> None:
        """Initialize pruning engine with required helpers."""

        from .confidence_scorer import ConfidenceScorer
        from .tiered_storage import TieredStorage

        self._confidence_scorer = ConfidenceScorer()
        self._tiered_storage = TieredStorage()

    def _node_importance_text(self, graph: KnowledgeGraph, node_id: str) -> str:
        """Derive a lightweight importance text for the node from recent outgoing edges."""

        try:
            importance_chunks: List[str] = []
            # Find top few recent edges by tcommit.
            edges: List[Tuple[str, Dict[str, Any]]] = []
            for _, to_entity, key, data in graph.graph.out_edges(node_id, keys=True, data=True):
                _ = to_entity, key
                tcommit = data.get("tcommit")
                if isinstance(tcommit, str):
                    edges.append((tcommit, data))
                else:
                    edges.append((datetime.fromtimestamp(0).isoformat(), data))
            edges = sorted(edges, key=lambda x: x[0])[-5:]
            for _, data in edges:
                importance_chunks.append(str(data.get("value", "")) + " " + str(data.get("context", "")))
            meta = graph.graph.nodes[node_id].get("metadata", {})
            if isinstance(meta, dict):
                importance_chunks.append(str(meta.get("description", "")))
            return " ".join(importance_chunks)
        except Exception:
            return ""

    def run_pruning_cycle(self, graph: KnowledgeGraph, memory_store: Optional[Any] = None) -> Dict[str, Any]:
        """Run a pruning cycle and return a detailed pruning report.

        Steps:
        a. Calculate confidence for ALL nodes
        b. Assign tiers
        c. Nodes with score < 0.1 -> archive (flag as archived=True)
        d. Merge duplicate edges by (from_entity, to_entity, relation_type, value)
           while merging access_history into the kept edge.
        e. Return pruning_report
        """

        try:
            stats_before = graph.get_graph_stats()
            nodes_before = graph.graph.number_of_nodes()
            edges_before = graph.graph.number_of_edges()

            current_time = datetime.now()
            archived_count = 0
            merged_count = 0

            # Confidence + tier assignment
            for node_id, node_data in graph.graph.nodes(data=True):
                access_history = node_data.get("access_history", [])
                importance_text = self._node_importance_text(graph, node_id)
                score = self._confidence_scorer.calculate_score(
                    node=node_data,
                    current_time=current_time,
                    access_history=access_history if isinstance(access_history, list) else [],
                    importance_text=importance_text,
                )
                node_data["confidence_score"] = score
                pinned = bool(node_data.get("never_forget", False))
                node_data["archived"] = (score < 0.1) and (not pinned)
                if pinned:
                    # Pinned nodes behave as always relevant for retention.
                    score = max(score, 0.8)
                    node_data["confidence_score"] = score
                node_data["tier"] = self._tiered_storage.assign_tier(node_id, score)
                if node_data["archived"]:
                    archived_count += 1
                    # Reduce memory footprint of archived nodes: clear history and heavy metadata.
                    node_data["access_history"] = []
                    if "metadata" in node_data and isinstance(node_data["metadata"], dict):
                        # Keep metadata keys but trim large blobs if present.
                        for k, v in list(node_data["metadata"].items()):
                            if isinstance(v, str) and len(v) > 200:
                                node_data["metadata"][k] = v[:200]

            # Trim archived-node edge payloads without deleting edges (shrinks pickled graph size).
            archived_node_ids = {str(nid) for nid, ndata in graph.graph.nodes(data=True) if bool(ndata.get("archived", False))}
            for u, v, key, data in list(graph.graph.edges(keys=True, data=True)):
                if str(u) in archived_node_ids or str(v) in archived_node_ids:
                    if "context" in data:
                        data["context"] = ""
                    if "access_history" in data and isinstance(data["access_history"], list):
                        data["access_history"] = []
                    data["archived"] = True

            # Deduplicate edges (merge duplicates into latest by tcommit).
            # Group key: (from_entity, to_entity, relation_type, value)
            edge_groups: Dict[Tuple[str, str, str, str], List[Tuple[str, str, str, Dict[str, Any]]]] = {}
            for u, v, key, data in graph.graph.edges(keys=True, data=True):
                relation_type = str(data.get("relation_type"))
                value = str(data.get("value"))
                gkey = (str(u), str(v), relation_type, value)
                tcommit = str(data.get("tcommit", "")) or "0000-01-01T00:00:00"
                edge_groups.setdefault(gkey, []).append(
                    (tcommit, str(u), str(v), {**data, "_key": str(key)})
                )

            for _, group in edge_groups.items():
                if len(group) <= 1:
                    continue
                # Keep latest tcommit.
                group_sorted = sorted(group, key=lambda x: x[0])
                _, kept_u, kept_v, kept_data = group_sorted[-1]
                kept_key = kept_data["_key"]
                merged_access: List[str] = []
                # Access history on edges is optional, but we support it.
                for _, _, _, item_data in group_sorted:
                    access = item_data.get("access_history", [])
                    if isinstance(access, list):
                        merged_access.extend([str(x) for x in access])

                # Update kept edge directly by location + key.
                try:
                    if graph.graph.has_edge(kept_u, kept_v, key=kept_key):
                        graph.graph[kept_u][kept_v][kept_key]["access_history"] = merged_access
                except Exception:
                    # Fallback: skip update if edge disappeared unexpectedly.
                    pass

                # Remove all other edges in the group by their stored keys.
                for _, rem_u, rem_v, rem_data in group_sorted[:-1]:
                    rem_key = str(rem_data.get("_key"))
                    if graph.graph.has_edge(rem_u, rem_v, key=rem_key):
                        graph.graph.remove_edge(rem_u, rem_v, key=rem_key)
                        merged_count += 1

            stats_after = graph.get_graph_stats()
            nodes_after = graph.graph.number_of_nodes()
            edges_after = graph.graph.number_of_edges()

            size_reduction_percent = 0.0
            try:
                if stats_before.memory_size_mb > 0.0:
                    size_reduction_percent = max(
                        0.0,
                        min(
                            100.0,
                            ((stats_before.memory_size_mb - stats_after.memory_size_mb) / stats_before.memory_size_mb) * 100.0,
                        ),
                    )
            except Exception:
                size_reduction_percent = 0.0

            report = PruningReport(
                nodes_before=nodes_before,
                nodes_after=nodes_after,
                edges_before=edges_before,
                edges_after=edges_after,
                archived_count=archived_count,
                merged_count=merged_count,
                size_reduction_percent=size_reduction_percent,
                tier_stats=self._tiered_storage.get_tier_stats(),
            )
            return report.to_dict()
        except Exception as e:
            console.print("[red]GraphPruner.run_pruning_cycle failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def benchmark_pruning(self, graph: KnowledgeGraph, memory_store: Optional[Any] = None) -> Dict[str, Any]:
        """Benchmark graph retrieval latency before/after pruning.

        This benchmark measures how long it takes to pull recent relationship
        histories for a sample of nodes/relations.
        """

        try:
            # Collect candidate entities.
            node_ids = list(graph.graph.nodes())
            if not node_ids:
                return {"before_ms": 0.0, "after_ms": 0.0}

            # Pick a fixed subset for repeatability.
            random.seed(42)
            sample_nodes = node_ids[: min(30, len(node_ids))]

            def _retrieve_sample() -> None:
                for n in sample_nodes:
                    # For each node, query a few relations found on outgoing edges.
                    relations = list({str(d.get("relation_type")) for _, _, _, d in graph.graph.out_edges(n, keys=True, data=True)})
                    for rel in relations[:3]:
                        _ = graph.get_full_history(n, rel)

            start_before = time.perf_counter()
            _retrieve_sample()
            before_ms = (time.perf_counter() - start_before) * 1000.0

            # Run pruning once.
            _ = self.run_pruning_cycle(graph, memory_store=memory_store)

            start_after = time.perf_counter()
            _retrieve_sample()
            after_ms = (time.perf_counter() - start_after) * 1000.0

            return {"before_ms": before_ms, "after_ms": after_ms}
        except Exception as e:
            console.print("[red]GraphPruner.benchmark_pruning failed[/red]")
            console.print_exception(show_locals=False)
            raise e
