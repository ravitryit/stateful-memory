from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


def _iso_now() -> str:
    """Return the current time as an ISO 8601 datetime string."""

    return datetime.now().isoformat()


def _to_iso_datetime_str(value: Optional[object]) -> Optional[str]:
    """Convert supported datetime-like values to ISO strings."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        # Covers datetime and similar objects
        return value.isoformat()  # type: ignore[no-any-return]
    raise TypeError(f"Unsupported datetime value type: {type(value)!r}")


@dataclass(frozen=True)
class RelationshipState:
    """Represents a single relationship edge's current or historical state."""

    from_entity: str
    to_entity: str
    relation_type: str
    value: str
    tcommit: str
    tvalid: Optional[str]
    context: str
    confidence_score: float
    edge_key: Optional[str] = None
    archived: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphStats:
    """Return-type for knowledge graph statistics."""

    total_nodes: int
    total_edges: int
    avg_confidence: float
    memory_size_mb: float


class KnowledgeGraph:
    """Append-only knowledge graph using a NetworkX MultiDiGraph.

    Notes:
    - Nodes represent entities (entity_id) and store creation time.
    - Edges represent relationships and are appended without overwriting.
    - Each edge key is unique, enabling multiple edges between the same nodes.
    """

    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""

        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_entity(self, entity_id: str, entity_type: str, metadata: Dict[str, Any]) -> None:
        """Add an entity node to the graph if it does not already exist.

        If the entity already exists, do not duplicate the node. Metadata keys
        are merged only when missing to avoid overwriting.
        """

        try:
            if entity_id not in self.graph:
                self.graph.add_node(
                    entity_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    created_at=_iso_now(),
                    metadata=dict(metadata),
                    access_history=[],
                    archived=False,
                )
                return

            node_data = self.graph.nodes[entity_id]
            if "metadata" not in node_data or not isinstance(node_data.get("metadata"), dict):
                node_data["metadata"] = {}
            for k, v in metadata.items():
                if k not in node_data["metadata"]:
                    node_data["metadata"][k] = v
        except Exception as e:
            console.print("[red]KnowledgeGraph.add_entity failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relation: str,
        value: str,
        tvalid: Optional[object] = None,
        context: str = "",
        confidence_score: float = 1.0,
        archived: bool = False,
    ) -> str:
        """Append a new relationship edge (Git-style).

        The method never overwrites an existing edge. It always adds a new edge
        with a unique edge key and assigns:
        - tcommit = ingestion time (datetime.now())
        - confidence_score = 1.0 (default)
        """

        try:
            if from_entity not in self.graph:
                self.add_entity(from_entity, "entity", {})
            if to_entity not in self.graph:
                self.add_entity(to_entity, "entity", {})

            tcommit = _iso_now()
            tvalid_str = _to_iso_datetime_str(tvalid)
            edge_key = uuid.uuid4().hex

            self.graph.add_edge(
                from_entity,
                to_entity,
                key=edge_key,
                relation_type=relation,
                value=value,
                tcommit=tcommit,
                tvalid=tvalid_str,
                context=context,
                confidence_score=float(confidence_score),
                access_history=[],
                archived=bool(archived),
            )
            return str(edge_key)
        except Exception as e:
            console.print("[red]KnowledgeGraph.add_relationship failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def _iter_relationship_edges(self, entity_id: str, relation: str) -> List[tuple[str, str, str, Dict[str, Any]]]:
        """Iterate edges matching `entity_id` as source and `relation` as relation_type."""

        edges: List[tuple[str, str, str, Dict[str, Any]]] = []
        for to_entity, _, key, data in self.graph.out_edges(entity_id, keys=True, data=True):
            if data.get("relation_type") == relation:
                edges.append((entity_id, to_entity, key, data))
        return edges

    def get_current_state(self, entity_id: str, relation: str) -> Optional[RelationshipState]:
        """Return the most recent edge by `tcommit` for a given entity and relation."""

        try:
            edges = self._iter_relationship_edges(entity_id, relation)
            if not edges:
                return None

            def _sort_key(item: tuple[str, str, str, Dict[str, Any]]) -> datetime:
                tcommit = item[3].get("tcommit")
                if not isinstance(tcommit, str):
                    return datetime.fromtimestamp(0)
                return datetime.fromisoformat(tcommit)

            latest = max(edges, key=_sort_key)
            from_entity, to_entity, edge_key, data = latest
            return RelationshipState(
                from_entity=from_entity,
                to_entity=to_entity,
                relation_type=str(data.get("relation_type")),
                value=str(data.get("value")),
                tcommit=str(data.get("tcommit")),
                tvalid=data.get("tvalid"),
                context=str(data.get("context", "")),
                confidence_score=float(data.get("confidence_score", 1.0)),
                edge_key=str(edge_key),
                archived=bool(data.get("archived", False)),
                extra={k: v for k, v in data.items() if k not in {
                    "relation_type",
                    "value",
                    "tcommit",
                    "tvalid",
                    "context",
                    "confidence_score",
                    "access_history",
                    "archived",
                }},
            )
        except Exception as e:
            console.print("[red]KnowledgeGraph.get_current_state failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_full_history(self, entity_id: str, relation: str) -> List[RelationshipState]:
        """Return ALL edges sorted by `tcommit` (complete timeline)."""

        try:
            edges = self._iter_relationship_edges(entity_id, relation)

            def _sort_key(item: tuple[str, str, str, Dict[str, Any]]) -> datetime:
                tcommit = item[3].get("tcommit")
                if not isinstance(tcommit, str):
                    return datetime.fromtimestamp(0)
                return datetime.fromisoformat(tcommit)

            edges_sorted = sorted(edges, key=_sort_key)
            history: List[RelationshipState] = []
            for from_entity, to_entity, edge_key, data in edges_sorted:
                history.append(
                    RelationshipState(
                        from_entity=from_entity,
                        to_entity=to_entity,
                        relation_type=str(data.get("relation_type")),
                        value=str(data.get("value")),
                        tcommit=str(data.get("tcommit")),
                        tvalid=data.get("tvalid"),
                        context=str(data.get("context", "")),
                        confidence_score=float(data.get("confidence_score", 1.0)),
                        edge_key=str(edge_key),
                        archived=bool(data.get("archived", False)),
                        extra={k: v for k, v in data.items() if k not in {
                            "relation_type",
                            "value",
                            "tcommit",
                            "tvalid",
                            "context",
                            "confidence_score",
                            "access_history",
                            "archived",
                        }},
                    )
                )
            return history
        except Exception as e:
            console.print("[red]KnowledgeGraph.get_full_history failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_graph_stats(self) -> GraphStats:
        """Compute graph statistics including a rough memory usage estimate."""

        try:
            total_nodes = self.graph.number_of_nodes()
            total_edges = self.graph.number_of_edges()
            confidences: List[float] = []
            for _, _, _, data in self.graph.edges(keys=True, data=True):
                confidences.append(float(data.get("confidence_score", 1.0)))
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            try:
                blob = pickle.dumps(self.graph, protocol=pickle.HIGHEST_PROTOCOL)
                memory_size_mb = len(blob) / (1024 * 1024)
            except Exception:
                memory_size_mb = 0.0

            return GraphStats(
                total_nodes=total_nodes,
                total_edges=total_edges,
                avg_confidence=avg_confidence,
                memory_size_mb=memory_size_mb,
            )
        except Exception as e:
            console.print("[red]KnowledgeGraph.get_graph_stats failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_entity_history(self, entity_name: str) -> List[Dict[str, Any]]:
        """Return a timeline of all edges connected to an entity (case-insensitive)."""

        try:
            query = (entity_name or "").strip().lower()
            if not query:
                return []

            matched_nodes = [
                str(n)
                for n in self.graph.nodes()
                if query == str(n).lower() or query in str(n).lower() or str(n).lower() in query
            ]
            if not matched_nodes:
                return []

            timeline: List[Dict[str, Any]] = []
            for node in matched_nodes:
                for u, v, key, data in self.graph.in_edges(node, keys=True, data=True):
                    timeline.append(
                        {
                            "tcommit": str(data.get("tcommit", "")),
                            "from_entity": str(u),
                            "to_entity": str(v),
                            "relation_type": str(data.get("relation_type", "")),
                            "value": str(data.get("value", "")),
                            "context": str(data.get("context", "")),
                            "sentiment_score": data.get("sentiment_score"),
                            "intensity_label": data.get("intensity_label"),
                            "edge_key": str(key),
                        }
                    )
                for u, v, key, data in self.graph.out_edges(node, keys=True, data=True):
                    timeline.append(
                        {
                            "tcommit": str(data.get("tcommit", "")),
                            "from_entity": str(u),
                            "to_entity": str(v),
                            "relation_type": str(data.get("relation_type", "")),
                            "value": str(data.get("value", "")),
                            "context": str(data.get("context", "")),
                            "sentiment_score": data.get("sentiment_score"),
                            "intensity_label": data.get("intensity_label"),
                            "edge_key": str(key),
                        }
                    )

            # Remove duplicate edges gathered from both directions.
            seen = set()
            deduped: List[Dict[str, Any]] = []
            for item in timeline:
                ek = item.get("edge_key")
                if ek in seen:
                    continue
                seen.add(ek)
                deduped.append(item)

            return sorted(
                deduped,
                key=lambda d: datetime.fromisoformat(d["tcommit"]) if d.get("tcommit") else datetime.fromtimestamp(0),
            )
        except Exception as e:
            console.print("[red]KnowledgeGraph.get_entity_history failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_all_relations(self) -> List[str]:
        """Return all unique relation types currently present in the graph."""

        try:
            relation_types = {
                str(data.get("relation_type"))
                for _, _, _, data in self.graph.edges(keys=True, data=True)
                if data.get("relation_type") is not None
            }
            return sorted(relation_types)
        except Exception as e:
            console.print("[red]KnowledgeGraph.get_all_relations failed[/red]")
            console.print_exception(show_locals=False)
            raise e

