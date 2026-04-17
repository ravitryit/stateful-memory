from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
from rich.console import Console
from rich.traceback import install as rich_install

from hydradb_plus.core.graph_engine import KnowledgeGraph, RelationshipState

rich_install(show_locals=False)
console = Console()


def _iso_now() -> str:
    """Return the current time as an ISO 8601 datetime string."""

    return datetime.now().isoformat()


class SentimentGraph(KnowledgeGraph):
    """Knowledge graph specialization for sentiment memory.

    Sentiment edges:
        relation_type: "FEELS_ABOUT"
        value: entity_name
        sentiment_score: float
        intensity_label: str
        raw_text: str
        tcommit: datetime (ISO string, via append-only logic)
    """

    SENTIMENT_RELATION = "FEELS_ABOUT"
    STRONG_LABELS = {"STRONG_POSITIVE", "STRONG_NEGATIVE"}

    def _normalize_entity(self, entity: str) -> str:
        """Normalize sentiment entity keys for robust matching."""

        return (entity or "").strip().lower()

    def store_sentiment(self, user_id: str, entity: str, sentiment_data: Dict[str, Any]) -> None:
        """Store sentiment as an append-only edge in the graph.

        Args:
            user_id: Identifier for the speaker/user.
            entity: Entity about which the sentiment is expressed.
            sentiment_data: Dict containing at least:
                - sentiment_score (float)
                - intensity_label (str)
                - raw_text (str)
        """

        try:
            original_entity = str(entity).strip()
            entity = self._normalize_entity(original_entity)
            user_id = str(user_id).strip()
            self.add_entity(user_id, entity_type="user", metadata={})
            self.add_entity(entity, entity_type="entity", metadata={})
            if original_entity and original_entity != entity and original_entity not in self.graph:
                self.add_entity(
                    original_entity,
                    entity_type="entity_alias",
                    metadata={"canonical_entity": entity},
                )

            edge_key = uuid.uuid4().hex
            tcommit = _iso_now()

            sentiment_score = float(
                sentiment_data.get(
                    "sentiment_score",
                    sentiment_data.get("intensity_score", sentiment_data.get("combined_score", 0.0)),
                )
            )
            intensity_label = str(sentiment_data.get("intensity_label", sentiment_data.get("intensity", "NEUTRAL")))
            raw_text = str(sentiment_data.get("raw_text", ""))

            # Skip duplicate sentiment entries for same user+entity+intensity.
            existing = self.get_current_sentiment(user_id, entity)
            if existing and str(existing.get("intensity_label", "")) == intensity_label:
                return

            self.graph.add_edge(
                user_id,
                entity,
                key=edge_key,
                relation_type=self.SENTIMENT_RELATION,
                value=entity,
                tcommit=tcommit,
                tvalid=sentiment_data.get("tvalid"),
                context=raw_text,
                confidence_score=1.0,
                access_history=[],
                archived=False,
                sentiment_score=sentiment_score,
                intensity_label=intensity_label,
                raw_text=raw_text,
            )
            console.print(f"[dim]Stored sentiment for {entity}: {sentiment_score}[/dim]")

            # High salience: never forget strong sentiments.
            if intensity_label in self.STRONG_LABELS:
                pin_targets = [entity]
                if original_entity and original_entity in self.graph:
                    pin_targets.append(original_entity)

                for target in pin_targets:
                    node_data = self.graph.nodes[target]
                    node_data["never_forget"] = True
                    # Seed access history so the retention score starts high.
                    ah = node_data.get("access_history")
                    if not isinstance(ah, list):
                        node_data["access_history"] = []
                        ah = node_data["access_history"]
                    # Add a few "pin" accesses at near-now.
                    now = datetime.now()
                    # Use minute-level offsets so ISO parsing works.
                    for i in range(3):
                        ts = (now).isoformat()
                        ah.append(ts)
        except Exception as e:
            console.print("[red]SentimentGraph.store_sentiment failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def _sentiment_edges(self, user_id: str, entity: str) -> List[tuple[str, Dict[str, Any]]]:
        """Return sentiment edges for (user_id, entity)."""

        edges: List[tuple[str, Dict[str, Any]]] = []
        entity_query = self._normalize_entity(entity)
        if user_id not in self.graph:
            return edges

        for _, to_entity, key, data in self.graph.out_edges(user_id, keys=True, data=True):
            if data.get("relation_type") != self.SENTIMENT_RELATION:
                continue
            to_entity_norm = self._normalize_entity(str(to_entity))
            raw = str(data.get("raw_text", data.get("context", ""))).lower()
            # Exact + partial/fuzzy style match
            if entity_query:
                if (
                    to_entity_norm != entity_query
                    and entity_query not in to_entity_norm
                    and to_entity_norm not in entity_query
                    and entity_query not in raw
                ):
                    continue
            edges.append((str(key), data))
        return edges

    def get_sentiment_history(self, user_id: str, entity: str) -> List[Dict[str, Any]]:
        """Return all opinions over time for a given user+entity."""

        try:
            entity_query = self._normalize_entity(entity)
            edges = self._sentiment_edges(user_id, entity_query)
            def _sort_key(item: tuple[str, Dict[str, Any]]) -> datetime:
                t = item[1].get("tcommit")
                if isinstance(t, str):
                    return datetime.fromisoformat(t)
                return datetime.fromtimestamp(0)

            edges_sorted = sorted(edges, key=_sort_key)
            out: List[Dict[str, Any]] = []
            for key, data in edges_sorted:
                out.append(
                    {
                        "relation": self.SENTIMENT_RELATION,
                        "value": str(data.get("value", entity_query)),
                        "sentiment_score": float(data.get("sentiment_score", 0.0)),
                        "intensity_label": str(data.get("intensity_label", "NEUTRAL")),
                        "raw_text": str(data.get("raw_text", data.get("context", ""))),
                        "tcommit": str(data.get("tcommit", "")),
                        "edge_key": key,
                    }
                )
            return out
        except Exception as e:
            console.print("[red]SentimentGraph.get_sentiment_history failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_current_sentiment(self, user_id: str, entity: str) -> Optional[Dict[str, Any]]:
        """Return the latest opinion about an entity by the user."""

        try:
            entity_query = self._normalize_entity(entity)
            history = self.get_sentiment_history(user_id, entity_query)
            if not history:
                return None
            return history[-1]
        except Exception as e:
            console.print("[red]SentimentGraph.get_current_sentiment failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_all_sentiments(self, user_id: str) -> List[Dict[str, Any]]:
        """Return all sentiments for a user sorted by commit time."""

        try:
            out: List[Dict[str, Any]] = []
            if user_id not in self.graph:
                return out
            for _, to_entity, key, data in self.graph.out_edges(user_id, keys=True, data=True):
                if data.get("relation_type") != self.SENTIMENT_RELATION:
                    continue
                out.append(
                    {
                        "relation": self.SENTIMENT_RELATION,
                        "value": str(to_entity),
                        "sentiment_score": float(data.get("sentiment_score", 0.0)),
                        "intensity_label": str(data.get("intensity_label", "NEUTRAL")),
                        "raw_text": str(data.get("raw_text", data.get("context", ""))),
                        "tcommit": str(data.get("tcommit", "")),
                        "edge_key": str(key),
                    }
                )
            return sorted(
                out,
                key=lambda d: datetime.fromisoformat(d["tcommit"]) if d.get("tcommit") else datetime.fromtimestamp(0),
            )
        except Exception as e:
            console.print("[red]SentimentGraph.get_all_sentiments failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_all_strong_sentiments(self, user_id: str) -> List[Dict[str, Any]]:
        """Return all STRONG_* sentiment facts for a user.

        These should NEVER be forgotten (high salience).
        """

        try:
            strong: List[Dict[str, Any]] = []
            if user_id not in self.graph:
                return strong

            for _, to_entity, _, data in self.graph.out_edges(user_id, keys=True, data=True):
                if data.get("relation_type") != self.SENTIMENT_RELATION:
                    continue
                intensity_label = str(data.get("intensity_label", "NEUTRAL"))
                if intensity_label not in self.STRONG_LABELS:
                    continue
                strong.append(
                    {
                        "relation": self.SENTIMENT_RELATION,
                        "value": str(to_entity),
                        "sentiment_score": float(data.get("sentiment_score", 0.0)),
                        "intensity_label": intensity_label,
                        "raw_text": str(data.get("raw_text", data.get("context", ""))),
                        "tcommit": str(data.get("tcommit", "")),
                    }
                )
            # Sort by tcommit
            strong_sorted = sorted(
                strong, key=lambda d: datetime.fromisoformat(d["tcommit"]) if d.get("tcommit") else datetime.fromtimestamp(0)
            )
            return strong_sorted
        except Exception as e:
            console.print("[red]SentimentGraph.get_all_strong_sentiments failed[/red]")
            console.print_exception(show_locals=False)
            raise e

