from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


def _iso_now() -> str:
    """Return the current time as an ISO 8601 datetime string."""

    return datetime.now().isoformat()


@dataclass(frozen=True)
class TemporalCommit:
    """Represents a single Git-style append-only commit in the temporal engine."""

    commit_id: str
    timestamp: str
    session_id: str
    facts: List[str]
    entities: List[str]
    relations: List[Dict[str, Any]]
    parent_commit: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the commit to a serializable dict."""

        return {
            "commit_id": self.commit_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "facts": self.facts,
            "entities": self.entities,
            "relations": self.relations,
            "parent_commit": self.parent_commit,
        }


class TemporalEngine:
    """Git-style commit log for extracted facts, entities and relationships."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        """Initialize the temporal engine.

        Args:
            data_dir: Optional persistence directory. If provided, the commit log
                is saved to and loaded from JSON.
        """

        self._data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent.parent / ".data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._commit_path = self._data_dir / "temporal_commits.json"

        self.commit_log: List[Dict[str, Any]] = []
        self._session_last_commit: Dict[str, Optional[str]] = {}
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Load persisted commit log if available."""

        try:
            if not self._commit_path.exists():
                return
            raw = self._commit_path.read_text(encoding="utf-8").strip()
            if not raw:
                return
            loaded = json.loads(raw)
            if not isinstance(loaded, list):
                return
            self.commit_log = loaded
            # Rebuild last commit pointers per session
            for c in self.commit_log:
                sid = str(c.get("session_id"))
                cid = str(c.get("commit_id"))
                self._session_last_commit[sid] = cid
        except Exception as e:
            console.print("[red]TemporalEngine load failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def _persist(self) -> None:
        """Persist commit log to disk."""

        try:
            self._commit_path.write_text(json.dumps(self.commit_log, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            console.print("[red]TemporalEngine persist failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def create_commit(self, session_id: str, raw_text: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new commit and append it to the commit log.

        The resulting commit dict uses the exact keys specified in the assignment.
        """

        try:
            facts = extracted_data.get("facts", [])
            entities = extracted_data.get("entities", [])
            relations = extracted_data.get("relations", [])

            if not isinstance(facts, list) or not isinstance(entities, list) or not isinstance(relations, list):
                raise TypeError("extracted_data must contain facts/entities/relations lists")

            commit_id = uuid.uuid4().hex
            timestamp = _iso_now()
            parent_commit = self._session_last_commit.get(session_id)

            commit = TemporalCommit(
                commit_id=commit_id,
                timestamp=timestamp,
                session_id=session_id,
                facts=[str(x) for x in facts],
                entities=[str(x) for x in entities],
                relations=[x for x in relations if isinstance(x, dict)],
                parent_commit=parent_commit,
            ).to_dict()

            self.commit_log.append(commit)
            self._session_last_commit[session_id] = commit_id
            self._persist()
            return commit
        except Exception as e:
            console.print("[red]TemporalEngine.create_commit failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_commit_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all commits for a session, in chronological order."""

        try:
            commits = [c for c in self.commit_log if str(c.get("session_id")) == session_id]
            return sorted(commits, key=lambda c: str(c.get("timestamp")))
        except Exception as e:
            console.print("[red]TemporalEngine.get_commit_history failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def checkout_commit(self, commit_id: str) -> Dict[str, Any]:
        """Return the aggregated state at a specific commit (git checkout style)."""

        try:
            lookup = {str(c.get("commit_id")): c for c in self.commit_log}
            if commit_id not in lookup:
                raise KeyError(f"Unknown commit_id: {commit_id}")

            current = lookup[commit_id]
            aggregated_facts: List[str] = []
            aggregated_entities: List[str] = []
            aggregated_relations: List[Dict[str, Any]] = []

            # Walk backwards to root, collecting then reversing for chronological order.
            chain: List[Dict[str, Any]] = []
            cur = current
            while cur is not None:
                chain.append(cur)
                parent_id = cur.get("parent_commit")
                if parent_id is None:
                    break
                parent_id_str = str(parent_id)
                cur = lookup.get(parent_id_str)

            for c in reversed(chain):
                aggregated_facts.extend([str(x) for x in c.get("facts", [])])
                aggregated_entities.extend([str(x) for x in c.get("entities", [])])
                aggregated_relations.extend([x for x in c.get("relations", []) if isinstance(x, dict)])

            # De-duplicate while preserving order.
            def _dedupe(items: List[Any]) -> List[Any]:
                seen = set()
                out: List[Any] = []
                for it in items:
                    key = json.dumps(it, sort_keys=True, ensure_ascii=False) if isinstance(it, dict) else str(it)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(it)
                return out

            return {
                "commit_id": str(current.get("commit_id")),
                "timestamp": str(current.get("timestamp")),
                "facts": _dedupe(aggregated_facts),
                "entities": _dedupe(aggregated_entities),
                "relations": _dedupe(aggregated_relations),
            }
        except Exception as e:
            console.print("[red]TemporalEngine.checkout_commit failed[/red]")
            console.print_exception(show_locals=False)
            raise e

