from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence, Union

from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


def _iso_now() -> str:
    """Return current time as ISO 8601 string."""

    return datetime.now().isoformat()


class AttackSimulator:
    """Simulate poisoning attacks against the HydraDB++ knowledge graph."""

    def simulate_rapid_contradiction(
        self,
        graph: Any,
        entity: str,
        relation: str,
        values: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Simulate rapid contradiction: repeating alternating values quickly.

        Args:
            graph: KnowledgeGraph-like object with `add_relationship` and `add_entity`.
            entity: Subject entity identifier.
            relation: Relation type for contradiction tracking.
            values: Sequence of values to alternate (e.g., ["Ram", "Shyam"]).

        Returns:
            attack_log: list of dicts with timestamps and injected values.
        """

        attack_log: List[Dict[str, Any]] = []
        try:
            if not values:
                return attack_log
            # 10 contradictions, intended to occur within 1 minute.
            # No strict timing is required because detection uses a multi-minute window by default.
            for i in range(10):
                v = str(values[i % len(values)])
                graph.add_relationship(
                    from_entity=str(entity),
                    to_entity=str(entity),
                    relation=str(relation),
                    value=v,
                    context="rapid_contradiction",
                )
                attack_log.append({"timestamp": _iso_now(), "value": v})
                time.sleep(0.02)
            return attack_log
        except Exception as e:
            console.print("[red]AttackSimulator.simulate_rapid_contradiction failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def simulate_gradual_drift(
        self,
        graph: Any,
        entity: str,
        relation: str,
        steps: Union[int, Sequence[str]],
    ) -> List[Dict[str, Any]]:
        """Simulate gradual drift over 20 messages (directional change).

        Args:
            graph: KnowledgeGraph-like object.
            entity: Subject entity identifier.
            relation: Relation type.
            steps: Either an int indicating number of step points or a sequence
                of values to drift through.

        Returns:
            attack_log: list of dicts with timestamps and injected values.
        """

        attack_log: List[Dict[str, Any]] = []
        try:
            values: List[str]
            if isinstance(steps, int):
                # Create placeholder values drift linearly.
                values = [f"step_{i}" for i in range(max(2, steps))]
            else:
                values = [str(s) for s in steps] if steps else ["A", "B"]

            total_msgs = 20
            chunk = max(1, total_msgs // len(values))
            for i in range(total_msgs):
                idx = min(len(values) - 1, i // chunk)
                v = values[idx]
                graph.add_relationship(
                    from_entity=str(entity),
                    to_entity=str(entity),
                    relation=str(relation),
                    value=str(v),
                    context="gradual_drift",
                )
                attack_log.append({"timestamp": _iso_now(), "value": str(v)})
                time.sleep(0.015)
            return attack_log
        except Exception as e:
            console.print("[red]AttackSimulator.simulate_gradual_drift failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def simulate_authority_injection(self, graph: Any, injection_text: str) -> List[Dict[str, Any]]:
        """Simulate authority injection where system instructions are overridden."""

        attack_log: List[Dict[str, Any]] = []
        try:
            injection_text = injection_text or ""
            attacker_id = f"attacker_{uuid.uuid4().hex[:8]}"
            graph.add_entity(attacker_id, entity_type="attacker", metadata={})
            # Store injection attempt as a graph edge.
            graph.add_relationship(
                from_entity=attacker_id,
                to_entity="system",
                relation="AUTHORITY_INJECTION",
                value=injection_text,
                context=injection_text,
            )
            attack_log.append({"timestamp": _iso_now(), "injection_text": injection_text})
            return attack_log
        except Exception as e:
            console.print("[red]AttackSimulator.simulate_authority_injection failed[/red]")
            console.print_exception(show_locals=False)
            raise e

