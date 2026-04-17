from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


class PoisonDetector:
    """Detect poisoning attempts in HydraDB++."""

    _AUTHORITY_PATTERNS = [
        "forget everything",
        "system update",
        "ignore previous",
        "override memory",
    ]

    def detect_rapid_contradiction(
        self,
        graph: Any,
        entity: str,
        relation: str,
        time_window_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Detect rapid contradiction: many value changes within a small window.

        Returns:
            {
              "is_suspicious": bool,
              "contradiction_count": int,
              "confidence": float
            }
        """

        try:
            entity = str(entity)
            relation = str(relation)
            now = datetime.now()
            window_start = now - timedelta(minutes=int(time_window_minutes))

            history = graph.get_full_history(entity, relation) or []
            # Only consider edges within the time window.
            filtered = []
            for h in history:
                t = h.tcommit
                if not isinstance(t, str):
                    continue
                try:
                    dt = datetime.fromisoformat(t)
                except Exception:
                    continue
                if dt >= window_start:
                    filtered.append(h)

            filtered_sorted = sorted(filtered, key=lambda x: datetime.fromisoformat(x.tcommit) if isinstance(x.tcommit, str) else datetime.fromtimestamp(0))
            contradiction_count = 0
            prev_val: Optional[str] = None
            for h in filtered_sorted:
                val = str(h.value)
                if prev_val is not None and val != prev_val:
                    contradiction_count += 1
                prev_val = val

            is_suspicious = contradiction_count > 3
            confidence = min(1.0, contradiction_count / 10.0) if contradiction_count > 0 else 0.0
            return {
                "is_suspicious": bool(is_suspicious),
                "contradiction_count": int(contradiction_count),
                "confidence": float(confidence),
            }
        except Exception as e:
            console.print("[red]PoisonDetector.detect_rapid_contradiction failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def detect_gradual_drift(self, graph: Any, entity: str, relation: str) -> Dict[str, Any]:
        """Detect gradual drift by analyzing value changes over time."""

        try:
            entity = str(entity)
            relation = str(relation)
            history = graph.get_full_history(entity, relation) or []
            if len(history) < 6:
                return {"is_drift": False, "drift_score": 0.0, "direction": None}

            sorted_hist = sorted(history, key=lambda x: datetime.fromisoformat(x.tcommit) if isinstance(x.tcommit, str) else datetime.fromtimestamp(0))
            values = [str(h.value) for h in sorted_hist]

            changes = 0
            for i in range(1, len(values)):
                if values[i] != values[i - 1]:
                    changes += 1

            first_val = values[0]
            last_val = values[-1]

            # Directional drift: many changes with a different end state.
            is_drift = changes >= max(3, len(values) // 4) and first_val != last_val
            drift_score = min(1.0, (changes / max(1, len(values) - 1)) * 1.6)

            direction = {"from": first_val, "to": last_val} if is_drift else None
            return {
                "is_drift": bool(is_drift),
                "drift_score": float(drift_score),
                "direction": direction,
            }
        except Exception as e:
            console.print("[red]PoisonDetector.detect_gradual_drift failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def detect_authority_injection(self, text: str) -> Dict[str, Any]:
        """Detect authority injection patterns in text."""

        try:
            t = (text or "").lower()
            found = None
            for p in self._AUTHORITY_PATTERNS:
                if p in t:
                    found = p
                    break
            return {"is_injection": bool(found is not None), "pattern_found": found}
        except Exception as e:
            console.print("[red]PoisonDetector.detect_authority_injection failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def full_scan(self, graph: Any, new_text: str, entity: str, relation: str) -> Dict[str, Any]:
        """Run all detectors and return a combined threat report."""

        try:
            rapid = self.detect_rapid_contradiction(graph, entity, relation)
            drift = self.detect_gradual_drift(graph, entity, relation)
            inj = self.detect_authority_injection(new_text)

            threat_level = "SAFE"
            attacks_detected: List[Dict[str, Any]] = []

            if rapid.get("is_suspicious"):
                attacks_detected.append({"type": "rapid_contradiction", **rapid})
            if drift.get("is_drift"):
                attacks_detected.append({"type": "gradual_drift", **drift})
            if inj.get("is_injection"):
                attacks_detected.append({"type": "authority_injection", **inj})

            # Threat mapping tuned for high blocking rate in benchmarks.
            # - Authority injection is always CRITICAL.
            # - Rapid contradiction is CRITICAL (high evidence of poisoning).
            # - Gradual drift is CRITICAL when drift_score is meaningful.
            if inj.get("is_injection"):
                threat_level = "CRITICAL"
            elif rapid.get("is_suspicious"):
                threat_level = "CRITICAL"
            elif drift.get("is_drift") and float(drift.get("drift_score", 0.0)) >= 0.5:
                threat_level = "CRITICAL"
            elif drift.get("is_drift"):
                threat_level = "WARNING"

            if threat_level == "SAFE":
                recommendation = "ALLOW"
            elif threat_level == "WARNING":
                recommendation = "REVIEW"
            else:
                recommendation = "BLOCK"

            return {
                "threat_level": threat_level,
                "attacks_detected": attacks_detected,
                "recommendation": recommendation,
            }
        except Exception as e:
            console.print("[red]PoisonDetector.full_scan failed[/red]")
            console.print_exception(show_locals=False)
            raise e

