from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


@dataclass(frozen=True)
class ConfidenceParameters:
    """Parameters for the retention/confidence scoring formula."""

    lam: float = 0.1
    sigma: float = 0.5


class ConfidenceScorer:
    """Retention score calculator for memory nodes.

    Uses the requested formula:
        R(m,t) = Isalience * e^(-λ*Δt) + σ * Σ(1/(t-t_access_i))
    """

    def __init__(self, params: Optional[ConfidenceParameters] = None) -> None:
        """Initialize the scorer with optional custom parameters."""

        self._params = params or ConfidenceParameters()

    def classify_importance(self, fact_text: str) -> float:
        """Classify importance based on simple keyword rules.

        Args:
            fact_text: Text from a fact/opinion about an entity.

        Returns:
            Importance value in [0.3, 0.6, 0.9].
        """

        text = (fact_text or "").lower()
        if any(k in text for k in ["allergy", "medical", "name"]):
            return 0.9
        if any(k in text for k in ["prefer", "like", "hate"]):
            return 0.6
        return 0.3

    def calculate_score(
        self,
        node: Dict[str, Any],
        current_time: datetime,
        access_history: List[str],
        importance_text: Optional[str] = None,
    ) -> float:
        """Calculate retention score for a node.

        Time-based short-circuit rules ensure recently created nodes always
        start in HOT tier rather than being immediately decayed to COLD:
          - created < 1 day ago  → 1.0  (HOT)
          - created < 30 days    → 0.8  (HOT)
          - created < 180 days   → 0.5  (WARM)
          - older               → standard exponential decay formula

        Args:
            node: Node attribute dict (must contain `created_at` as ISO string).
            current_time: Time at which to score retention.
            access_history: List of ISO access timestamps.
            importance_text: Optional text used to classify salience.

        Returns:
            Confidence score clamped into [0.0, 1.0].
        """

        try:
            created_at_raw = node.get("created_at")
            if isinstance(created_at_raw, str):
                created_at = datetime.fromisoformat(created_at_raw)
            else:
                # No timestamp → treat as brand new.
                created_at = current_time

            delta_days = max((current_time - created_at).total_seconds() / (60 * 60 * 24), 0.0)

            # --- Time-based short-circuit for new nodes ---
            if delta_days < 1:
                return 1.0   # < 1 day → always HOT
            if delta_days < 30:
                return 0.8   # < 30 days → HOT
            if delta_days < 180:
                return 0.5   # < 180 days → WARM

            # Older nodes: use the full exponential decay + access reinforcement.
            fact_text = importance_text if importance_text is not None else str(node.get("importance_text", ""))
            isalience = float(self.classify_importance(fact_text))

            decay = isalience * math.exp(-self._params.lam * delta_days)

            reinforcement = 0.0
            eps = 1e-6
            for t_access in access_history:
                try:
                    if not isinstance(t_access, str):
                        continue
                    t = datetime.fromisoformat(t_access)
                    dt_days = max((current_time - t).total_seconds() / (60 * 60 * 24), 0.0) + eps
                    reinforcement += 1.0 / dt_days
                except Exception:
                    continue

            score = decay + self._params.sigma * reinforcement
            return max(0.0, min(1.0, float(score)))
        except Exception as e:
            console.print("[red]ConfidenceScorer.calculate_score failed[/red]")
            console.print_exception(show_locals=False)
            raise e



