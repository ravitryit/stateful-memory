from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


@dataclass(frozen=True)
class TierConfig:
    """Threshold configuration for tier assignment."""

    hot_threshold: float = 0.7
    warm_threshold: float = 0.4  # was 0.3 — raised so WARM gets nodes in 0.4-0.7 band


class TieredStorage:
    """Assign and manage HOT/WARM/COLD tiers for nodes."""

    def __init__(self, config: TierConfig | None = None) -> None:
        """Initialize tiered storage."""

        self._config = config or TierConfig()
        self._confidence_by_node: Dict[str, float] = {}
        self._tier_by_node: Dict[str, str] = {}

    def assign_tier(self, node_id: str, confidence_score: float) -> str:
        """Assign a tier based on confidence score and persist it internally."""

        tier: str
        if confidence_score > self._config.hot_threshold:
            tier = "HOT"
        elif confidence_score > self._config.warm_threshold:
            tier = "WARM"
        else:
            tier = "COLD"

        self._confidence_by_node[node_id] = float(confidence_score)
        self._tier_by_node[node_id] = tier
        return tier

    def get_tier(self, node_id: str) -> str:
        """Get the current tier for a node."""

        return self._tier_by_node.get(node_id, "COLD")

    def promote_node(self, node_id: str) -> None:
        """Boost confidence score for a node and update its tier."""

        cur = self._confidence_by_node.get(node_id, 0.0)
        boosted = min(1.0, cur + 0.1)
        self.assign_tier(node_id, boosted)

    def demote_node(self, node_id: str) -> None:
        """Lower confidence score for a node and update its tier."""

        cur = self._confidence_by_node.get(node_id, 1.0)
        demoted = max(0.0, cur - 0.1)
        self.assign_tier(node_id, demoted)

    def get_tier_stats(self) -> Dict[str, int]:
        """Return counts per tier."""

        stats = {"HOT": 0, "WARM": 0, "COLD": 0}
        for _, tier in self._tier_by_node.items():
            if tier in stats:
                stats[tier] += 1
        return stats

