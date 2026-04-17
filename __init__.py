"""HydraDB++: persistent memory layer for LLM agents."""

__all__ = ["HydraDBPlusPlus"]


def __getattr__(name: str):
    """Lazily expose top-level exports to avoid heavy optional imports at import time."""

    if name == "HydraDBPlusPlus":
        from .pipeline.unified_pipeline import HydraDBPlusPlus

        return HydraDBPlusPlus
    raise AttributeError(name)
