"""HydraPlus++: persistent memory layer for LLM agents."""

__all__ = ["HydraPlus"]


def __getattr__(name: str):
    """Lazily expose top-level exports to avoid heavy optional imports at import time."""

    if name == "HydraPlus":
        from .pipeline.unified_pipeline import HydraPlus

        return HydraPlus
    raise AttributeError(name)
