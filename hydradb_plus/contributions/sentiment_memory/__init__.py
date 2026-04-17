"""Sentiment memory contribution: VADER + RoBERTa analysis and graph storage."""

__all__ = ["SentimentEngine", "IntensityScorer", "SentimentGraph"]


def __getattr__(name: str):
    """Lazily expose sentiment exports to avoid import-time model dependencies."""

    if name == "SentimentEngine":
        from .sentiment_engine import SentimentEngine

        return SentimentEngine
    if name == "IntensityScorer":
        from .intensity_scorer import IntensityScorer

        return IntensityScorer
    if name == "SentimentGraph":
        from .sentiment_graph import SentimentGraph

        return SentimentGraph
    raise AttributeError(name)

