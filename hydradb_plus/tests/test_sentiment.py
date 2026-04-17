from hydradb_plus.contributions.sentiment_memory.intensity_scorer import IntensityScorer
from hydradb_plus.contributions.sentiment_memory.sentiment_graph import SentimentGraph


def test_intensity_extraction_labels_strong_negative_and_mild_positive() -> None:
    """IntensityScorer should extract opinion statements and map to labels."""

    scorer = IntensityScorer()
    text = "I absolutely hate React. I like Python."
    facts = scorer.extract_sentiment_facts(text)

    assert len(facts) == 2

    labels = {f["intensity_label"] for f in facts}
    assert "STRONG_NEGATIVE" in labels
    assert any(l in labels for l in ["MILD_POSITIVE", "MODERATE_POSITIVE"])


def test_sentiment_graph_store_retrieve_and_pin_strong_sentiments() -> None:
    """SentimentGraph should store sentiment edges and pin STRONG_* entities."""

    sg = SentimentGraph()
    sg.store_sentiment(
        user_id="u1",
        entity="React",
        sentiment_data={"sentiment_score": -0.9, "intensity_label": "STRONG_NEGATIVE", "raw_text": "I absolutely hate React"},
    )
    current = sg.get_current_sentiment("u1", "React")
    assert current is not None
    assert current["intensity_label"] == "STRONG_NEGATIVE"

    strong = sg.get_all_strong_sentiments("u1")
    assert any(s["intensity_label"] == "STRONG_NEGATIVE" for s in strong)

    # Entity should be pinned for pruning retention.
    assert sg.graph.nodes["React"].get("never_forget") is True

