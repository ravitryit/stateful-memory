from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


@dataclass(frozen=True)
class SentimentAnalysis:
    """Internal representation of a combined sentiment analysis result."""

    vader_score: float
    roberta_score: float
    combined_score: float
    intensity: str
    emotion_label: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to the dict format expected by the assignment."""

        return {
            "vader_score": float(self.vader_score),
            "roberta_score": float(self.roberta_score),
            "combined_score": float(self.combined_score),
            "intensity": self.intensity,
            "emotion_label": self.emotion_label,
        }


class SentimentEngine:
    """Analyze text sentiment using VADER + RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)."""

    def __init__(
        self,
        roberta_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        enable_roberta: bool = True,
    ) -> None:
        """Initialize the sentiment engine.

        Args:
            roberta_model_name: Hugging Face model for deep sentiment.
            enable_roberta: If False, skips RoBERTa and falls back to VADER.
        """

        self._roberta_model_name = roberta_model_name
        self._enable_roberta = enable_roberta

        self._vader_analyzer = None
        self._roberta_pipeline = None
        self.roberta_available = False

        self._init_vader()
        self._init_roberta()

    def _init_vader(self) -> None:
        """Initialize VADER sentiment analyzer."""

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            console.print("[red]Failed to initialize VADER; sentiment will be neutral[/red]")
            console.print_exception(show_locals=False)
            self._vader_analyzer = None
            raise e

    def _init_roberta(self) -> None:
        """Initialize RoBERTa sentiment pipeline (if enabled)."""

        if not self._enable_roberta:
            self._roberta_pipeline = None
            self.roberta_available = False
            return

        try:
            from transformers import pipeline

            self._roberta_pipeline = pipeline(
                "sentiment-analysis",
                model=self._roberta_model_name,
                return_all_scores=True,
            )
            self.roberta_available = True
        except Exception as e:
            _ = e
            self.roberta_available = False
            # Silent fallback: keep CLI output clean when transformer stack is unavailable.
            self._roberta_pipeline = None

    def _intensity_from_score(self, score: float) -> str:
        """Map a [-1,1] combined score to an intensity label.

        Thresholds tuned to standard VADER compound score distribution.
        """

        if score < -0.6:
            return "STRONG_NEGATIVE"
        if score < -0.3:
            return "MODERATE_NEGATIVE"
        if score < -0.05:
            return "MILD_NEGATIVE"
        if -0.05 <= score <= 0.05:
            return "NEUTRAL"
        if score > 0.6:
            return "STRONG_POSITIVE"
        if score > 0.3:
            return "MODERATE_POSITIVE"
        return "MILD_POSITIVE"

    def _emotion_from_intensity(self, intensity_label: str) -> str:
        """Map intensity label to a coarse emotion label."""

        if "POSITIVE" in intensity_label:
            return "POSITIVE"
        if "NEGATIVE" in intensity_label:
            return "NEGATIVE"
        return "NEUTRAL"

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using VADER + optional RoBERTa.

        When RoBERTa is unavailable or fails, falls back to VADER-only
        (weight=1.0) so the score is still meaningful.

        Combined score (when both available):
            final_score = 0.4 * vader_score + 0.6 * roberta_score
        VADER-only:
            final_score = vader_score
        """

        try:
            text = text or ""
            if self._vader_analyzer is None:
                vader_score = 0.0
            else:
                vader_score = float(self._vader_analyzer.polarity_scores(text).get("compound", 0.0))

            roberta_score: Optional[float] = None
            if self._roberta_pipeline is not None:
                try:
                    pred = self._roberta_pipeline(text[:512])[0]
                    # return_all_scores=True -> list[{"label","score"}]
                    if isinstance(pred, list):
                        label_scores = {str(item.get("label", "")).lower(): float(item.get("score", 0.0)) for item in pred}
                        pos = label_scores.get("positive", label_scores.get("label_2", 0.0))
                        neg = label_scores.get("negative", label_scores.get("label_0", 0.0))
                        roberta_score = float(pos - neg)
                    else:
                        label = str(pred.get("label", "")).lower()
                        prob = float(pred.get("score", 0.0))
                        if "positive" in label:
                            roberta_score = prob
                        elif "negative" in label:
                            roberta_score = -prob
                        else:
                            roberta_score = 0.0
                except Exception:
                    # RoBERTa inference failed; fall back to VADER-only.
                    roberta_score = None

            if roberta_score is not None:
                combined_score = 0.4 * vader_score + 0.6 * roberta_score
            else:
                # VADER-only mode: use full VADER weight.
                combined_score = vader_score

            intensity = self._intensity_from_score(combined_score)
            emotion_label = self._emotion_from_intensity(intensity)

            return SentimentAnalysis(
                vader_score=vader_score,
                roberta_score=roberta_score if roberta_score is not None else vader_score,
                combined_score=combined_score,
                intensity=intensity,
                emotion_label=emotion_label,
            ).to_dict()
        except Exception as e:
            console.print("[red]SentimentEngine.analyze failed[/red]")
            console.print_exception(show_locals=False)
            raise e
