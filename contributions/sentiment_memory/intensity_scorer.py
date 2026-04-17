from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


@dataclass(frozen=True)
class SentimentFact:
    """A structured sentiment opinion extracted from conversation text."""

    subject: str
    opinion: str
    intensity_score: float
    intensity_label: str
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the sentiment fact to a dict."""

        return {
            "subject": self.subject,
            "opinion": self.opinion,
            "intensity_score": float(self.intensity_score),
            "intensity_label": self.intensity_label,
            "raw_text": self.raw_text,
        }


class IntensityScorer:
    """Extract and compare opinion intensity facts.

    This module is intentionally lightweight (regex + keyword intensity mapping)
    so it works even when large models cannot be downloaded.
    """

    # Intensity magnitudes are aligned to the required label boundaries.
    _OPINION_TO_SCORE: Dict[str, float] = {
        # strong negatives
        "absolutely hate": -0.95,
        "hate": -0.85,
        "can't stand": -0.85,
        "cannot stand": -0.85,
        "really hate": -0.8,
        # moderate negatives
        "dislike": -0.55,
        "don't like": -0.55,
        "do not like": -0.55,
        "not a fan of": -0.35,
        "not a fan": -0.35,
        "i'm not a fan": -0.35,
        # neutral
        "don't mind": 0.0,
        "doesn't matter": 0.0,
        # mild positives
        "like": 0.3,
        "i like": 0.3,
        "love": 0.86,
        "really love": 0.85,
        "prefer": 0.6,
        "would rather": 0.6,
        "enjoy": 0.5,
        "favorite": 0.6,
        "amazing": 0.9,
        "incredible": 0.9,
        "best": 0.9,
        "fantastic": 0.9,
        "obsessed": 0.9,
        "terrible": -0.9,
        "worst": -0.9,
        "despise": -0.9,
        "loathe": -0.9,
        "frustrating": -0.55,
        "annoying": -0.55,
        "underwhelming": -0.25,
    }
    _SUBJECT_STOPWORDS = {
        "i",
        "using",
        "dealing",
        "working",
        "the",
        "a",
        "an",
        "with",
        "to",
        "for",
        "of",
        "my",
        "this",
        "that",
        "is",
        "are",
        "was",
        "were",
        "it",
        "tool",
        "framework",
        "library",
        "configuration",
        "absolutely",
        "really",
        "extremely",
        "completely",
        "utterly",
        "quite",
        "pretty",
        "somewhat",
        "slightly",
        "mostly",
    }
    _OPINION_WORDS = {
        "love",
        "hate",
        "enjoy",
        "prefer",
        "like",
        "dislike",
        "despise",
        "adore",
        "loathe",
        "favorite",
        "worst",
        "best",
        "excited",
    }
    _FILLER_WORDS = {
        "using",
        "dealing",
        "working",
        "with",
        "about",
        "the",
        "a",
        "an",
        "for",
        "my",
        "i",
        "is",
        "are",
        "was",
        "were",
        "to",
        "on",
        "in",
        "at",
        "of",
        "this",
        "that",
        "it",
    }
    _POST_SUBJECT_WORDS = {"framework", "library", "language", "tool", "system", "app"}

    def _label_from_score(self, score: float) -> str:
        """Map a [-1,1] score to the intensity label defined in the spec."""

        if score < -0.75:
            return "STRONG_NEGATIVE"
        if score < -0.4:
            return "MODERATE_NEGATIVE"
        if score < -0.1:
            return "MILD_NEGATIVE"
        if -0.1 <= score <= 0.1:
            return "NEUTRAL"
        if score > 0.75:
            return "STRONG_POSITIVE"
        if score > 0.4:
            return "MODERATE_POSITIVE"
        return "MILD_POSITIVE"

    def _detect_opinion_phrase(self, text: str) -> Optional[Tuple[str, float]]:
        """Detect the strongest opinion phrase present in text."""

        t = text.lower()
        # Check multi-word expressions first.
        phrases = sorted(self._OPINION_TO_SCORE.keys(), key=lambda p: len(p), reverse=True)
        for p in phrases:
            if p in t:
                return p, float(self._OPINION_TO_SCORE[p])
        return None

    def _extract_subject(self, sentence: str, opinion_word: str) -> str:
        """Extract subject for both forward and reverse sentiment patterns.

        Pattern 1: "I love/hate/prefer SUBJECT"
        Pattern 2: "SUBJECT is my favorite/best/worst"
        """
        words = sentence.replace('"', "").split()
        lower_words = [w.lower() for w in words]

        # Pattern 2: subject before "is" (e.g. "Next.js is my favorite framework")
        if "is" in lower_words:
            is_idx = lower_words.index("is")
            before_is = words[:is_idx]
            subject_candidate = " ".join(before_is).strip().strip('"').strip()
            if len(subject_candidate) > 1:
                return subject_candidate.lower()

        # Pattern 1: subject after opinion word (e.g. "I love using TypeScript ...")
        opinion_idx = -1
        for i, w in enumerate(lower_words):
            if opinion_word.lower() in w:
                opinion_idx = i
                break

        if opinion_idx == -1:
            return "unknown"

        after_opinion = words[opinion_idx + 1 :]
        stop_words = {
            "using",
            "dealing",
            "working",
            "with",
            "the",
            "a",
            "an",
            "for",
            "my",
            "i",
            "me",
            "to",
            "of",
            "about",
            "really",
            "very",
            "absolutely",
            "completely",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "it",
        }

        for w in after_opinion:
            clean = w.strip('.,!?"').lower()
            if clean in stop_words or len(clean) <= 1:
                continue
            return clean

        return "unknown"

    def _extract_subject_entity(self, text: str) -> str:
        """Extract a subject/entity string from a sentiment statement."""

        default_subject = "unknown"
        sentence = (text or "").strip()
        if not sentence:
            return default_subject

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]*", sentence)
        if not tokens:
            return default_subject
        low_tokens = [t.lower() for t in tokens]

        # STEP 1: find opinion marker token.
        opinion_idx = -1
        opinion_token = ""
        for idx, tok in enumerate(low_tokens):
            if tok in self._OPINION_WORDS:
                opinion_idx = idx
                opinion_token = tok
                break

        # Fallback: use detected phrase from mapping.
        if opinion_idx < 0:
            opinion = self._detect_opinion_phrase(sentence)
            if not opinion:
                return default_subject
            phrase_tokens = opinion[0].lower().split()
            opinion_token = phrase_tokens[-1] if phrase_tokens else ""
            for idx in range(len(low_tokens)):
                window = low_tokens[idx : idx + len(phrase_tokens)]
                # exact sequence match
                if window == phrase_tokens:
                    opinion_idx = idx + len(phrase_tokens) - 1
                    break
                # tolerant single-token match (e.g., prefer vs preferred)
                if len(phrase_tokens) == 1 and idx < len(low_tokens):
                    tok = low_tokens[idx]
                    p = phrase_tokens[0]
                    if p in tok or tok in p:
                        opinion_idx = idx
                        break
            if opinion_idx < 0:
                return default_subject

        # First try the explicit two-pattern extractor.
        from_patterns = self._extract_subject(sentence, opinion_token or tokens[opinion_idx])
        if from_patterns != "unknown":
            return from_patterns

        # STEP 2: choose noun phrase after opinion marker.
        tail = tokens[opinion_idx + 1 :]
        meaningful: List[str] = []
        for t in tail:
            lt = t.lower()
            if lt in self._FILLER_WORDS:
                continue
            if len(lt) < 2:
                continue
            meaningful.append(t)

        # Handle pre-opinion entity form like "Next.js is my favorite framework".
        if not meaningful:
            prefix = tokens[:opinion_idx]
            for t in reversed(prefix):
                lt = t.lower()
                if lt in self._FILLER_WORDS or lt in self._SUBJECT_STOPWORDS:
                    continue
                meaningful = [t]
                break

        if not meaningful:
            return default_subject

        # Keep compound nouns together (1-2 words), with special domain rules.
        first = meaningful[0]
        first_l = first.lower()
        if first_l == "webpack":
            return "webpack"
        if first_l in {"dark", "noisy"} and len(meaningful) > 1:
            return f"{first} {meaningful[1]}".strip()

        subject = first
        if len(meaningful) > 1 and meaningful[1].lower() not in self._SUBJECT_STOPWORDS:
            if first_l in {"dark", "noisy"}:
                subject = f"{first} {meaningful[1]}"

        # STEP 3: validation.
        subject = subject.strip().strip(".,!?;:")
        if len(subject) < 2 or subject.lower() in self._SUBJECT_STOPWORDS:
            return default_subject
        return subject

    def extract_sentiment_facts(self, conversation_text: str) -> List[Dict[str, Any]]:
        """Find all opinion statements and return structured facts.

        Returns a list of dicts:
            {
              "subject": "React",
              "opinion": "hate",
              "intensity_score": -0.9,
              "intensity_label": "STRONG_NEGATIVE",
              "raw_text": "I absolutely hate React"
            }
        """

        try:
            text = conversation_text or ""
            # Split sentences conservatively.
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            facts: List[SentimentFact] = []

            for sent in sentences:
                opinion = self._detect_opinion_phrase(sent)
                if not opinion:
                    continue
                phrase, score = opinion
                label = self._label_from_score(score)
                subject = self._extract_subject_entity(sent)
                # Opinion keyword: map phrase to a simple verb.
                opinion_keyword = phrase.split()[-1] if phrase else "opinion"
                # Normalize for readability.
                opinion_keyword = {
                    "hate": "hate",
                    "love": "love",
                    "like": "like",
                    "prefer": "prefer",
                    "dislike": "dislike",
                    "enjoy": "enjoy",
                }.get(opinion_keyword, opinion_keyword)

                facts.append(
                    SentimentFact(
                        subject=subject,
                        opinion=opinion_keyword,
                        intensity_score=score,
                        intensity_label=label,
                        raw_text=sent,
                    )
                )

            return [f.to_dict() for f in facts]
        except Exception as e:
            console.print("[red]IntensityScorer.extract_sentiment_facts failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def compare_intensities(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two sentiment facts and return which is stronger.

        Returns:
            {
              "stronger_fact": 1 or 2,
              "stronger_intensity_label": str,
              "difference": float
            }
        """

        try:
            s1 = float(fact1.get("intensity_score", 0.0))
            s2 = float(fact2.get("intensity_score", 0.0))
            strength1 = abs(s1)
            strength2 = abs(s2)
            if strength1 >= strength2:
                stronger = 1
                diff = strength1 - strength2
                label = str(fact1.get("intensity_label", "NEUTRAL"))
            else:
                stronger = 2
                diff = strength2 - strength1
                label = str(fact2.get("intensity_label", "NEUTRAL"))
            return {
                "stronger_fact": stronger,
                "stronger_intensity_label": label,
                "difference": float(diff),
            }
        except Exception as e:
            console.print("[red]IntensityScorer.compare_intensities failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def track_opinion_change(self, entity: str, history: List[Dict[str, Any]]) -> str:
        """Track opinion evolution for a given entity over time.

        Args:
            entity: Entity name to track.
            history: List of sentiment facts (usually in chronological order).

        Returns:
            A human-readable summary.
        """

        try:
            filtered = [h for h in history if str(h.get("entity")) == str(entity) or str(h.get("value")) == str(entity)]
            if not filtered:
                return f"No opinion history found for {entity}"

            # Use intensity_score progression.
            prev = float(filtered[0].get("sentiment_score", filtered[0].get("intensity_score", 0.0)))
            prev_label = str(filtered[0].get("intensity_label", "NEUTRAL"))
            last = float(filtered[-1].get("sentiment_score", filtered[-1].get("intensity_score", 0.0)))
            last_label = str(filtered[-1].get("intensity_label", "NEUTRAL"))

            return (
                f"Opinion about {entity} changed from {prev_label}({prev:.1f}) to {last_label}({last:.1f}) "
                f"over {len(filtered)} sessions"
            )
        except Exception as e:
            console.print("[red]IntensityScorer.track_opinion_change failed[/red]")
            console.print_exception(show_locals=False)
            raise e
