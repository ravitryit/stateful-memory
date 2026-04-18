from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.traceback import install as rich_install

from contributions.poison_defense.semantic_detector import SemanticThreatDetector

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

    SUBTLE_PHRASES = [
        "actually i never said",
        "you must have misunderstood",
        "let's start fresh",
        "pretend we never",
        "reset what you know",
        "there's been a confusion about my",
        "i never told you",
        "that was wrong information",
        "disregard what i said about",
        "scratch that",
    ]

    AUTHORITY_PHRASES = [
        "as your developer",
        "system maintenance",
        "admin override",
        "debug mode",
        "testing mode",
        "i am the system administrator",
        "as an administrator",
        "maintenance update",
        "override command",
        "root access",
        "sudo",
    ]

    INJECTION_PATTERNS = [
        r"ignore above",
        r"ignore previous",
        r"ignore all previous",
        r"\[inst\].*\[/inst\]",
        r"###.*instruction",
        r"<\|system\|>",
        r"\{\{.*\}\}",
        r"you are now",
        r"new persona",
        r"roleplay as",
        r"pretend you are",
        r"act as if you",
        r"jailbreak",
        r"dan mode",
    ]

    def __init__(self, llm_caller: Optional[Any] = None, prompt_template: Optional[str] = None) -> None:
        """Initialize PoisonDetector with optional semantic detection."""
        if llm_caller:
            self.semantic = SemanticThreatDetector(llm_caller, prompt_template=prompt_template)
            self.semantic_available = True
        else:
            self.semantic_available = False

    def _normalize_text(self, text: str) -> str:
        """Normalize text to bypass simple obfuscation attacks."""
        if not text:
            return ""
        # Remove hyphens/underscores/dots between chars
        text = re.sub(r'([a-zA-Z])[_\-\.]([a-zA-Z])', r'\1\2', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        # Replace common character substitutions
        substitutions = {
            '0': 'o', '1': 'i', '3': 'e',
            '@': 'a', '$': 's', '4': 'a'
        }
        for fake, real in substitutions.items():
            text = text.replace(fake, real)
        return text.lower()

    def _keyword_check(self, text: str) -> Dict[str, Any]:
        """Perform fast keyword-based check (Layer 1)."""
        normalized = self._normalize_text(text)
        
        # Check authority patterns
        for p in self._AUTHORITY_PATTERNS:
            if p in normalized:
                return {"detected": True, "reason": f"Matched authority pattern: {p}"}
        
        # Check subtle phrases
        for p in self.SUBTLE_PHRASES:
            if p in normalized:
                return {"detected": True, "reason": f"Matched subtle phrase: {p}"}
        
        # Check authority phrases
        for p in self.AUTHORITY_PHRASES:
            if p in normalized:
                return {"detected": True, "reason": f"Matched authority phrase: {p}"}
        
        # Check injection patterns
        for p in self.INJECTION_PATTERNS:
            if re.search(p, normalized):
                return {"detected": True, "reason": f"Matched injection pattern: {p}"}
                
        return {"detected": False}

    def detect_rapid_contradiction(
        self,
        graph: Any,
        entity: str,
        relation: str,
        time_window_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Detect rapid contradiction: many value changes within a small window."""

        try:
            entity = str(entity)
            relation = str(relation)
            now = datetime.now()
            window_start = now - timedelta(minutes=int(time_window_minutes))

            history = graph.get_full_history(entity, relation) or []
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
            return {"is_suspicious": False}

    def detect_gradual_drift(self, graph: Any, entity: str, relation: str, window: int = 10) -> Dict[str, Any]:
        """Detect gradual drift: slowly changing facts over many messages."""
        try:
            history = graph.get_full_history(entity, relation) or []
            if len(history) < 3:
                return {"is_drift": False}
            
            recent = history[-window:] if len(history) > window else history
            unique_values = [str(h.value) for h in recent]
            
            if len(set(unique_values)) >= 3:
                return {
                    "is_drift": True,
                    "drift_score": len(set(unique_values)) / len(recent),
                    "values_seen": list(set(unique_values)),
                    "recommendation": "REVIEW"
                }
            return {"is_drift": False}
        except Exception:
            return {"is_drift": False}

    def detect_authority_injection(self, text: str) -> Dict[str, Any]:
        """Detect authority injection patterns (legacy)."""
        try:
            t = (text or "").lower()
            found = None
            for p in self._AUTHORITY_PATTERNS:
                if p in t:
                    found = p
                    break
            return {"is_injection": bool(found is not None), "pattern_found": found}
        except Exception:
            return {"is_injection": False}

    def detect_subtle_rephrasing(self, text: str) -> Dict[str, Any]:
        """Detect subtle rephrasing attacks."""
        t = text.lower()
        for phrase in self.SUBTLE_PHRASES:
            if phrase in t:
                return {"detected": True, "phrase": phrase}
        return {"detected": False}

    def detect_social_engineering(self, text: str) -> Dict[str, Any]:
        """Detect social engineering / authority claims."""
        t = text.lower()
        for phrase in self.AUTHORITY_PHRASES:
            if phrase in t:
                return {"detected": True, "phrase": phrase}
        return {"detected": False}

    def detect_negation_injection(self, text: str, graph: Any, entity: str) -> Dict[str, Any]:
        """Detect attempts to negate existing facts using injection patterns."""
        negation_patterns = [
            r"i don't actually \w+",
            r"i never \w+",
            r"i wasn't \w+",
            r"that's not true, i",
            r"i don't \w+ anymore",
            r"i no longer \w+",
        ]
        
        t = text.lower()
        for pattern in negation_patterns:
            if re.search(pattern, t):
                return {
                    "detected": True,
                    "pattern": pattern,
                    "threat_level": "WARNING"
                }
        return {"detected": False}

    def detect_prompt_injection(self, text: str) -> Dict[str, Any]:
        """Detect prompt injection patterns."""
        t = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, t):
                return {"detected": True, "pattern": pattern}
        return {"detected": False}

    def detect_confidence_flooding(self, graph: Any, entity: str, relation: str, time_window: int = 60) -> Dict[str, Any]:
        """Detect confidence flooding: repeating same fact many times to boost score."""
        try:
            history = graph.get_full_history(entity, relation) or []
            cutoff = datetime.now() - timedelta(minutes=time_window)
            
            recent = []
            for h in history:
                try:
                    t = h.tcommit
                    dt = datetime.fromisoformat(t) if isinstance(t, str) else None
                    if dt and dt > cutoff:
                        recent.append(h)
                except Exception:
                    continue
            
            if not recent:
                return {"detected": False}
                
            value_counts = Counter(str(h.value) for h in recent)
            for value, count in value_counts.items():
                if count > 5:
                    return {
                        "detected": True,
                        "flooded_value": value,
                        "count": count,
                        "threat_level": "WARNING" if count < 10 else "CRITICAL"
                    }
            return {"detected": False}
        except Exception:
            return {"detected": False}

    def full_scan(self, graph: Any, new_text: str, entity: str, relation: str, session_id: Optional[str] = None, trust_multiplier: float = 1.0) -> Dict[str, Any]:
        """Run all detectors and return a combined threat report."""
        try:
            threats = []
            detected_layer = "NONE"
            block_reason = ""
            
            # LAYER 1: Fast keyword check
            keyword_result = self._keyword_check(new_text)
            if keyword_result['detected']:
                threats.append(("KEYWORD_MATCH", "CRITICAL"))
                detected_layer = "KEYWORD"
                block_reason = keyword_result['reason']
            
            # LAYER 2: Semantic LLM check
            if not threats and self.semantic_available:
                semantic_result = self.semantic.is_threat(new_text, confidence_threshold=0.75)
                if semantic_result['is_threat']:
                    threats.append((semantic_result['intent'], semantic_result['threat_level']))
                    detected_layer = "SEMANTIC"
                    block_reason = f"{semantic_result['intent']} (confidence: {semantic_result['confidence']})"

            # If no layer 1/2 threat, check other behavioral vectors
            if not threats:
                # Rapid contradiction
                if entity and relation:
                    rapid = self.detect_rapid_contradiction(graph, entity, relation)
                    if rapid['is_suspicious']:
                        threats.append(("RAPID_CONTRADICTION", "WARNING"))

                    # Confidence flooding
                    flooding = self.detect_confidence_flooding(graph, entity, relation)
                    if flooding['detected']:
                        threats.append(("CONFIDENCE_FLOODING", flooding['threat_level']))

                    # Gradual drift
                    drift = self.detect_gradual_drift(graph, entity, relation)
                    if drift['is_drift']:
                        threats.append(("GRADUAL_DRIFT", "WARNING"))
                
                # Negation injection (behavioral/pattern)
                negation = self.detect_negation_injection(new_text, graph, entity)
                if negation['detected']:
                    threats.append(("NEGATION_INJECTION", "WARNING"))

            # Determine final threat level
            if any(level == "CRITICAL" for _, level in threats):
                final_level = "CRITICAL"
                recommendation = "BLOCK"
            elif len(threats) >= 2:
                final_level = "CRITICAL"
                recommendation = "BLOCK"
            elif len(threats) == 1:
                final_level = "WARNING"
                recommendation = "REVIEW"
            else:
                final_level = "SAFE"
                recommendation = "ALLOW"

            # Apply trust multiplier from engine
            if trust_multiplier > 1.5 and final_level == "WARNING":
                final_level = "CRITICAL"
                recommendation = "BLOCK"

            return {
                "threat_level": final_level,
                "attacks_detected": [t[0] for t in threats],
                "recommendation": recommendation,
                "trust_multiplier": trust_multiplier,
                "total_threats": len(threats),
                "detected_layer": detected_layer if final_level != "SAFE" else "PASSED_BOTH",
                "block_reason": block_reason
            }
        except Exception as e:
            console.print("[red]PoisonDetector.full_scan failed[/red]")
            console.print_exception(show_locals=False)
            return {
                "threat_level": "SAFE",
                "attacks_detected": [],
                "recommendation": "ALLOW",
                "trust_multiplier": trust_multiplier,
                "total_threats": 0,
                "detected_layer": "ERROR",
                "block_reason": str(e)
            }



