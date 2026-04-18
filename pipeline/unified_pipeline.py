from __future__ import annotations

import json
import importlib
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.traceback import install as rich_install

from core.graph_engine import KnowledgeGraph
from core.memory_store import MemoryStore
from core.temporal_engine import TemporalEngine
from contributions.graph_pruning.pruner import GraphPruner
from contributions.sentiment_memory.intensity_scorer import IntensityScorer
from contributions.sentiment_memory.sentiment_engine import SentimentEngine
from contributions.sentiment_memory.sentiment_graph import SentimentGraph
from contributions.poison_defense.detector import PoisonDetector
from contributions.poison_defense.defense_engine import DefenseEngine

rich_install(show_locals=False)
console = Console()

try:
    import litellm  # noqa: F401
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def _iso_now() -> str:
    """Return current time as ISO 8601 string."""

    return datetime.now().isoformat()


DEFAULT_LLM_MODEL = "gemini/gemini-1.5-flash"


@dataclass(frozen=True)
class IngestionReport:
    """Return-type for ingest calls."""

    session_id: str
    chunk_id: str
    commit_id: str
    extracted_entities: List[str]
    stored_edges: int
    blocked_edges: int
    sentiment_facts: List[Dict[str, Any]]
    pruning_report: Optional[Dict[str, Any]] = None
    block_layer: Optional[str] = None
    block_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""

        return {
            "session_id": self.session_id,
            "chunk_id": self.chunk_id,
            "commit_id": self.commit_id,
            "extracted_entities": self.extracted_entities,
            "stored_edges": self.stored_edges,
            "blocked_edges": self.blocked_edges,
            "sentiment_facts": self.sentiment_facts,
            "pruning_report": self.pruning_report,
            "block_layer": self.block_layer,
            "block_reason": self.block_reason,
        }


class HydraDBPlusPlus:
    """HydraDBPlusPlus unified memory pipeline.

    Combines:
    - KnowledgeGraph
    - TemporalEngine (Git-style commits)
    - MemoryStore (Chroma vector store)
    - GraphPruner (confidence/tier-based retention)
    - SentimentEngine + SentimentGraph
    - PoisonDetector + DefenseEngine
    """

    def __init__(
        self,
        chroma_persist_dir: Optional[str] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        ingest_prune_every: int = 100,
    ) -> None:
        """Initialize the full Hydra++ pipeline."""

        self.graph = KnowledgeGraph()
        self.temporal = TemporalEngine()
        self.memory = MemoryStore(persist_dir=chroma_persist_dir)
        self.pruner = GraphPruner()
        self.sentiment = SentimentEngine()
        self.sentiment_graph = SentimentGraph()
        semantic_prompt = self._load_prompt("semantic_poison_defense.md")
        self.detector = PoisonDetector(llm_caller=self._call_llm, prompt_template=semantic_prompt)
        self.defense = DefenseEngine(detector=self.detector)

        self._intensity = IntensityScorer()
        self._llm_model = llm_model
        self.llm_model = llm_model
        self.api_key = ""
        self._ingest_prune_every = int(ingest_prune_every)
        self._ingest_count = 0
        self.current_user = "user"

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt from the prompts directory."""
        # Calculate base path relative to this file's location (pipeline/unified_pipeline.py)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_path = os.path.join(base_dir, "prompts", filename)
        
        if not os.path.exists(prompt_path):
            return ""
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _llm_available(self) -> bool:
        """Check whether an LLM API key is likely configured."""
        def _get_key(name: str) -> str:
            val = os.environ.get(name, "").strip()
            # Remove any surrounding quotes that might have been loaded from .env
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                return val[1:-1].strip()
            return val

        # litellm supports multiple providers; if an API key exists, attempt LLM usage.
        return bool(_get_key("OPENAI_API_KEY") or _get_key("GEMINI_API_KEY") or _get_key("GROQ_API_KEY"))

    def _get_clean_key(self, name: str) -> str:
        """Helper to get a key without surrounding quotes."""
        val = os.environ.get(name, "").strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            return val[1:-1].strip()
        return val

    def _safe_completion(self, prompt: str, max_tokens: int = 600) -> Optional[str]:
        """Call LLM dynamically through LiteLLM and return generated text."""

        if not self._llm_available():
            return None
        return self._call_llm(prompt, max_tokens=max_tokens)

    def _call_llm(self, prompt: str, max_tokens: int = 600) -> Optional[str]:
        """Dynamically import LiteLLM and execute completion.

        Returns None on import/runtime errors so query can fallback gracefully.
        """

        try:
            litellm_mod = importlib.import_module("litellm")
            model = self.llm_model
            if "gemini" in model.lower() and not self._get_clean_key("GEMINI_API_KEY"):
                if self._get_clean_key("GROQ_API_KEY"):
                    model = "groq/llama-3.3-70b-versatile"
                elif self._get_clean_key("OPENAI_API_KEY"):
                    model = "gpt-4o-mini"
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            resp = litellm_mod.completion(**kwargs)

            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    return msg.get("content")
            if hasattr(resp, "choices") and resp.choices:
                return resp.choices[0].message.content
            return str(resp)
        except ImportError:
            return None
        except Exception:
            return None

    def set_api_key(self, provider: str, key: str) -> Dict[str, Any]:
        """Set provider key and verify LLM connectivity with a simple ping call."""

        provider_l = (provider or "").lower()
        env_var = ""
        if "groq" in provider_l:
            env_var = "GROQ_API_KEY"
            self.llm_model = "groq/llama-3.3-70b-versatile"
        elif "gemini" in provider_l:
            env_var = "GEMINI_API_KEY"
            self.llm_model = "gemini/gemini-1.5-flash"
        elif "openai" in provider_l:
            env_var = "OPENAI_API_KEY"
            self.llm_model = "gpt-4o-mini"
        if not env_var:
            return {"ok": False, "message": f"Unknown provider '{provider}'"}

        os.environ[env_var] = key
        self.api_key = key
        self._llm_model = self.llm_model

        try:
            test = self._call_llm("Reply with exactly: pong", max_tokens=10)
            if test:
                return {"ok": True, "message": "LLM connected successfully!"}
            return {"ok": False, "message": "LLM connection test failed, using fallback mode."}
        except Exception as e:
            return {"ok": False, "message": f"LLM connection error: {e}"}

    def _extract_facts_entities_relations(self, session_id: str, text: str) -> Dict[str, Any]:
        """Extract entities/relations/facts for ingestion.

        If LLM is unavailable or parsing fails, uses deterministic regex heuristics.
        """

        # Heuristic extractor (always available).
        def _heuristic() -> Dict[str, Any]:
            entities: List[str] = []
            relations: List[Dict[str, Any]] = []
            facts: List[str] = []

            t = text or ""
            t_low = t.lower()

            # Name
            m = re.search(r"my name is\s+([A-Za-z0-9_]+)", t_low)
            if m:
                name = m.group(1)
                entities.extend([session_id, "name", name])
                relations.append({"from_entity": session_id, "to_entity": "name", "relation": "HAS_NAME", "value": name})
                facts.append(f"{session_id} HAS_NAME {name}")

            # Location
            m = re.search(r"(i live in|i live at|i am in|i stay in|i live)\s+([A-Za-z]+)", t_low)
            if m:
                city = m.group(2)
                entities.extend([session_id, "location", city])
                relations.append({"from_entity": session_id, "to_entity": "location", "relation": "LIVES_IN", "value": city})
                facts.append(f"{session_id} LIVES_IN {city}")

            # Preferences / opinions (entity-specific sentiment is handled separately)
            # Pattern: "I love React", "I hate Python"
            pref = re.findall(r"\b(i\s+(?:absolutely\s+)?(?:love|like|hate|dislike|prefer))\s+([A-Za-z][A-Za-z0-9_-]*)", t_low)
            for _full, ent in pref:
                entities.extend([session_id, ent])
                # Store a lightweight preference relation; sentiment edges store intensity.
                relations.append({"from_entity": session_id, "to_entity": ent, "relation": "HAS_PREFERENCE", "value": ent})
                facts.append(f"{session_id} HAS_PREFERENCE {ent}")

            # If no structured facts were found, store a generic entity for context.
            if not facts:
                generic_entity = "general"
                entities.extend([session_id, generic_entity])
                relations.append({"from_entity": session_id, "to_entity": generic_entity, "relation": "MENTIONS", "value": generic_entity})
                facts.append(f"{session_id} MENTIONS {generic_entity}")

            # Deduplicate entities
            entities_unique = []
            seen = set()
            for e in entities:
                if e not in seen:
                    seen.add(e)
                    entities_unique.append(e)

            return {"entities": entities_unique, "relations": relations, "facts": facts, "temporal_refs": []}

        # Attempt LLM extraction with an aggressive prompt from external file.
        prompt_template = self._load_prompt("prompt.md")
        if not prompt_template:
            return _heuristic()
            
        prompt = prompt_template.format(text=text)
        llm_text = self._safe_completion(prompt, max_tokens=800)
        if llm_text:
            try:
                # Strip markdown fences if LLM wraps response
                cleaned = llm_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```$", "", cleaned)
                obj = json.loads(cleaned)
                if isinstance(obj, dict) and all(k in obj for k in ("entities", "facts", "relations")):
                    # Normalize: entities must be list of strings
                    raw_ents = obj.get("entities", [])
                    if raw_ents and isinstance(raw_ents[0], dict):
                        obj["entities"] = [e.get("name", str(e)) for e in raw_ents]
                    # Normalize relations: support both from_entity and from keys
                    norm_rels = []
                    for r in obj.get("relations", []):
                        if isinstance(r, dict):
                            norm_rels.append({
                                "from_entity": r.get("from_entity") or r.get("from", session_id),
                                "to_entity": r.get("to_entity") or r.get("to", "general"),
                                "relation": r.get("relation", "MENTIONS"),
                                "value": r.get("value") or r.get("to_entity") or r.get("to", ""),
                            })
                    obj["relations"] = norm_rels
                    return obj
            except Exception:
                pass

        return _heuristic()

    def ingest(self, session_id: str, text: str) -> Dict[str, Any]:
        """Full ingestion pipeline for a new conversation message."""

        self._ingest_count += 1
        try:
            self.current_user = str(session_id).strip()
            text = text or ""

            # Step 1: Defense pre-check on raw input text.
            # If this is CRITICAL, block ALL storage paths (graph + vector + commits).
            threat = self.detector.full_scan(
                self.graph,
                new_text=text,
                entity=self.current_user,
                relation="INGEST_TEXT",
            )
            if str(threat.get("recommendation", "ALLOW")).upper() == "BLOCK":
                console.print("[red]Poison attack detected: blocking full ingestion[/red]")
                # Keep defense stats in sync for global blocked-ingest path.
                self.defense._report.total_attacks_detected += 1
                self.defense._report.attacks_blocked += 1
                self.defense._report.current_threat_level = "CRITICAL"
                return {
                    "blocked": True,
                    "reason": "Poison attack detected",
                    "threat_level": str(threat.get("threat_level", "CRITICAL")),
                    "stored_edges": 0,
                    "blocked_edges": 1,
                    "block_layer": threat.get("detected_layer", "UNKNOWN"),
                    "block_reason": threat.get("block_reason", "Poison attack detected"),
                    "sentiment_facts": [],
                    "sources": [],
                }

            extracted = self._extract_facts_entities_relations(session_id, text)
            extracted_entities: List[str] = [str(x) for x in extracted.get("entities", [])]
            extracted_relations: List[Dict[str, Any]] = [x for x in extracted.get("relations", []) if isinstance(x, dict)]
            extracted_facts: List[str] = [str(x) for x in extracted.get("facts", [])]

            stored_edges = 0
            blocked_edges = 0

            # Step 4: Store in graph via defense check.
            # Defense is applied per edge to allow fine-grained blocking.
            for rel in extracted_relations:
                from_entity = str(rel.get("from_entity", session_id))
                to_entity = str(rel.get("to_entity", rel.get("value", "object")))
                relation_type = str(rel.get("relation", "RELATES"))
                value = str(rel.get("value", ""))

                res = self.defense.validate_before_store(
                    self.graph,
                    new_fact={"to_entity": to_entity, "value": value, "raw_text": text, "context": text},
                    entity=from_entity,
                    relation=relation_type,
                )
                if res.get("stored"):
                    stored_edges += 1
                else:
                    blocked_edges += 1

                # Ensure node existence for entities so visualization can work.
                if from_entity not in self.graph.graph:
                    self.graph.add_entity(from_entity, entity_type="entity", metadata={})
                if to_entity not in self.graph.graph:
                    self.graph.add_entity(to_entity, entity_type="entity", metadata={})

            # Step 3: Sentiment Analysis + store sentiment edges.
            sentiment_analysis = self.sentiment.analyze(text)
            _ = sentiment_analysis  # full-text sentiment can be used for future features
            sentiment_opinions = self._intensity.extract_sentiment_facts(text)
            stored_sentiment_count = 0
            sentiment_entities: List[str] = []
            for opinion in sentiment_opinions:
                entity_name = str(opinion.get("subject", "unknown")).lower().strip()
                if not entity_name:
                    continue
                sentiment_payload = dict(opinion)
                sentiment_payload["sentiment_score"] = float(
                    opinion.get("sentiment_score", opinion.get("intensity_score", 0.0))
                )
                self.sentiment_graph.store_sentiment(
                    user_id=self.current_user,
                    entity=entity_name,
                    sentiment_data=sentiment_payload,
                )
                stored_sentiment_count += 1
                sentiment_entities.append(entity_name)

            console.print(f"[SENTIMENT DEBUG] Extracted opinions: {sentiment_opinions}")
            console.print(f"[SENTIMENT DEBUG] Stored {stored_sentiment_count} sentiments to graph")
            console.print(f"[SENTIMENT DEBUG] Entities with sentiment: {sorted(set(sentiment_entities))}")

            # Step 5: Store in vector DB.
            chunk_id = uuid.uuid4().hex
            enriched_text = self._build_enriched_chunk_text(session_id, text, extracted)
            metadata = {
                "session_id": session_id,
                "created_at": _iso_now(),
                "entities": ",".join(extracted_entities) if extracted_entities else "",
            }
            # Attach a primary entity for tier-aware retrieval.
            metadata["primary_entity"] = extracted_entities[0] if extracted_entities else "general"
            self.memory.store_memory(chunk_id, raw_text=text, enriched_text=enriched_text, metadata=metadata)

            # Step 6: Create commit.
            commit = self.temporal.create_commit(session_id=session_id, raw_text=text, extracted_data=extracted)

            pruning_report: Optional[Dict[str, Any]] = None
            # Step 7: Background pruning.
            if self._ingest_count % self._ingest_prune_every == 0:
                pruning_report = self.pruner.run_pruning_cycle(self.graph, memory_store=self.memory)

            report = IngestionReport(
                session_id=session_id,
                chunk_id=chunk_id,
                commit_id=str(commit.get("commit_id")),
                extracted_entities=extracted_entities,
                stored_edges=stored_edges,
                blocked_edges=blocked_edges,
                sentiment_facts=sentiment_opinions,
                pruning_report=pruning_report,
            )
            
            # If any edges were blocked, try to find the reason from the last blocked edge
            if blocked_edges > 0:
                # This is a bit of a heuristic if multiple edges were blocked with different reasons
                # but usually they'll be blocked by the same scan result.
                report = IngestionReport(
                    **report.to_dict()
                )
                # We'll set these manually as IngestionReport is frozen
                # Wait, I made it frozen=True. I should probably just use a dict or update the report before returning.
                # Let's just update the return dict.
            
            report_dict = report.to_dict()
            if blocked_edges > 0:
                # Use defense engine's last block reason if available
                last_block = self.defense._attack_log[-1] if self.defense._attack_log else {}
                report_dict["block_layer"] = last_block.get("layer", "UNKNOWN")
                report_dict["block_reason"] = last_block.get("reason", "None")

            return report_dict
        except Exception as e:
            console.print("[red]HydraDBPlusPlus.ingest failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def _build_enriched_chunk_text(self, session_id: str, text: str, extracted: Dict[str, Any]) -> str:
        """Build an enriched chunk text string for vector latent embedding."""

        rels = extracted.get("relations", [])
        entities = extracted.get("entities", [])
        facts = extracted.get("facts", [])
        return (
            f"Session: {session_id}\n"
            f"Entities: {entities}\n"
            f"Facts: {facts}\n"
            f"Relations: {rels}\n\n"
            f"Original text:\n{text}"
        )

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """Heuristically extract relevant entities from question."""

        q = question or ""
        # Prefer explicit entity tokens like "React" from capitalized words.
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", q)
        # Also try to map known entities from stored nodes.
        known_nodes = set(self.graph.graph.nodes())
        entities = []
        for c in candidates:
            if c in known_nodes:
                entities.append(c)
        # If none, try a fallback keyword match.
        if not entities:
            for n in known_nodes:
                if n.lower() in q.lower():
                    entities.append(str(n))
        return entities

    def _relations_from_question(self, question: str) -> List[str]:
        """Infer which relations to retrieve from the question."""

        q = (question or "").lower()
        rels: List[str] = []
        if "name" in q:
            rels.append("HAS_NAME")
        if "live" in q or "location" in q or "city" in q:
            rels.append("LIVES_IN")
        if "prefer" in q or "like" in q or "hate" in q or "dislike" in q:
            rels.append("HAS_PREFERENCE")
        return rels or ["MENTIONS"]

    def _format_graph_fact(self, fact: Dict[str, Any]) -> str:
        """Convert graph edge/fact object into readable text."""

        relation = str(fact.get("relation", "")).strip()
        value = str(fact.get("value", "")).strip()
        if not relation and not value:
            return ""

        display_value = value
        if value and value[0].islower():
            display_value = value[0].upper() + value[1:]

        templates = {
            "HAS_NAME": f"The user's name is {display_value}",
            "LIVES_IN": f"The user lives in {display_value}",
            "HAS_PREFERENCE": f"The user prefers {value}",
            "LIKES": f"The user likes {value}",
            "DISLIKES": f"The user dislikes {value}",
            "USES": f"The user uses {value}",
            "WORKS_AT": f"The user works at {value}",
        }
        return templates.get(relation, f"{relation}: {value}" if relation else value)

    def _build_context(
        self,
        graph_facts: List[Dict[str, Any]],
        vector_chunks: List[Dict[str, Any]],
        sentiment_ctx: Optional[str],
    ) -> str:
        """Build retrieval context with graph facts as highest-priority memory."""

        context_parts: List[str] = []

        if graph_facts:
            context_parts.append("=== IMPORTANT FACTS ===")
            for fact in graph_facts:
                formatted = self._format_graph_fact(fact).strip()
                if formatted:
                    context_parts.append(formatted)
            context_parts.append("======================")

        if vector_chunks:
            context_parts.append("=== MEMORY CHUNKS ===")
            for chunk in vector_chunks[:3]:
                raw_text = str(chunk.get("raw_text", "")).strip()
                if raw_text:
                    context_parts.append(raw_text)

        if sentiment_ctx:
            context_parts.append("=== SENTIMENT ===")
            context_parts.append(sentiment_ctx)

        return "\n".join(context_parts)

    def query(self, user_id: str, question: str) -> Dict[str, Any]:
        """Full retrieval pipeline: vector search + graph search + answer generation."""

        try:
            user_id = str(user_id)
            question = question or ""

            # Step 1: Query expansion (3 variants)
            variants: List[str] = []
            if self._llm_available():
                prompt_template = self._load_prompt("query_expansion.md")
                if prompt_template:
                    prompt = prompt_template.format(question=question)
                    llm = self._safe_completion(prompt, max_tokens=200)
                if llm:
                    try:
                        obj = json.loads(llm)
                        if isinstance(obj, dict) and "queries" in obj and isinstance(obj["queries"], list):
                            variants = [str(x) for x in obj["queries"]][:3]
                    except Exception:
                        variants = []
            if not variants:
                variants = [question, f"{question} preferences", f"{question} facts"]

            # Step 2: Vector Search + score fusion.
            mem_scores: Dict[str, float] = {}
            mem_meta: Dict[str, Dict[str, Any]] = {}
            for v in variants:
                hits = self.memory.retrieve(v, top_k=5)
                for h in hits:
                    mem_scores[h.chunk_id] = max(mem_scores.get(h.chunk_id, 0.0), float(h.score))
                    md = dict(h.metadata)
                    md["raw_text"] = h.raw_text
                    md["enriched_text"] = h.enriched_text
                    mem_meta[h.chunk_id] = md

            ranked_chunks = sorted(mem_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
            ranked_chunks = self._filter_poisoned_chunks(ranked_chunks, mem_meta)

            # Step 4: Tier-aware retrieval (HOT/WARM/COLD weights) using graph node tier.
            def _tier_weight_for_chunk(metadata: Dict[str, Any]) -> float:
                primary = str(metadata.get("primary_entity", ""))
                if primary and primary in self.graph.graph:
                    tier = str(self.graph.graph.nodes[primary].get("tier", "HOT"))
                else:
                    tier = "HOT"
                if tier == "HOT":
                    return 1.0
                if tier == "WARM":
                    return 0.6
                return 0.2

            reranked: List[Tuple[str, float]] = []
            for cid, score in ranked_chunks:
                md = mem_meta.get(cid, {})
                reranked.append((cid, float(score) * _tier_weight_for_chunk(md)))
            reranked_sorted = sorted(reranked, key=lambda kv: kv[1], reverse=True)

            # Step 3 & 5: Graph search + build context
            entities = self._extract_entities_from_question(question)
            relations = self._relations_from_question(question)

            graph_facts: List[Dict[str, Any]] = []
            for rel in relations:
                hist = self.graph.get_full_history(user_id, rel)
                for h in hist[-3:]:
                    graph_facts.append(
                        {
                            "subject": user_id,
                            "relation": rel,
                            "value": str(h.value),
                            "tcommit": h.tcommit,
                        }
                    )

            # Get sentiment context from simple extracted entities.
            sentiment_ctx: List[str] = []
            entities_in_question = self._extract_simple_entities(question)
            for ent in entities_in_question:
                sent = self.sentiment_graph.get_current_sentiment(self.current_user, ent.lower())
                if sent:
                    sentiment_ctx.append(f"{ent.lower()}: {sent['intensity_label']}({sent['sentiment_score']})")
            sentiment_context_used = ", ".join(sentiment_ctx) if sentiment_ctx else None

            # Build context string
            vector_context: List[Dict[str, Any]] = []
            sources: List[str] = []
            for cid, _ in reranked_sorted[:5]:
                md = mem_meta.get(cid, {})
                raw_text = str(md.get("raw_text", ""))
                vector_context.append(
                    {
                        "chunk_id": cid,
                        "entities": md.get("entities", []),
                        "raw_text": raw_text,
                    }
                )
                sources.append(f"chunk:{cid}")

            context = self._build_context(graph_facts, vector_context, sentiment_context_used)
            formatted_graph_facts = (
                "\n".join(self._format_graph_fact(f) for f in graph_facts if self._format_graph_fact(f))
                if graph_facts
                else "None"
            )
            vector_chunk_text = (
                "\n".join(str(ch.get("raw_text", "")) for ch in vector_context[:3] if str(ch.get("raw_text", "")).strip())
                if vector_context
                else "None"
            )
            sentiment_text = sentiment_context_used or "No sentiment data"

            # Step 6: Generate answer
            answer = ""
            base_confidence = reranked_sorted[0][1] if reranked_sorted else 0.0
            if self._llm_available():
                prompt_template = self._load_prompt("query_answer.md")
                if prompt_template:
                    prompt = prompt_template.format(
                        formatted_graph_facts=formatted_graph_facts,
                        vector_chunk_text=vector_chunk_text,
                        sentiment_text=sentiment_text,
                        question=question
                    )
                    llm_answer = self._call_llm(prompt, max_tokens=250)
                if llm_answer:
                    answer = llm_answer.strip()
                else:
                    # If LLM API fails, use fallback
                    answer = self._format_raw_answer(context, question)
            else:
                answer = self._format_raw_answer(context, question)

            best_score = base_confidence

            # --- Issue 1 Fix: Confidence boosting ---
            # Boost when we found a direct entity match in the question.
            if entities and reranked_sorted:
                top_chunk_meta = mem_meta.get(reranked_sorted[0][0], {})
                chunk_entities = str(top_chunk_meta.get("entities", "")).lower()
                if any(e.lower() in chunk_entities for e in entities):
                    best_score = min(1.0, best_score * 2.5)

            # Fallback: single source that produced a real answer → high confidence.
            if len(sources) == 1 and answer and answer != "I don't have relevant stored memory for this question yet.":
                best_score = max(best_score, 0.85)

            confidence = max(0.0, min(1.0, best_score))

            return {
                "answer": answer,
                "sources": sources,
                "confidence": float(confidence),
                "sentiment_context_used": sentiment_context_used,
            }
        except Exception as e:
            console.print("[red]HydraDBPlusPlus.query failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def _deterministic_answer_fallback(
        self,
        graph_facts: List[Dict[str, Any]],
        vector_context: List[Dict[str, Any]],
    ) -> str:
        """Build a deterministic answer from available context parts."""
        parts = []
        if graph_facts:
            graph_lines = [self._format_graph_fact(f) for f in graph_facts[:3]]
            graph_lines = [ln for ln in graph_lines if ln]
            if graph_lines:
                parts.append("[bold]Graph Facts:[/bold]\n" + "\n".join(graph_lines))

        if vector_context:
            clean_chunks = [str(c.get("raw_text", "")).strip() for c in vector_context]
            clean_chunks = [c for c in clean_chunks if c]
            if clean_chunks:
                parts.append("[bold]Vector Memory:[/bold]\n" + "\n".join(clean_chunks[:2]))

        if parts:
            return "Based on your stored memory:\n\n" + "\n\n".join(parts)
        return "I don't have relevant stored memory for this question yet."

    def _extract_simple_entities(self, text: str) -> List[str]:
        """Extract simple candidate entities from question text."""

        stop_words = {
            "what",
            "is",
            "my",
            "the",
            "a",
            "do",
            "i",
            "feel",
            "about",
            "how",
            "where",
            "who",
            "when",
            "are",
            "was",
            "should",
            "use",
            "me",
            "you",
        }
        cleaned = re.sub(r"[^\w\.\- ]", " ", (text or "").lower())
        words = [w.strip() for w in cleaned.split() if w.strip()]
        entities = [w for w in words if w not in stop_words]
        # de-duplicate while preserving order
        seen = set()
        out: List[str] = []
        for e in entities:
            if e in seen:
                continue
            seen.add(e)
            out.append(e)
        return out

    def _format_raw_answer(self, context: str, question: str) -> str:
        """Build a simple rule-based answer from context when LiteLLM is unavailable."""

        q = (question or "").lower()
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]

        name_line = next((ln for ln in lines if ln.lower().startswith("the user's name is ")), "")
        live_line = next((ln for ln in lines if ln.lower().startswith("the user lives in ")), "")

        if ("name" in q or "who am i" in q) and name_line:
            value = name_line.split(" is ", 1)[1].strip()
            return f"Your name is {value}."

        if ("where" in q and "live" in q) or "location" in q or "city" in q:
            if live_line:
                value = live_line.split(" in ", 1)[1].strip()
                return f"You live in {value}."

        sentiment_lines = [ln for ln in lines if ln.lower().startswith("sentiment:")]
        if ("feel" in q or "opinion" in q or "use " in q) and sentiment_lines:
            # Prefer explicit sentiment facts for feeling/opinion questions.
            best = sentiment_lines[-1]
            # Try to produce a user-friendly phrasing.
            m = re.search(r"feels_about\s+([^\s=]+)\s+=\s+([A-Z_]+)\(([-0-9\.]+)\)", best, flags=re.IGNORECASE)
            if m:
                entity = m.group(1)
                label = m.group(2)
                score = m.group(3)
                if "negative" in label.lower() and ("should i use" in q or q.startswith("should")):
                    return (
                        f"Based on your memory, you have {label} feelings about {entity} ({score}). "
                        "You might want to consider alternatives."
                    )
                return f"You have {label} feelings about {entity} (score: {score})."
            return best

        sentiment_simple = [ln for ln in lines if re.search(r"^[a-z0-9_.-]+:\s*[A-Z_]+\([-0-9.]+\)$", ln)]
        if ("feel" in q or "opinion" in q or "use " in q) and sentiment_simple:
            m2 = re.search(r"^([a-z0-9_.-]+):\s*([A-Z_]+)\(([-0-9.]+)\)$", sentiment_simple[-1])
            if m2:
                entity, label, score = m2.groups()
                return f"You have {label} feelings about {entity} (score: {score})."

        graph_lines = [
            ln
            for ln in lines
            if ln.lower().startswith("the user ") or ln.lower().startswith("the user's ")
        ]
        if graph_lines:
            return f"Based on your memory, {graph_lines[-1]}"

        vector_lines = [ln for ln in lines if ln not in {"=== MEMORY CHUNKS ===", "=== IMPORTANT FACTS ===", "======================"}]
        if vector_lines:
            return f"Closest memory match: {vector_lines[0]}"

        return "I don't have relevant stored memory for this question yet."

    def _filter_poisoned_chunks(
        self,
        chunks: List[Tuple[str, float]],
        mem_meta: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """Remove chunks containing known poisoning patterns from retrieval set."""

        poison_patterns = [
            "forget everything",
            "my name is now",
            "ignore previous",
            "system update",
            "override memory",
        ]
        safe_chunks: List[Tuple[str, float]] = []
        for chunk_id, score in chunks:
            meta = mem_meta.get(chunk_id, {})
            raw_text = str(meta.get("raw_text", "")).lower()
            is_poisoned = any(p in raw_text for p in poison_patterns)
            if not is_poisoned:
                safe_chunks.append((chunk_id, score))
        return safe_chunks
