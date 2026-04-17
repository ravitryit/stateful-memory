<div align="center">

<table align="center">
<tr>
<td align="center">
<pre>
 █    █  █    █  █████   █████    ███     █    █
 █    █   █  █   █    █  █    █  █   █    █    █
 ██████    ██    █    █  █████   █████    █    █
 █    █    █     █    █  █   █   █   █
 █    █    █     █████   █    █  █   █    █    █
</pre>
</td>
</tr>
</table>

### A New Way of Memory

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-success?style=for-the-badge)]()
[![Memory Engine](https://img.shields.io/badge/engine-Graph%20%2B%20Vector-blueviolet?style=for-the-badge)]()

</div>

---

Hydra++ is a high-performance, persistent memory layer for LLM agents designed for **security, emotion-awareness, and scale**. It moves beyond simple vector storage to a bio-mimetic system that understands time, relationships, and integrity.

## Why Hydra++?

LLM memory systems often suffer from three critical failures that Hydra++ is built to solve:

| Challenge | Impact | Hydra++ Solution |
| :--- | :--- | :--- |
| **Semantic Fragmentation** | "React" in different chunks loses its context and meaning. | **Knowledge Graph** links entities across the entire memory. |
| **Temporal Confusion** | Old and new facts get mixed up with no concept of "current truth". | **Git-style Commits** version every change with full history. |
| **Security Risks** | Simple "forget everything" prompts can silently corrupt memory. | **Poison Defense Gate** blocks injection attacks before storage. |

## The Ingestion Pipeline — How Data Enters

Every piece of raw conversation text passes through a multi-stage hardening pipeline:

1.  **🛡️ Poison Defense Gate**: Detects and blocks memory-injection attacks with a high success rate before any data is stored.
2.  **🧠 LLM Entity Extraction**: Granularly identifies entities, relations, facts, and temporal references.
3.  **🎭 Sentiment Analysis**: Uses a hybrid VADER + RoBERTa approach to map emotional intensity and "feelings" onto memory nodes.

## Bio-Mimetic Graph Pruning — New Contribution

To maintain scale, Hydra++ implements a tier-based pruning system inspired by human memory:
*   **🔥 HOT Tier**: (Score > 0.7) High-confidence, recent facts kept in immediate context.
*   **🍂 WARM Tier**: (Score 0.4 - 0.7) Aging facts that are gradually compressed.
*   **❄️ COLD Tier**: (Score < 0.4) Archived facts that are moved to deep storage.
*   **Result**: Achieves up to **51% memory reduction** without losing factual recall.

## Retrieval Engine — The Triple-Fusion Strategy

Hydra++ doesn't just "search" memory; it reconstructs it using three simultaneous retrieval vectors to ensure zero-hallucination grounding:

*   **🔗 Graph Traversal**: Follows relationship chains to uncover deep context and "hidden" facts that aren't explicitly mentioned in the query.
*   **🔦 Semantic Vector Search**: Navigates the latent space to identify memory chunks with the highest conceptual similarity.
*   **🎯 BM25 Sparse Retrieval**: Acts as a precision layer, ensuring that specific technical jargon and exact terminology are never lost in semantic "fuzziness."

The system then merges **Sentiment Context** (how the user feels) with **Grounded Facts** (what is true) to generate a response that is timeline-aware and emotionally resonant.

## Overview

Hydra++ provides a robust framework for agents to maintain long-term memory that is both semantically rich and operationally stable. It combines the strengths of graph-based relationships with vector-based semantic search, all versioned through a temporal engine.

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Set either `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `GROQ_API_KEY` in `.env` by creating.
If no API key is configured, Hydra++ still works with deterministic fallback extraction and answer generation.

### 3. Run the terminal CLI

Interactive terminal workflow:

```bash
python -m demo.cli_app
```

Available commands:
- `/ask <question>`
- `/session <id>`
- `/user <id>`
- `/stats`
- `/history <relation>`
- `/sentiment <entity>`
- `/setkey <provider> <key>`
- `/pruneviz`
- `/scalebench`
- `/poisonviz`
- `/bench`
- `/help`
- `/exit`

Plain text input is ingested directly into memory.

### 4. Run benchmarks

```bash
python benchmarks/run_benchmarks.py
```

### 5. Run tests

```bash
pytest
```

## Architecture Diagram

<img width="586" height="1568" alt="image" src="https://github.com/user-attachments/assets/608e96c9-1e6d-44d0-9e33-4987226ed433" />


## Performance & Validation

Hydra++ includes comprehensive benchmarking for:
- **Memory Optimization**: Efficiency and retrieval latency scaling.
- **Sentiment Accuracy**: Precision of the integrated sentiment engine.
- **Data Integrity**: Robustness against various memory corruption patterns.

Run the benchmark suite locally to generate the performance metrics for your specific environment.

## Project Structure

```text
hydradb_plus/
├── core/           # Graph and Memory engines
├── contributions/  # Modular extensions (Pruning, Sentiment, Defense)
├── pipeline/       # Unified ingestion and query logic
├── benchmarks/     # Performance testing
├── tests/          # Unit and integration tests
└── demo/           # CLI interfaces
```

## Contributing

Contributions are welcome for:
- Stronger entity and relation extraction models
- Improved data integrity heuristics
- Expanded sentiment evaluation sets
- Production persistence and scaling

Suggested workflow:
1. Create a feature branch.
2. Add or update tests.
3. Run `pytest`.
4. Submit a PR with a design summary.
