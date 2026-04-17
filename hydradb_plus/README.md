# HydraDB++ : 3 Critical Contributions to HydraDB

HydraDB++ is a persistent memory layer for LLM agents built around:
- Git-style append-only temporal commits
- A `networkx` knowledge graph for factual and sentiment relationships
- A `ChromaDB` hybrid memory store for semantic and sparse retrieval
- Three critical additions:
  pruning, sentiment memory, and poisoning defense

## What HydraDB Missed

| Problem | Real Impact | Our Solution | Result |
|---------|-------------|--------------|--------|
| Graph Explosion | Slow retrieval | Smart Pruning | 60-70% smaller in synthetic benchmark |
| No Sentiment | Wrong context | Intensity Scoring | Combined model benchmark included |
| Memory Poisoning | Security risk | Attack Defense | High block rate in benchmark |

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Set either `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env`.

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
```

If no API key is configured, HydraDB++ still works with deterministic fallback extraction and answer generation.

### 3. Run the terminal CLI

Claude-style terminal workflow:

```bash
python -m hydradb_plus.demo.cli_app
```

Alternative:

```bash
python -m hydradb_plus.demo
```

Available commands:
- `/ask <question>`
- `/session <id>`
- `/user <id>`
- `/stats`
- `/history <relation>`
- `/sentiment <entity>`
- `/bench`
- `/exit`

Plain text input is ingested directly into memory.

### 4. Run benchmarks

```bash
python benchmarks/run_benchmarks.py
```

Results are saved automatically to:

```bash
benchmarks/results/benchmark_<TIMESTAMP>.json
```

### 5. Run tests

```bash
pytest
```

## Architecture Diagram

```text
                +-----------------------+
                |   CLI / Agent Input   |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                |   Defense Engine      |
                |  - contradiction      |
                |  - drift detection    |
                |  - authority injection|
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | Extraction Pipeline   |
                |  - LiteLLM / fallback |
                |  - entities           |
                |  - relations          |
                |  - facts              |
                +-----+-----------+-----+
                      |           |
          +-----------+           +-------------------+
          v                                           v
+-----------------------+                 +-----------------------+
| Knowledge Graph       |                 | Sentiment Memory      |
| - append-only edges   |                 | - VADER + RoBERTa     |
| - temporal history    |                 | - opinion intensity   |
| - current truth       |                 | - FEELS_ABOUT edges   |
+-----------+-----------+                 +-----------+-----------+
            |                                           |
            +-------------------+-----------------------+
                                |
                                v
                    +-----------------------+
                    | Memory Store          |
                    | - raw embeddings      |
                    | - enriched embeddings |
                    | - sparse keywords     |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Temporal Engine       |
                    | - git-style commits   |
                    | - session history     |
                    | - checkout snapshots  |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Graph Pruner          |
                    | - confidence score    |
                    | - HOT/WARM/COLD tiers |
                    | - archive + merge     |
                    +-----------------------+
```

## Results

HydraDB++ includes three benchmark tracks:
- Graph pruning: graph size and retrieval latency before/after pruning
- Sentiment accuracy: VADER-only vs VADER+RoBERTa
- Poison defense: successful attacks without defense vs blocked attacks with defense

Run the benchmark suite locally to generate the current numbers for your machine and environment.

## Project Structure

```text
hydradb_plus/
├── .env
├── requirements.txt
├── README.md
├── core/
├── contributions/
├── pipeline/
├── benchmarks/
├── tests/
└── demo/
```

## Notes

- The original request included a Streamlit demo. This implementation uses a terminal-first CLI instead so the system can run directly from the shell.
- `streamlit` remains listed in `requirements.txt` because it was part of the required tech stack, but the primary interface is `demo/cli_app.py`.
- All timestamps are stored as ISO datetime strings.
- All modules use docstrings and type hints.

## Contributing

Contributions are welcome for:
- stronger entity and relation extraction
- better poisoning heuristics and false-positive analysis
- larger sentiment evaluation sets
- production persistence and scaling improvements

Suggested workflow:
1. Create a feature branch.
2. Add or update tests.
3. Run `pytest`.
4. Run `python benchmarks/run_benchmarks.py` when changes affect benchmarked behavior.
5. Submit a PR with benchmark deltas and a short design summary.

