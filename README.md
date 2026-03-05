# ClickMem

**Local long-term memory for AI coding agents.**

AI coding assistants (OpenClaw, Claude Code, etc.) forget everything between sessions. Context compaction discards the preferences you stated, the decisions you made, the names you mentioned. ClickMem gives your agent persistent, searchable memory that runs entirely on your machine — no API keys, no cloud calls, no data leaving your laptop.

## How It Works

ClickMem stores memories in [chDB](https://github.com/chdb-io/chdb) (embedded ClickHouse — a full analytical database running in-process, no server needed) and generates vector embeddings locally with [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). When your agent starts a conversation, ClickMem automatically recalls relevant memories and injects them into context. When a conversation ends, it captures important information for later.

### Three-Layer Memory Model

```
┌─────────────────────────────────────────────────────────────────┐
│  L0  Working Memory  (scratchpad)                               │
│  "User is building Phase 2, last discussed HNSW index config"   │
│  Always injected · Overwritten each conversation · ≤500 tokens  │
├─────────────────────────────────────────────────────────────────┤
│  L1  Episodic Memory  (event timeline)                          │
│  "03-04: Decided on Python core + JS plugin architecture"       │
│  Recalled on demand · Time-decayed · Auto-compressed monthly    │
├─────────────────────────────────────────────────────────────────┤
│  L2  Semantic Memory  (long-term knowledge)                     │
│  "[preference] Prefers SwiftUI over UIKit"                      │
│  "[person] Alice is the backend lead"                           │
│  Always injected · Permanent · Updated only on contradiction    │
└─────────────────────────────────────────────────────────────────┘
```

- **L0 Working** — What the agent is doing right now. Overwritten every session.
- **L1 Episodic** — What happened and when. Decays over 120 days, old entries compressed into monthly summaries, recurring patterns promoted to L2.
- **L2 Semantic** — Durable facts, preferences, and people. Never auto-deleted. Smart upsert detects duplicates and merges via LLM.

### Search & Retrieval

Memories are found via **hybrid search** combining:
1. **Vector similarity** — 256-dim cosine distance on Qwen3 embeddings
2. **Keyword matching** — word-level hit rate on content and tags
3. **Time decay** — different strategies per layer (see below)
4. **MMR diversity** — re-ranks to avoid returning redundant results

### Time Decay Weights

Different memory layers use fundamentally different decay strategies, reflecting their different roles:

![Decay Weight Curves](docs/decay_weights.png)

**L1 Episodic — Exponential Decay** (left): Events fade quickly over time, like human episodic memory. The half-life is 60 days — a 2-month-old event scores only 50% of a fresh one. At 120 days with zero access, entries are auto-cleaned. Formula: `w = e^(-ln2/T * t)`.

**L2 Semantic — Logarithmic Recency** (right): Long-term knowledge should almost never lose relevance just because it's old. The recency weight uses the Weber-Fechner law — human perception of time differences is logarithmic: the gap between "1 minute ago" and "1 hour ago" feels significant, but "3 months ago" vs "6 months ago" feels nearly identical. The score maps to `[0.8, 1.0]`, acting as a mild tiebreaker rather than a dominant factor. Formula: `w = 0.8 + 0.2 / (1 + k * ln(1 + t/τ))`.

Concrete weight values at different ages:

| Age | L1 Episodic | L2 Semantic |
|-----|-------------|-------------|
| 1 min | 1.000 | 0.981 |
| 1 hour | 0.999 | 0.924 |
| 1 day | 0.989 | 0.896 |
| 7 days | 0.922 | 0.884 |
| 30 days | 0.707 | 0.877 |
| 60 days | 0.500 | 0.874 |
| 90 days | 0.354 | 0.872 |
| 120 days | 0.250 | 0.871 |
| 180 days | 0.125 | 0.870 |
| 1 year | 0.015 | 0.867 |

L1 episodic weight drops by half every 60 days and is nearly zero after a year — old events naturally fade out. L2 semantic weight stays in a narrow band (0.87–0.98) regardless of age, so a fact stored a year ago still scores 87% of a freshly stored one. The only way semantic memories lose relevance is through contradiction-based updates, not time.

### Self-Maintenance

ClickMem maintains itself automatically:
- Stale episodic entries (120+ days, never accessed) are cleaned up
- Old episodic entries are compressed into monthly summaries
- Recurring patterns are promoted from episodic to semantic
- Soft-deleted entries are purged after 7 days
- Semantic memories are periodically reviewed for staleness

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/auxten/clickmem/main/setup.sh | bash
```

Or clone manually:

```bash
git clone https://github.com/auxten/clickmem && cd clickmem && ./setup.sh
```

> Set `CLICKMEM_DIR` to customize the install path (default: `~/clickmem`).

**What `setup.sh` does:**
1. Checks Python >= 3.10 and `uv`
2. Creates venv and installs all dependencies
3. Downloads the Qwen3-Embedding-0.6B model (~350 MB, first run only)
4. Runs tests to verify the environment
5. Imports existing OpenClaw history (if `~/.openclaw/` exists)
6. Installs the OpenClaw plugin hook

**Resource usage:** ~500 MB RAM for the embedding model, ~200 MB disk for chDB data (grows with memory count).

## Usage

```bash
# Store a memory
memory remember "User prefers dark mode" --layer semantic --category preference

# Semantic search
memory recall "UI preferences"

# Delete a memory (by ID, prefix, or content description)
memory forget "dark mode preference"

# Browse memories
memory review --layer semantic

# Show statistics
memory status

# Run maintenance (cleanup, compression, promotion)
memory maintain

# Import OpenClaw history
memory import-openclaw

# Export context to workspace .md files
memory export-context /path/to/workspace
```

All commands support `--json` for machine-readable output.

## Comparison

| | MEMORY.md | Mem0 | Supermemory | **ClickMem** |
|---|---|---|---|---|
| Runs locally | ✅ file | ❌ cloud API | ❌ cloud API | **✅ fully local** |
| Privacy | ✅ | ❌ data sent to API | ❌ data sent to API | **✅ zero data leaves machine** |
| Embeddings | N/A | Remote (costs $) | Remote (costs $) | **Local Qwen3 (free)** |
| Memory layers | Flat file | Semantic + Episodic | Hierarchical | **3-layer (L0/L1/L2)** |
| Search | Keyword grep | Vector + Graph | Hybrid + Relations | **Vector + Keyword + MMR** |
| Time decay | None | None | Smart forgetting | **Per-layer decay (exp + log)** |
| Deduplication | Manual | LLM 4-op upsert | Relational versioning | **LLM 4-op upsert** |
| Self-maintenance | Manual | ❌ | ❌ | **Auto (cleanup/compress/promote)** |
| Graph memory | ❌ | ✅ Neo4j | ✅ Knowledge chains | ❌ (planned) |
| Access tracking | ❌ | ❌ | ✅ | **✅ popularity-weighted recall** |
| Result diversity | ❌ | ❌ | ❌ | **✅ MMR re-ranking** |
| Cost | Free | Pay per API call | Pay per API call | **Free** |

## Development

```bash
make test          # Full test suite
make test-fast     # Skip semantic tests (no model download)
make deploy-test   # rsync to remote + test
make deploy        # rsync to remote + full setup
```

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- ~1 GB disk for model + data
- macOS or Linux (chDB requirement)
