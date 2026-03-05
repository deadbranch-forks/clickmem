# ClickMem Competitive Analysis & Optimization Plan

## 1. Competitor Overview

| Dimension | **Mem0** | **Supermemory** | **MemOS** | **ClickMem (current)** |
|-----------|---------|----------------|-----------|----------------------|
| Positioning | General AI memory layer | Agent memory engine | LLM memory OS | Local AI coding agent memory |
| Deployment | Cloud + Self-host | Cloud (MCP) | Self-host | Fully local |
| Storage | Vector DB + Graph DB | Cloudflare DO | SQLite / multi-backend | chDB (embedded ClickHouse) |
| Embeddings | Remote API | Remote API | Local / Remote | **Local Qwen3** (advantage) |
| Privacy | Requires API key | Cloud-hosted | Can run locally | **Fully local** (advantage) |
| License | Apache 2.0 | Proprietary | Apache 2.0 | (TBD) |

## 2. Core Capability Comparison

### 2.1 Memory Types

| Memory Type | Mem0 | Supermemory | MemOS | ClickMem |
|-------------|------|-------------|-------|----------|
| Semantic / long-term | ✅ vector + graph | ✅ knowledge chains | ✅ textual memory | ✅ L2 semantic |
| Episodic / short-term | ✅ episodic | ✅ session-based | ✅ activation memory | ✅ L1 episodic |
| Working memory | ✅ context window | ✅ working memory | ✅ KV cache | ✅ L0 working |
| **Graph memory** | ✅ Neo4j / Memgraph | ✅ relation chains | ❌ | ❌ **missing** |
| **Parametric memory** | ❌ | ❌ | ✅ LoRA weights | ❌ (N/A) |

### 2.2 Write / Update Strategy

| Capability | Mem0 | Supermemory | ClickMem |
|-----------|------|-------------|----------|
| Extract → dedup pipeline | ✅ 2-phase (Extract → Update) | ✅ chunk + contextual | ✅ upsert (search → LLM judge) |
| Contradiction detection | ✅ LLM 4-op (ADD/UPDATE/DELETE/NOOP) | ✅ isLatest field | ✅ same as Mem0 |
| **Relational versioning** | ✅ graph edges | ✅ Updates/Extends/Derives | ❌ **missing** |
| **Entity extraction** | ✅ LLM → graph nodes/edges | ✅ implicit in knowledge chains | ⚠️ entities field exists but unused in retrieval |
| Batch operations | ✅ batch up to 1000 | ✅ | ❌ |

### 2.3 Retrieval Strategy

| Capability | Mem0 | Supermemory | ClickMem |
|-----------|------|-------------|----------|
| Vector similarity | ✅ | ✅ | ✅ |
| Keyword matching | ✅ | ✅ | ✅ |
| Time decay | ❌ (not explicit) | ✅ smart forgetting | ✅ **per-layer decay** (advantage) |
| MMR diversity | ❌ | ❌ | ✅ **MMR** (advantage) |
| **Graph traversal** | ✅ multi-hop reasoning | ✅ relation chain traversal | ❌ **missing** |
| **Temporal reasoning** | ⚠️ basic | ✅ temporal grounding | ❌ **missing** |
| access_count weighting | ❌ | ✅ frequency-weighted | ⚠️ field exists but ignored in scoring |

### 2.4 Self-Maintenance

| Capability | Mem0 | Supermemory | ClickMem |
|-----------|------|-------------|----------|
| Stale cleanup | ✅ | ✅ smart forgetting | ✅ cleanup_stale |
| Compression / summarization | ❌ (external) | ✅ context rewriting | ✅ compress_episodic |
| Pattern promotion | ❌ | ✅ hierarchical promotion | ✅ promote_to_semantic |
| Semantic review | ❌ | ❌ | ✅ review_semantic |
| **Event-driven maintenance** | ❌ | ❌ | ✅ session boundary hook |

## 3. ClickMem Strengths & Gaps

### Strengths (Keep)
1. **Fully local** — zero API cost, zero data leakage; the only solution that needs no cloud
2. **Local embedding model** — Qwen3-Embedding-0.6B, no remote API required
3. **Per-layer time decay** — L1 exponential + L2 logarithmic, finer-grained than competitors
4. **MMR diversity** — result de-duplication via re-ranking; no competitor offers this
5. **Event-driven maintenance** — triggered on session boundary, more efficient than scheduled cron
6. **Embedded chDB** — no separate database process needed

### Gaps
1. **No entity relation graph** — Mem0 has Neo4j, Supermemory has knowledge chains; our entities field is underutilized
2. **No relational versioning** — no update/extends/derives relationship tracking between memories
3. **access_count not used in scoring** — field exists but retrieval ignores it
4. **No temporal reasoning** — cannot answer "what did I do last week" style time-range queries
5. **Capture too simplistic** — stores raw dialog text without LLM-based extraction
6. **Entities unused in retrieval** — stored but never matched during search

## 4. Optimization Plan

Ordered by ROI — low-cost, high-impact improvements first.

### P0: Immediate (this iteration)

#### 4.1 access_count in retrieval scoring
- **Before**: `access_count` exists in DB but `retrieval.py` ignores it
- **After**: logarithmic popularity boost — frequently recalled memories score higher
- **Effort**: ~20 lines
- **Impact**: high-frequency memories surface first, directly improves recall quality

#### 4.2 Entities in keyword matching
- **Before**: `entities` field stored but not matched during retrieval
- **After**: `_keyword_score` matches against content + tags + entities
- **Effort**: ~5 lines
- **Impact**: exact name/tool matching improves relevance

#### 4.3 Smart capture via LLM extraction
- **Before**: `capture.js` stores raw `user: xxx\nassistant: yyy` text
- **After**: LLM extracts key facts/decisions before storing (uses existing `extractor.py`)
- **Effort**: ~30 lines in capture.js
- **Impact**: reduces noise, dramatically improves episodic content quality

#### 4.4 Update access_count on recall
- **Before**: recalled memories don't update access counters
- **After**: recall bumps `access_count += 1` and `accessed_at`
- **Effort**: ~10 lines
- **Impact**: provides data foundation for 4.1 popularity weighting

### P1: Short-term (next iterations)

#### 4.5 Time-range queries
- Support `memory recall --after 2026-03-01 --before 2026-03-05`
- Enables agents to answer "what happened last week" style questions

#### 4.6 Lightweight entity relations
- Build a simple co-occurrence graph from the entities field (no Neo4j needed)
- Support "all memories related to Alice" style association queries

#### 4.7 Memory version chains
- Add `parent_id` field to track UPDATE relationships
- Enable review/audit of how memories evolve over time

### P2: Long-term

#### 4.8 Graph Memory (Neo4j-free)
- Implement a lightweight knowledge graph using chDB or simple adjacency tables
- Support multi-hop queries

#### 4.9 MCP Server mode
- Provide an MCP interface alongside the OpenClaw plugin
- Enable Claude Desktop / Cursor / VS Code integration
