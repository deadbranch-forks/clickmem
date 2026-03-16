## Learned User Preferences

**Code quality & process**
- Fix root causes for all users; no case-by-case patches, retries, or workarounds; revert speculative changes promptly
- All config via env vars with sensible defaults; never hardcode IPs or machine-specific values
- Repo code, docs, and comments in English; extracted CEO Brain knowledge preserved in the user's original language
- Batch commits by logical grouping; commit and push only when explicitly asked; release only via git tags + CI

**Architecture & design**
- Plan and discuss architecture before coding; deliver plans in current chat mode — don't switch to Plan mode unless asked
- All interfaces (CLI, MCP tools, plugins) must expose the same capabilities as the HTTP API; never access chDB directly
- Use `asyncio.to_thread()` for all blocking calls (chDB, embedding, LLM) inside the async server
- Prefer event-driven hooks over periodic cron; hooks source code in project tree (`cursor-hooks/`), not `.cursor/`
- Distinguish project-scoped facts from global principles; use project_id + score boosting to prevent cross-project knowledge pollution
- AGENTS.md bullets parsed directly as principles (not re-extracted through LLM); CLAUDE.md goes through LLM extraction

**Workflow**
- Deploy changes AND verify end-to-end yourself; don't tell the user to verify
- Test with representative cases before bulk operations; import data newest-first; don't proactively scan beyond what's specified
- Coordinate parallel sessions: if another session implements a feature, revert speculative changes and wait for merge

## Learned Workspace Facts

- ClickMem: local-first CEO Brain for AI coding agents; shared by Claude Code, Cursor, OpenClaw via MCP + REST API
- CEO Brain: projects, decisions (with reasoning), principles (with confidence + evidence), episodes (TTL 180d), raw_transcripts; dedup gate prevents duplicates on insert
- `memory setup` one-click: install service + hooks + discover agents + import history (background); `memory status` tracks progress
- `memory import` reads Claude Code JSONL + Cursor agent-transcripts + CLAUDE.md/AGENTS.md; newest-first; incremental via `~/.clickmem/import-state.json`
- Project-scoped recall: same-project results boosted 1.3x, global 1.0x, other-project penalized 0.6x; prevents cross-project knowledge pollution
- chDB (embedded ClickHouse); `ReplacingMergeTree(updated_at)`; single-process lock; single port 9527 (REST + MCP SSE + hooks)
- Local LLM: Qwen3.5 4-bit on MLX (8GB→2B, 16GB→4B, 32GB→9B); `enable_thinking=False` for structured JSON; embedding Qwen3-Embedding-0.6B (256d, never MPS)
- Chunked extraction: conversations segmented at turn boundaries (max 5 × 4000 chars); dedup merges across segments; startup dedup cleans existing principles
- PyPI `clickmem`; CI on Python 3.10/3.12/3.13; release via `v*` tags + Trusted Publisher
- Config: `CLICKMEM_SERVER_HOST/PORT` (9527), `CLICKMEM_REMOTE`, `CLICKMEM_API_KEY`, `CLICKMEM_DB_PATH`, `CLICKMEM_LLM_MODE/MODEL/LOCAL_MODEL`
- Tests: `MockEmbeddingEngine` + `MockLLMComplete` via conftest; in-memory chDB
- Deploy: Mac Mini M4 32GB (`mini.local`) via Tailscale; launchd service with auto-restart
