## Learned User Preferences

**Code quality & process**
- Product-first: fix code and dependencies for all users; never apply case-by-case machine patches
- Fix root causes, not symptoms; don't paper over architectural problems with retries or workarounds
- Avoid unnecessary complexity; revert speculative changes promptly
- All config via env vars with sensible defaults; never hardcode IPs or machine-specific values
- All repo content (docs, comments, code) in English; no Chinese
- Batch commits by logical grouping; commit and push only when explicitly asked
- Release only via `git tag v0.x.x && git push origin v0.x.x` (GitHub Actions pipeline); never manual `twine upload`

**Architecture & design**
- Plan and discuss architecture before coding; deliver plans in current chat mode — don't switch to Plan mode unless asked
- All interfaces (CLI, MCP tools, plugins) must expose the same capabilities as the HTTP API; never access chDB directly or leave endpoints HTTP-only
- Use `asyncio.to_thread()` for all blocking calls (chDB, embedding, LLM) inside the async server
- Prefer event-driven hooks (session boundaries) over periodic cron to avoid idle token waste
- Hooks source code lives in project source tree (`cursor-hooks/`), not under `.cursor/`
- Keep docs user-focused; omit internal implementation details that don't help users

**Workflow**
- Deploy changes AND verify end-to-end yourself; don't tell the user to verify
- Stay focused on the current task; don't get sidetracked
- Coordinate parallel sessions: if another session implements a feature, revert speculative changes and wait for merge

## Learned Workspace Facts

**Product**
- ClickMem: local-first memory system for AI coding agents; shared by Claude Code, Cursor, OpenClaw via MCP + REST API
- Memory layers: Raw (separate append-only table) → L1 Episodic (`raw_id` links to `raw_transcripts`) → L2 Semantic (most refined, at top)

**Storage**
- chDB (embedded ClickHouse); single-process lock per data dir; all access through one server process
- `ReplacingMergeTree(updated_at)` engine; all SELECTs use `FINAL`; updates via INSERT; no ALTER mutations in hot paths

**Server**
- Single port 9527: REST `/v1/*` + MCP SSE `/sse` + MCP stdio (`clickmem-mcp`) for local
- LAN discovery via mDNS `_clickmem._tcp`; Bearer auth via `CLICKMEM_API_KEY`
- Read-only SQL (`SELECT`/`SHOW`/`DESCRIBE`) allowed without `--debug`; write queries require `--debug`

**LLM & embedding**
- Local LLM auto-selects by GPU memory: Apple Silicon MLX (8GB→2B, 16GB→4B, 32GB→9B), CUDA (4GB→2B, 8GB→4B, 16GB→9B); CPU-only falls back to remote API; override via `CLICKMEM_LOCAL_MODEL`
- Embedding: Qwen3-Embedding-0.6B (256 dims) on CPU or CUDA; **never** PyTorch MPS (deadlocks via GCD `dispatch_sync` in asyncio worker threads)

**Retrieval**
- Hybrid vector + keyword search with `since`/`until` time filtering; MMR dedup (threshold 0.92); semantic boost 1.3x, refinement boost 1.15x
- `clickmem_list` MCP tool for chronological browsing with layer/category/time filters and pagination
- `access_count` popularity boost (log scale); `entities` field participates in keyword matching

**Distribution & CI**
- PyPI package: `clickmem`; `pip install clickmem` includes all deps (mlx-lm on macOS ARM64); only `litellm` is optional (`pip install 'clickmem[llm]'`)
- CI: `ci.yml` tests on push/PR (Python 3.10/3.12/3.13); `release.yml` publishes on `v*` tags via Trusted Publisher

**Config env vars**
- `CLICKMEM_SERVER_HOST`, `CLICKMEM_SERVER_PORT` (default 9527)
- `CLICKMEM_REMOTE` — remote server URL; makes CLI/MCP act as pure client
- `CLICKMEM_API_KEY` — Bearer token auth
- `CLICKMEM_DB_PATH` (default `~/.openclaw/memory/chdb-data`)
- `CLICKMEM_LLM_MODE` (`auto`|`local`|`remote`), `CLICKMEM_LLM_MODEL`, `CLICKMEM_LOCAL_MODEL`
- `CLICKMEM_REFINE_THRESHOLD` (default 1), `CLICKMEM_LOG_LEVEL`

**Testing**
- Mocks: `MockEmbeddingEngine` + `MockLLMComplete` via conftest fixtures; `CLICKMEM_LOCAL_MODEL` pinned to 2B; all tests use in-memory chDB

**Deploy**
- Target: Mac Mini M4 32GB (`mini.local`) via Tailscale; rsync to `~/clickmem`; launchd service with auto-restart
