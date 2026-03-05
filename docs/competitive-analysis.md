# ClickMem 竞品分析与优化计划

## 一、竞品概览

| 维度 | **Mem0** | **Supermemory** | **MemOS** | **ClickMem (当前)** |
|------|---------|----------------|-----------|-------------------|
| 定位 | 通用AI记忆层 | Agent记忆引擎 | LLM记忆操作系统 | 本地AI编码助手记忆 |
| 部署 | Cloud + Self-host | Cloud (MCP) | Self-host | 纯本地 |
| 存储 | Vector DB + Graph DB | Cloudflare DO | SQLite/多后端 | chDB (嵌入式ClickHouse) |
| 嵌入 | Remote API | Remote API | Local/Remote | **本地 Qwen3** (优势) |
| 隐私 | 需API Key | 云端 | 可本地 | **完全本地** (优势) |
| 许可 | Apache 2.0 | Proprietary | Apache 2.0 | (待定) |

## 二、核心能力对比

### 1. 记忆类型

| 记忆类型 | Mem0 | Supermemory | MemOS | ClickMem |
|---------|------|-------------|-------|----------|
| 语义/长期 | ✅ vector + graph | ✅ knowledge chains | ✅ textual memory | ✅ L2 semantic |
| 事件/短期 | ✅ episodic | ✅ session-based | ✅ activation memory | ✅ L1 episodic |
| 工作记忆 | ✅ context window | ✅ working memory | ✅ KV cache | ✅ L0 working |
| **图谱记忆** | ✅ Neo4j/Memgraph | ✅ 关系链 | ❌ | ❌ **缺失** |
| **参数记忆** | ❌ | ❌ | ✅ LoRA weights | ❌ (不适用) |

### 2. 写入/更新策略

| 能力 | Mem0 | Supermemory | ClickMem |
|-----|------|-------------|----------|
| 提取 → 去重管道 | ✅ 2-phase (Extract → Update) | ✅ chunk + contextual | ✅ upsert (search → LLM judge) |
| 矛盾检测 | ✅ LLM 4操作 (ADD/UPDATE/DELETE/NOOP) | ✅ isLatest 字段 | ✅ 同 Mem0 模式 |
| **关系版本控制** | ✅ graph edges | ✅ Updates/Extends/Derives 三种关系 | ❌ **缺失** |
| **实体抽取** | ✅ LLM → 图节点/边 | ✅ 隐含于知识链 | ⚠️ entities 字段存在但未用于检索 |
| 批量操作 | ✅ batch up to 1000 | ✅ | ❌ |

### 3. 检索策略

| 能力 | Mem0 | Supermemory | ClickMem |
|-----|------|-------------|----------|
| 向量相似度 | ✅ | ✅ | ✅ |
| 关键词匹配 | ✅ | ✅ | ✅ |
| 时间衰减 | ❌ (无明确) | ✅ smart forgetting | ✅ **差异化衰减** (优势) |
| MMR 去重 | ❌ | ❌ | ✅ **MMR** (优势) |
| **图遍历** | ✅ multi-hop reasoning | ✅ 关系链追踪 | ❌ **缺失** |
| **时间推理** | ⚠️ 基础 | ✅ temporal grounding | ❌ **缺失** |
| access_count 加权 | ❌ | ✅ 频次加权 | ⚠️ 字段存在但未用于评分 |

### 4. 自维护

| 能力 | Mem0 | Supermemory | ClickMem |
|-----|------|-------------|----------|
| 过期清理 | ✅ | ✅ smart forgetting | ✅ cleanup_stale |
| 压缩/摘要 | ❌ (依赖外部) | ✅ context rewriting | ✅ compress_episodic |
| 模式晋升 | ❌ | ✅ hierarchical promotion | ✅ promote_to_semantic |
| 语义审查 | ❌ | ❌ | ✅ review_semantic |
| **事件驱动维护** | ❌ | ❌ | ✅ session boundary hook |

## 三、ClickMem 优劣势总结

### 优势 (Keep)
1. **完全本地运行** — 零API费用、零数据泄露，是唯一不需要云端的方案
2. **本地嵌入模型** — Qwen3-Embedding-0.6B 无需远程API
3. **差异化时间衰减** — L1指数衰减 + L2对数衰减，比竞品更精细
4. **MMR去重** — 检索结果多样性，竞品均无
5. **事件驱动维护** — session boundary触发，比定时cron更高效
6. **chDB嵌入式** — 无需独立数据库进程

### 劣势/差距 (Gap)
1. **无实体关系图** — Mem0有Neo4j图谱，Supermemory有知识链，我们的entities字段未利用
2. **无关系版本控制** — 记忆之间没有update/extends/derives关系追踪
3. **access_count未用于评分** — 字段存在但检索时被忽略
4. **无时间推理** — 无法回答"上周做了什么"这类时间范围查询
5. **capture过于简单** — 直接存原始对话文本，未做LLM提取摘要
6. **实体未用于检索** — entities字段只存不用

## 四、优化计划

基于投入产出比排序，优先做"低成本高收益"的改进：

### P0: 立即执行 (本次)

#### 4.1 access_count 参与检索评分
- **现状**: `access_count` 字段在DB中存在，但 `retrieval.py` 完全忽略
- **改进**: 检索时将 access_count 作为加权因子，常被访问的记忆分数更高
- **工作量**: ~20行代码
- **收益**: 高频记忆优先返回，直接提升召回质量

#### 4.2 entities 参与关键词匹配
- **现状**: `entities` 字段存储了但检索时不参与匹配
- **改进**: `_keyword_score` 中加入 entities 匹配
- **工作量**: ~5行代码
- **收益**: 人名/工具名精确匹配，提升相关性

#### 4.3 capture 智能摘要替代原始文本
- **现状**: `capture.js` 直接存 `user: xxx\nassistant: yyy` 原始文本
- **改进**: 用 LLM 提取关键事实/决定再存储（利用已有的 `extractor.py`）
- **工作量**: ~30行代码改 capture.js
- **收益**: 减少噪音，episodic内容质量大幅提升

#### 4.4 recall 时更新 access_count 和 accessed_at
- **现状**: 检索到的记忆没有更新访问计数
- **改进**: recall 后更新 access_count += 1 和 accessed_at
- **工作量**: ~10行代码
- **收益**: 为 4.1 的 access_count 加权提供数据基础

### P1: 短期 (后续迭代)

#### 4.5 时间范围查询
- 支持 `memory recall --after 2026-03-01 --before 2026-03-05` 时间过滤
- 让agent能回答"上周做了什么"之类的问题

#### 4.6 轻量实体关系
- 从 entities 字段构建简单的共现图（不需要Neo4j）
- 支持"和 Alice 相关的所有记忆"这类关联查询

#### 4.7 记忆版本链
- 新增 `parent_id` 字段追踪 UPDATE 关系
- 让 review/audit 能看到记忆演变历史

### P2: 中长期

#### 4.8 Graph Memory (Neo4j-free)
- 用 chDB 内置的 graph 能力或简单邻接表实现轻量知识图谱
- 支持 multi-hop 查询

#### 4.9 MCP Server 模式
- 除了 OpenClaw 插件，额外提供 MCP 接口
- 让 Claude Desktop / Cursor / VS Code 也能使用
