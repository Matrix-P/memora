# MEMORA 技术文档

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户消息                                   │
└─────────┬───────────────────────────────────────────────────────┘
          │
          ▼
   ShortTermMemory (带时间戳, token 计数)
          │
          │ token 超限?
          │
    ┌─────▼──────────┐
    │ trimmer.py      │
    │ LLM 生成摘要     │
    └─────┬──────────┘
          │
          ▼
   LongTermMemory ─────────────────────┐
                                       │
   AI Agent 主动判断 (对话中实时触发)      │
    │                    │             │
    ▼                    ▼             │
  KnowledgeLibrary  PersonaLibrary     │
                                       │
    ┌──────────────────────────────────┘
    │
    ▼
   整仓维护 (rebuild)
    ├── 用户 fixed/ 扫描, 检测新增/修改 → 摘要 + 嵌入 → 入库
    ├── 计算所有条目的优先级
    └── 删除优先级低于阈值的条目 (遗忘)
```

三条入库管道:

| 管道 | 触发方式 | 目标 |
|------|----------|------|
| STM → 裁剪 → 摘要 | 自动 (token 超限) | LTM |
| Agent Tool 主动写入 | 对话中实时判断 | Knowledge / Persona |
| 用户丢入 fixed/ | 手动 rebuild | LTM / Knowledge / Persona |

一条出库管道:

- **rebuild 遗忘**: 计算所有条目优先级, 低于阈值删除

## 2. 存储布局

```
mem/
├── fixed/                  # 用户手动放置的 markdown (只读源)
│   ├── ltm/                #   长期记忆 .md
│   ├── knowledge/          #   知识条目 .md
│   └── persona/            #   人设 .md
│
└── memory/                 # 系统自动维护
    ├── ltm/
    │   ├── peo/            # 人可读 .md (Obsidian 可直接打开)
    │   │   └── ltm_20260512_161149.md
    │   └── mac/            # 机器读 .json (向量 + 元数据)
    │       └── ltm_20260512_161149.json
    ├── knowledge/
    │   ├── peo/
    │   │   └── k_20260512_161632.md
    │   └── mac/
    │       └── k_20260512_161632.json
    └── persona/
        ├── peo/
        │   └── p_20260512_161632.md
        └── mac/
            └── p_20260512_161632.json
```

- `fixed/` — 人把 markdown 丢在这里, rebuild 时系统检测新增/修改, 摘要 + 嵌入后写入 `memory/`
- `memory/peo/` — 人可读的完整格式 .md (标题层级、置信度进度条、证据列表、脚注), Obsidian 直接打开
- `memory/mac/` — 机器直接读的 .json, 包含向量和全部元数据, 跳过 YAML 解析

### mac/ JSON 格式

**LTM** (`memory/ltm/mac/{id}.json`):

```json
{
  "id": "ltm_20260512_161149",
  "index": 1,
  "summary": "用户询问了 Python 异步编程的核心概念...",
  "embedding": [0.01, 0.02, ...],
  "ref_count": 3,
  "created_at": 1778400000.0,
  "time_start": 1778390000.0,
  "time_end": 1778399000.0
}
```

**Knowledge** (`memory/knowledge/mac/{id}.json`):

```json
{
  "id": "k_20260512_161632",
  "title": "Python asyncio.gather 用法",
  "content": "asyncio.gather 并发执行多个协程...",
  "category": "tech_solution",
  "source": "",
  "embedding": [0.01, 0.02, ...],
  "ref_count": 1,
  "created_at": 1778403392.0,
  "updated_at": 1778403392.0
}
```

**Persona** (`memory/persona/mac/{id}.json`):

```json
{
  "id": "p_20260512_161632",
  "type": "user",
  "trait": "编程风格",
  "description": "偏好严格类型注解和 asyncio 异步编程...",
  "confidence": 0.85,
  "evidence": ["对话#42 提到 mypy strict"],
  "embedding": [0.01, 0.02, ...],
  "ref_count": 0,
  "created_at": 1778403392.0,
  "updated_at": 1778403392.0
}
```

### peo/ .md 格式

与现有格式一致 (YAML frontmatter + 结构化 Markdown), 详见 `mem/memory/{库}/peo/` 下的实际文件。

## 3. 核心模块

### 3.1 MemoryConfig — 配置中心

```python
from memora import MemoryConfig

cfg = MemoryConfig(
    # 短期记忆裁剪
    stm_max_tokens=8000,       # M: 触发裁剪阈值
    stm_trim_tokens=4000,      # N: 裁剪后保留 token 数

    # 检索
    recall_candidates=20,      # 向量搜索召回候选数
    recall_final=5,            # 最终返回记忆数

    # 优先级权重
    weight_similarity=0.5,     # 向量相似度
    weight_ref_count=0.2,      # 引用计数
    weight_recency=0.3,        # 时间近度

    # Sigmoid 温度
    tau_ref=5.0,               # 引用计数
    tau_time=86400.0,          # 时间衰减 (秒, ≈1天)

    # 引用计数基数
    ref_base=1,

    # 遗忘阈值 (priority < 此值 → 删除)
    forgetting_threshold=0.15,

    # 嵌入模型 (智谱)
    embedding_model="embedding-2",
    embedding_dim=1024,
    embedding_api_key="",
    embedding_api_base="https://open.bigmodel.cn/api/paas/v4/embeddings",

    # 摘要模型 (DeepSeek)
    summary_model="deepseek-v4-flash",
    summary_api_key="",
    summary_api_base="https://api.deepseek.com/v1/chat/completions",

    # 存储路径
    mem_dir="mem",
)
```

### 3.2 ShortTermMemory — 短期记忆

对话窗口, 每条消息带 `role`、`content`、`timestamp`、`token_count`。

```python
stm = ShortTermMemory(cfg)

# 添加消息 (自动估算 token)
stm.add("user", "文本")
stm.add("assistant", "文本")

# 查询
stm.total_tokens       # 当前总 token
len(stm)               # 消息条数
stm.needs_trim()       # total_tokens > M?

# 裁剪
keep, discard = stm.get_trim_candidates()   # 预览
discard = stm.trim()                        # 执行

# 导出
stm.format_with_time()   # 带时间戳格式化
stm.to_list()             # list[dict]
stm.clear()               # 清空
```

### 3.3 LongTermMemory — 长期记忆

摘要存入 `mem/memory/ltm/` 的 peo/ 和 mac/ 双目录。

```python
ltm = LongTermMemory(cfg)

# 写入 (同时写 peo/.md 和 mac/.json)
mem = ltm.add(
    summary="用户在学习 Python asyncio...",
    time_start=1715200000.0,
    time_end=1715200300.0,
    embedding=[0.01, 0.02, ...],
)

# 读取
mem = ltm.get(id)                   # LTMMemory | None
all = ltm.list_all()                # list[LTMMemory]

# 更新
ltm.increment_ref(id)               # 引用 +1
ltm.update_embedding(id, new_emb)   # 更新向量
ltm.delete(id)

# 向量搜索 (遍历 mac/ 目录)
candidates = ltm.search_by_vector(query_emb, top_k=20)
```

### 3.4 KnowledgeLibrary — 知识/经验库

Agent 主动从对话提取的可复用知识, 存入 `mem/memory/knowledge/peo/` + `mac/`。

```python
kb = KnowledgeLibrary(cfg)

e = kb.add(
    title="Python asyncio.gather 用法",
    content="asyncio.gather(*coros) 并发执行多个协程...",
    category="tech_solution",
    source="conversation_42",
    embedding=[...],
)

kb.get(id)
kb.list_all()
kb.list_by_category("tech_solution")
kb.list_categories()
kb.update(id, title="新标题", category="new_cat")
kb.increment_ref(id)
kb.delete(id)
kb.search_by_vector(query_emb, top_k=20)
```

### 3.5 PersonaLibrary — 人设/用户库

追踪用户画像和 AI 人设, 带置信度评分和证据链, 存入 `mem/memory/persona/peo/` + `mac/`。

```python
pl = PersonaLibrary(cfg)

# 写入 (推荐 upsert: 同 trait 自动合并证据、提升置信度)
e = pl.upsert(
    type="user",
    trait="编程风格",
    description="偏好严格类型注解和 asyncio 异步编程。",
    confidence=0.7,
    evidence=["对话#42 提到 mypy strict"],
)

pl.list_user_traits()
pl.list_ai_persona()
pl.find_by_trait("user", "编程风格")
pl.rebuild()       # 去重合并
pl.delete(id)
pl.search_by_vector(query_emb, top_k=20)
```

### 3.6 MemoryRetrieval — 统一检索

跨 LTM、知识库、人设库向量搜索, sigmoid 优先级排序。

```python
retrieval = MemoryRetrieval(cfg)

# 检索
results = retrieval.search(query_emb, memory_type="all", top_k=5)
# {"ltm": [...], "knowledge": [...], "persona": [...]}

# 检索 + 引用计数 +1
results = retrieval.search_and_record(query_emb, "all", top_k=5)

# 组装 LLM prompt 上下文
context = retrieval.build_context(results)
```

### 3.7 trimmer — 裁剪与摘要

当 STM 超过 M 时, 裁掉旧消息 → DeepSeek 生成摘要 → 智谱嵌入 → 写入 LTM。

```python
from memora import trim_and_summarize, naive_llm_trim

# 完整流程
mem = await trim_and_summarize(
    stm=stm,
    ltm=ltm,
    llm_call=deepseek_summarize_async,
    embed_fn=zhipu_embed_async,
)

# 降级: 不摘要, 直接丢弃
discard = naive_llm_trim(stm)
```

### 3.8 MemoryTools — Agent 工具集

注入嵌入函数后, 提供四个 Agent 可直接调用的工具。

```python
tools = MemoryTools(cfg)
tools.set_embed_fn(lambda text: zhipu_embed(text, cfg))

tools.add_knowledge(title, content, category, source)
tools.update_persona(type, trait, description, confidence, evidence)
tools.search_memory(query, memory_type, top_k)   # → 格式化文本
tools.rebuild_library(library)                    # → 统计信息
tools.stats()                                     # → dict 诊断
```

### 3.9 priority — 优先级计算

```python
from memora import sigmoid, compute_priority, rank_memories

priority = compute_priority(
    similarity=cosine_sim(query_emb, memory_emb),
    ref_count=mem.ref_count,
    created_at=mem.created_at,
    now=time.time(),
    config=cfg,
)
# priority = sim·w_sim + σ(ref/tau_ref)·w_ref + (1-σ(age/tau_time))·w_time

# 批量排序
ranked = rank_memories(query_emb, candidates, now, cfg)
```

## 4. 整仓维护 (rebuild)

手动调用的 `rebuild()` 做两件事: **遗忘** + **fixed 入库**。

### 4.1 遗忘

遍历 `memory/` 下所有三个库的全部条目, 计算优先级, 低于 `forgetting_threshold` 的从 peo/ 和 mac/ 同时删除。

```python
def _forget(cfg):
    for library in [ltm, knowledge, persona]:
        for entry in library.list_all():
            priority = compute_priority(
                similarity=0.5,  # 遗忘检查用中性相似度
                ref_count=entry.ref_count,
                created_at=entry.created_at,
                now=time.time(),
                config=cfg,
            )
            if priority < cfg.forgetting_threshold:
                library.delete(entry.id)  # 同时删 peo/.md 和 mac/.json
```

遗忘由两个因素驱动:

| 因素 | 效果 |
|------|------|
| **引用计数低** (σ(ref/tau_ref)) | 从没被检索命中的记忆优先遗忘 |
| **时间久远** (1-σ(age/tau_time)) | 越旧的记忆优先级越低 |

### 4.2 fixed/ 入库

扫描 `mem/fixed/{库}/` 下的 `.md` 文件, 检测新增或修改 (对内容做 hash), 变化则:

1. 用 DeepSeek V4 Flash 生成摘要 (LTM) 或直接读取内容 (Knowledge/Persona)
2. 用智谱 embedding-2 生成向量
3. 写入 `memory/{库}/peo/` 和 `memory/{库}/mac/`

```python
def _sync_fixed(cfg):
    for library_name in ["ltm", "knowledge", "persona"]:
        fixed_dir = f"mem/fixed/{library_name}"
        for md_file in scan_md_files(fixed_dir):
            content_hash = sha256(md_file.content)
            if content_hash == last_hash(md_file.name):
                continue  # 未变化, 跳过

            # 摘要 / 嵌入 / 入库
            summary = deepseek_summarize(f"总结以下内容:\n{md_file.content}", cfg)
            embedding = zhipu_embed(summary, cfg)
            library.add(summary=summary, embedding=embedding, ...)

            save_hash(md_file.name, content_hash)
```

### 4.3 调用方式

```python
tools = MemoryTools(cfg)

# 全部维护
tools.rebuild_library("all")

# 只维护指定库
tools.rebuild_library("knowledge")
tools.rebuild_library("persona")
tools.rebuild_library("ltm")
```

## 5. API 客户端

### 5.1 智谱嵌入

```python
from memora import zhipu_embed, zhipu_embed_async, zhipu_embed_batch

vec = zhipu_embed("文本", cfg)                          # 同步 → list[float]
vec = await zhipu_embed_async("文本", cfg)               # 异步
vecs = [zhipu_embed(t, cfg) for t in ["文本1", "文本2"]] # 批量
```

### 5.2 DeepSeek 摘要

```python
from memora import deepseek_summarize, deepseek_summarize_async

summary = deepseek_summarize("请总结:\n...", cfg)              # 同步 → str
summary = await deepseek_summarize_async("请总结:\n...", cfg)   # 异步
```

## 6. 快速开始

### 6.1 环境

```bash
# .env
ZHIPU_API_KEY=你的智谱key
DEEPSEEK_API_KEY=你的DeepSeek key
DEEPSEEK_API_URI=https://api.deepseek.com
```

### 6.2 最小示例

```python
from dotenv import load_dotenv; load_dotenv()
from memora import (
    ShortTermMemory, LongTermMemory, MemoryConfig,
    MemoryRetrieval, MemoryTools, trim_and_summarize,
    zhipu_embed, zhipu_embed_async, deepseek_summarize_async,
)

cfg = MemoryConfig(
    embedding_api_key=os.environ["ZHIPU_API_KEY"],
    summary_api_key=os.environ["DEEPSEEK_API_KEY"],
)

stm, ltm = ShortTermMemory(cfg), LongTermMemory(cfg)
tools = MemoryTools(cfg)
tools.set_embed_fn(lambda t: zhipu_embed(t, cfg))

# 对话循环
stm.add("user", "你好")
stm.add("assistant", "你好！有什么可以帮你？")

# 裁剪摘要嵌入一气呵成
await trim_and_summarize(
    stm, ltm,
    llm_call=lambda p: deepseek_summarize_async(p, cfg),
    embed_fn=lambda t: zhipu_embed_async(t, cfg),
)

# 检索
emb = zhipu_embed("Python 异步", cfg)
results = MemoryRetrieval(cfg).search_and_record(emb, "all", 5)
print(MemoryRetrieval(cfg).build_context(results))

# 主动提取知识
tools.add_knowledge("标题", "内容", "分类")

# 手动整仓维护
tools.rebuild_library("all")
```

### 6.3 用户手动添加记忆

把你的 `.md` 文件丢进 `mem/fixed/` 对应目录下:

```
mem/fixed/ltm/some_memory.md
mem/fixed/knowledge/python_tips.md
mem/fixed/persona/my_style.md
```

然后调用 `tools.rebuild_library("all")`, 系统会自动检测并入库。
