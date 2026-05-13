# MEMORA

**M**emory **E**mbedded with **M**arkdown, **O**bservable **R**etrieval & **A**daptive Forgetting

面向 LLM Agent 的记忆系统。人可读的 `.md` 存储 + 机器读的 `.json` 向量，自带遗忘机制。

## 架构

```
                      用户消息
                         │
                         ▼
                 ShortTermMemory
                  (带时间戳, token 计数)
                         │
                         │ token 超限?
                         │
                    ┌────▼──────┐
                    │ trimmer    │
                    │ LLM 摘要   │
                    └────┬──────┘
                         │
                         ▼
                  LongTermMemory  ──────────────────────┐
                                                        │
                  AI Agent 主动判断                       │
                   │          │                         │
                   ▼          ▼                         │
           KnowledgeLibrary  PersonaLibrary             │
                                                        │
                   ┌────────────────────────────────────┘
                   │
                   ▼
              rebuild() 手动调用
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
  遗忘      fixed 同步     去重合并
```

**三条入库管道**: 自动裁剪 → LTM / Agent Tool → Knowledge、Persona / 用户丢文件到 fixed

**一条维护管道**: 手动 `rebuild()` → 遍历计算优先级 → 低于阈值删除 + 扫描 fixed 新文件入库

## 使用指南

### 配置

```python
from memora import MemoryConfig

cfg = MemoryConfig(
    # 短期记忆裁剪阈值（超过 M 触发裁剪，保留最近 N 的 token）
    stm_max_tokens=8000,       # M: 默认 8000
    stm_trim_tokens=4000,      # N: 默认 4000

    # 检索
    recall_candidates=20,      # 向量搜索召回候选数
    recall_final=5,            # 最终返回记忆数

    # 遗忘阈值 (priority < 此值 → 删除)
    forgetting_threshold=0.15,

    # API 密钥
    embedding_api_key="你的智谱key",
    summary_api_key="你的DeepSeek key",
)
```

通过 `.env` 设置密钥：

```bash
ZHIPU_API_KEY=你的智谱key
DEEPSEEK_API_KEY=你的DeepSeek key
DEEPSEEK_API_URI=https://api.deepseek.com
```

### 短期记忆 (ShortTermMemory)

```python
from memora import ShortTermMemory

stm = ShortTermMemory(cfg)

stm.add("user", "Python 协程是什么？")
stm.add("assistant", "协程是用 async def 定义的函数...")

stm.total_tokens       # 总 token
len(stm)               # 消息条数
stm.needs_trim()       # total_tokens > M?

# 裁剪：保留最近 N 的 token，返回被丢弃的旧消息
discard = stm.trim()
# 或预览
keep, discard = stm.get_trim_candidates()

# 带时间戳导出
print(stm.format_with_time())
```

### 长期记忆 (LongTermMemory) + 裁剪摘要

当 STM token 超限，自动裁剪 → DeepSeek V4 Flash 摘要 → 智谱 embedding-2 向量化 → 存 LTM。

```python
from memora import LongTermMemory, trim_and_summarize

ltm = LongTermMemory(cfg)

# 一键：检测 → 摘要 → 嵌入 → 入库
mem = await trim_and_summarize(stm, ltm,
    llm_call=lambda p: deepseek_summarize_async(p, cfg),
    embed_fn=lambda t: zhipu_embed_async(t, cfg),
)

# 手动写入
mem = ltm.add(
    summary="用户在学习 Python asyncio 的最佳实践。",
    embedding=zhipu_embed("用户在学习 Python asyncio 的最佳实践。", cfg),
)

# 读取
mem = ltm.get(entry_id)
ltm.list_all()             # 按 index 排序
ltm.increment_ref(id)      # 引用 +1
ltm.delete(id)
```

### 知识库 (KnowledgeLibrary)

```python
from memora import KnowledgeLibrary

kb = KnowledgeLibrary(cfg)

kb.add(
    title="Python asyncio.gather 用法",
    content="asyncio.gather(*coros) 并发执行多个协程，返回结果列表。",
    category="tech_solution",
    source="conversation_42",
    embedding=zhipu_embed("asyncio gather 并发协程", cfg),
)

kb.get(id)
kb.list_all()
kb.list_by_category("tech_solution")
kb.list_categories()
kb.update(id, title="新标题")
kb.increment_ref(id)
kb.delete(id)
```

### 人设库 (PersonaLibrary)

```python
from memora import PersonaLibrary

pl = PersonaLibrary(cfg)

# upsert: 存在则合并证据提升置信度，不存在则新建
pl.upsert(
    type="user",
    trait="编程风格",
    description="偏好严格类型注解和 asyncio 异步编程。",
    confidence=0.7,
    evidence=["对话#42 提到 mypy strict"],
)

pl.list_user_traits()
pl.list_ai_persona()
pl.find_by_trait("user", "编程风格")
```

### 统一检索

```python
from memora import MemoryRetrieval, MemoryTools, zhipu_embed

retrieval = MemoryRetrieval(cfg)
query_emb = zhipu_embed("Python 协程", cfg)

results = retrieval.search(query_emb, "all", top_k=5)
# {"ltm": [...], "knowledge": [...], "persona": [...]}

results = retrieval.search_and_record(query_emb, "all", top_k=5)  # +引用计数

context = retrieval.build_context(results)  # LLM prompt 可用
```

### MemoryTools

```python
tools = MemoryTools(cfg)
tools.set_embed_fn(lambda t: zhipu_embed(t, cfg))

tools.add_knowledge("标题", "内容", "分类", "来源")  # → entry_id
tools.update_persona("user", "特征", "描述", 0.7)     # → entry_id
tools.search_memory("查询", "all", top_k=5)           # → 格式化文本
tools.rebuild_library("all")                          # 遗忘 + fixed 同步
tools.stats()                                          # → dict
```

### 整仓维护 rebuild

```python
tools = MemoryTools(cfg)
tools.set_embed_fn(lambda t: zhipu_embed(t, cfg))

tools.rebuild_library("all")
```

`rebuild` 做三件事：

1. **遗忘** — 遍历全部条目，计算 `priority(sim=0.5, ref_count, created_at)`，低于 `forgetting_threshold` 的删
2. **fixed 同步** — 扫描 `mem/fixed/` 下的 `.md`，检测新增/修改 → 摘要 + 嵌入 → 入库
3. **去重合并** — knowledge 同标题保留最新，persona 同 trait 合并证据提高置信度

### 用户手动添加 fixed

把 `.md` 文件放进对应目录，然后 call `rebuild`：

```
mem/fixed/
  ltm/some_memory.md
  knowledge/python_tips.md
  persona/my_style.md
```

## 存储格式

双格式存储，存放在 `mem/` 目录下：

```
mem/
  fixed/                        # 用户手动放置 .md（只读源）
    ltm/
    knowledge/
    persona/
  memory/                       # 系统自动维护
    ltm/
      peo/   # 人可读 .md (Obsidian 可直接打开浏览)
      mac/   # 机器读 .json (向量 + 元数据，检索直接加载)
    knowledge/
      peo/   # # 标题 + 内容 + 脚注
      mac/   # JSON: title, content, embedding, category...
    persona/
      peo/   # # [用户画像] 特征名 + 置信度进度条 + 证据列表
      mac/   # JSON: type, trait, description, embedding...
```

`peo/` 用 Obsidian 打开可直接浏览编辑，`mac/` 供向量搜索直接读 json 免解析 YAML。

## 许可证

MIT
