# MEMORA

**M**emory **E**mbedded with **M**arkdown, **O**bservable **R**etrieval & **A**daptive Forgetting

面向 LLM Agent 的 wiki 式记忆系统。对话历史、知识经验和用户画像都以人可读的 `.md` 文件存储——可以直接用 Obsidian 打开浏览编辑。

## 架构

```
用户消息 → ShortTermMemory (带时间戳)
              │
              │ token 超限?
              │
          ┌───▼──────────┐
          │ trimmer.py    │
          │ LLM 生成摘要   │
          └───┬──────────┘
              │
              ▼
       LongTermMemory (llmwiki #1)

       AI Agent 主动判断:
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
KnowledgeLibrary    PersonaLibrary
(llmwiki #2)        (llmwiki #3)

统一检索: 向量搜索 + sigmoid 优先级排序
```

## 使用指南

### 配置

```python
from memora import MemoryConfig

cfg = MemoryConfig(
    # 短期记忆裁剪阈值（超过 M 触发裁剪，保留最近 N 的 token）
    stm_max_tokens=8000,       # M: 默认 8000
    stm_trim_tokens=4000,      # N: 默认 4000

    # 检索参数
    recall_candidates=20,      # 向量搜索召回候选数
    recall_final=5,            # 最终返回的记忆数

    # API 密钥
    embedding_api_key="你的智谱key",
    embedding_model="embedding-2",     # 智谱 embedding-2，1024 维
    summary_api_key="你的DeepSeek key",
    summary_model="deepseek-v4-flash",
)
```

通过环境变量设置密钥：

```bash
ZHIPU_API_KEY=你的智谱key
DEEPSEEK_API_KEY=你的DeepSeek key
DEEPSEEK_API_URI=https://api.deepseek.com
```

```python
from dotenv import load_dotenv; load_dotenv()
cfg = MemoryConfig(
    embedding_api_key=os.environ["ZHIPU_API_KEY"],
    summary_api_key=os.environ["DEEPSEEK_API_KEY"],
)
```

### 短期记忆 (ShortTermMemory)

对话窗口，带 token 计数和时间戳。超限时自动裁剪。

```python
from memora import ShortTermMemory

stm = ShortTermMemory(cfg)

# 添加消息，自动估算 token 数
stm.add("user", "你好，我想了解一下 Python 异步编程。")
stm.add("assistant", "Python 的异步编程基于 asyncio...")

# 查看当前状态
print(stm.total_tokens)    # 总 token 数
print(len(stm))            # 消息条数
print(stm.needs_trim())    # 是否超过 M 阈值，需要裁剪

# 带时间戳导出对话
print(stm.format_with_time())
# [2026-05-12 14:30:00] user: 你好，我想了解一下 Python 异步编程。
# [2026-05-12 14:30:05] assistant: Python 的异步编程基于 asyncio...

# 裁剪：保留最近 N 的 token，返回被丢弃的旧消息
discarded = stm.trim()          # stm.messages 变为保留部分
# 或先预览再决定
keep, discard = stm.get_trim_candidates()

# 清空
stm.clear()
```

### 长期记忆 (LongTermMemory) + 裁剪摘要

当 STM token 超限，自动把裁剪掉的旧消息送给 DeepSeek V4 Flash 生成摘要，然后用智谱 embedding-2 向量化，存入 LTM。

```python
from memora import LongTermMemory, trim_and_summarize
from memora import zhipu_embed_async, deepseek_summarize_async

ltm = LongTermMemory(cfg)

# 一键流程：检测裁剪 → 摘要 → 嵌入 → 存入 LTM
mem = await trim_and_summarize(
    stm, ltm,
    llm_call=lambda p: deepseek_summarize_async(p, cfg),
    embed_fn=lambda t: zhipu_embed_async(t, cfg),
)
# 如果没超限，返回 None；超限则返回新创建的 LTMMemory 对象

# 手动写入 LTM
from memora import zhipu_embed
mem = ltm.add(
    summary="用户在学习 Python asyncio 的最佳实践。",
    original_tokens=200,
    embedding=zhipu_embed("用户在学习 Python asyncio 的最佳实践。", cfg),
)

# 读取
mem = ltm.get(entry_id)
all_mems = ltm.list_all()          # 按 index 排序

# 引用计数
ltm.increment_ref(entry_id)        # 被检索命中时 +1

# 删除
ltm.delete(entry_id)
```

### 知识库 (KnowledgeLibrary)

AI Agent 主动从对话中提取可复用知识，存入带分类标签的 wiki。

```python
from memora import KnowledgeLibrary

kb = KnowledgeLibrary(cfg)

# 写入知识条目
kid = kb.add(
    title="Python asyncio.gather 用法",
    content="asyncio.gather(*coros) 并发执行多个协程，返回结果列表。"
            "遇到异常默认取消所有协程，设 return_exceptions=True 可改为收集异常。",
    category="tech_solution",       # 分类标签
    source="conversation_42",       # 来源对话 ID
    embedding=zhipu_embed("asyncio gather 并发协程", cfg),
)

# 查询
entry = kb.get(kid)
all_entries = kb.list_all()
tech_entries = kb.list_by_category("tech_solution")
cats = kb.list_categories()        # 所有分类

# 更新
kb.update(kid, title="新标题", category="new_cat")
kb.increment_ref(kid)              # 命中 +1
kb.delete(kid)
```

### 人设库 (PersonaLibrary)

追踪用户画像和 AI 人设，带置信度评分和证据链。

```python
from memora import PersonaLibrary

pl = PersonaLibrary(cfg)

# 写入（推荐用 upsert：存在则合并证据提高置信度，不存在则新建）
pid = pl.upsert(
    type="user",                    # "user" | "ai_persona"
    trait="编程风格",
    description="偏好严格类型注解和 asyncio 异步编程。",
    confidence=0.7,
    evidence=["对话#42 提到 mypy strict"],
)

# 再次 upsert 同一 trait：置信度自动提升，证据自动合并
pl.upsert(
    type="user", trait="编程风格",
    description="偏好严格类型注解、asyncio 和函数式编程。",
    confidence=0.5,
    evidence=["对话#50 使用 map/filter"],
)  # confidence 现在是 0.75，evidence 合并为两条

# 查询
pl.list_user_traits()              # 所有用户特征
pl.list_ai_persona()               # 所有 AI 人设
pl.find_by_trait("user", "编程风格")

# 重构：去重合并
pl.rebuild()
```

### 统一检索

跨 LTM、知识库、人设库做语义搜索，返回排序后的结果。

```python
from memora import MemoryRetrieval, MemoryTools, zhipu_embed

retrieval = MemoryRetrieval(cfg)

query_emb = zhipu_embed("Python 异步协程", cfg)

# 检索（不更新引用计数）
results = retrieval.search(query_emb, memory_type="all", top_k=5)
# {"ltm": [...], "knowledge": [...], "persona": [...]}

# 检索并自动 +1 引用计数
results = retrieval.search_and_record(query_emb, "all", top_k=5)

# 组装成 LLM prompt 可用的上下文字符串
context = retrieval.build_context(results)

# 或用 MemoryTools 一行搞定
tools = MemoryTools(cfg)
tools.set_embed_fn(lambda t: zhipu_embed(t, cfg))
print(tools.search_memory("Python 协程", "all", top_k=5))
```

### 一键 Agent 工具

```python
from memora import MemoryTools

tools = MemoryTools(cfg)
tools.set_embed_fn(lambda t: zhipu_embed(t, cfg))

# 四个 Agent 可调用的工具
tools.add_knowledge("标题", "内容", "分类", "来源")   # → entry_id
tools.update_persona("user", "特征", "描述", 0.7)      # → entry_id
tools.search_memory("查询", "all", top_k=5)            # → 格式化文本
tools.rebuild_library("all")                            # → 统计信息
tools.stats()                                            # → dict 诊断
```

## Wiki 存储格式

所有记忆都是 `.md` 文件——用 Obsidian 打开 `wiki/` 目录即可：

**长期记忆** (`wiki/ltm/`): `# 长期记忆 #1` 标题 + 摘要正文 + 创建时间/引用/原始 token 脚注

**知识条目** (`wiki/knowledge/`): `# 标题` + 内容 + 分类/来源/创建时间/引用脚注

**人设/用户画像** (`wiki/persona/`): `# [用户画像] 特征名` + 描述 + 置信度进度条 + 证据列表 + 更新时间

## 存储目录

```
wiki/
  ltm/         # 长期记忆摘要
  knowledge/   # 知识经验条目
  persona/     # 用户画像 + AI 人设
```

## 许可证

MIT
