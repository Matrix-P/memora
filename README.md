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

## 特性

| 特性 | 说明 |
|------|------|
| **短期记忆** | 对话窗口 + token 计数，超限自动触发裁剪 |
| **长期记忆** | LLM 摘要存入结构化 markdown，带序号索引，可搜索 |
| **知识库** | Agent 主动从对话提取可复用知识，分类标签管理 |
| **人设库** | 用户画像 + AI 人设特征，置信度评分，证据链追溯 |
| **向量检索** | 智谱 `embedding-2` (1024d) 语义搜索 |
| **优先级排序** | 相似度 + 引用次数 + 时间近度，sigmoid 平滑加权 |
| **自适应遗忘** | 时间衰减 + 引用衰减，久置不用自动淡出 |
| **Wiki 格式** | `.md` + YAML frontmatter，原生兼容 Obsidian，支持 `tags` |
| **摘要生成** | DeepSeek V4 Flash 压缩对话历史 |

## 安装

```bash
pip install memora
# 或
pip install git+https://github.com/yourname/memora.git
```

## 快速开始

项目根目录创建 `.env`：

```bash
ZHIPU_API_KEY=你的智谱key
DEEPSEEK_API_KEY=你的DeepSeek key
DEEPSEEK_API_URI=https://api.deepseek.com
```

```python
import os
from dotenv import load_dotenv
load_dotenv()

from memora import (
    ShortTermMemory, LongTermMemory, MemoryConfig,
    MemoryTools, MemoryRetrieval, trim_and_summarize,
    zhipu_embed, zhipu_embed_async, deepseek_summarize_async,
)

cfg = MemoryConfig(
    embedding_api_key=os.environ["ZHIPU_API_KEY"],
    summary_api_key=os.environ["DEEPSEEK_API_KEY"],
)

stm = ShortTermMemory(cfg)
ltm = LongTermMemory(cfg)
tools = MemoryTools(cfg)
tools.set_embed_fn(lambda text: zhipu_embed(text, cfg))

# 向短期记忆添加消息
stm.add("user", "Python 协程是什么？")
stm.add("assistant", "协程是用 async def 定义的函数...")

# token 超限时自动裁剪 → DeepSeek 摘要 → 智谱嵌入 → 存入 LTM
await trim_and_summarize(
    stm, ltm,
    llm_call=lambda p: deepseek_summarize_async(p, cfg),
    embed_fn=lambda t: zhipu_embed_async(t, cfg),
)

# AI Agent 主动提取知识
tools.add_knowledge("Python 协程", "async def 定义协程函数...", "tech")

# AI Agent 更新用户画像
tools.update_persona("user", "编程语言", "偏好 Python", 0.7, "提到协程")

# 跨库检索
query_emb = zhipu_embed("协程", cfg)
results = MemoryRetrieval(cfg).search_and_record(query_emb)
```

## Wiki 存储格式

所有记忆都是 `.md` 文件——用 Obsidian 打开 `wiki/` 目录即可：

**长期记忆** (`wiki/ltm/`):

```markdown
---
id: ltm_20260510_165511
index: 1
summary: 用户询问了 Python 异步编程的核心概念...
tags: [memory, ltm]
---

# 长期记忆 #1

用户询问了 Python 异步编程的核心概念，包括协程（async def）、
事件循环和 asyncio.run() 的使用方式...

---

*创建: 2026-05-10 16:55 | 引用: 3 | 原始 token: 266*
```

**知识条目** (`wiki/knowledge/`):

```markdown
---
id: k_20260510_165632
title: Python asyncio 最佳实践
category: tech_solution
tags: [knowledge, tech_solution]
---

# Python asyncio 最佳实践

使用 asyncio.gather 并发执行多个协程，注意用 return_exceptions=True
避免单点异常取消其他任务。

---

*分类: tech_solution | 来源: conversation_42 | 创建: 2026-05-10 16:56*
```

**人设/用户画像** (`wiki/persona/`):

```markdown
---
id: p_20260510_165632
type: user
trait: 编程风格
confidence: 0.85
evidence:
- 对话#42 提到 mypy strict
- 对话#50 用 asyncio 重构
tags: [persona, user]
---

# [用户画像] 编程风格

偏好严格类型注解和 asyncio 异步编程，使用 mypy strict 模式检查代码。

## 置信度

████████░░ 85%

## 证据

- 对话#42 提到 mypy strict
- 对话#50 用 asyncio 重构

---

*更新: 2026-05-10 16:56*
```

## 存储目录

```
wiki/
  ltm/         # 长期记忆摘要
  knowledge/   # 知识经验条目
  persona/     # 用户画像 + AI 人设
```

## 许可证

MIT
