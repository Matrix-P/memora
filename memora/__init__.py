"""MEMORA: Memory Embedded with Markdown, Observable Retrieval & Adaptive Forgetting.

Wiki-based memory system for LLM agents:
  - ShortTermMemory: conversation window with timestamps
  - LongTermMemory: LLM-summarized conversation history (llmwiki #1)
  - KnowledgeLibrary: agent-extracted reusable knowledge (llmwiki #2)
  - PersonaLibrary: user profile + AI persona (llmwiki #3)
  - Unified retrieval: vector search + sigmoid priority scoring
  - Adaptive forgetting: time decay + ref_count decay
  - Wiki storage: .md + YAML frontmatter, Obsidian-compatible
"""

from .config import MemoryConfig, default_config
from .stm import ShortTermMemory, STMMessage
from .ltm import LongTermMemory, LTMMemory
from .knowledge import KnowledgeLibrary, KnowledgeEntry
from .persona import PersonaLibrary, PersonaEntry
from .retrieval import MemoryRetrieval
from .trimmer import trim_and_summarize, naive_llm_trim
from .tools import MemoryTools
from .priority import compute_priority, rank_memories, sigmoid
from .api_clients import (
    zhipu_embed,
    zhipu_embed_batch,
    zhipu_embed_async,
    deepseek_summarize,
    deepseek_summarize_async,
)

__all__ = [
    "MemoryConfig",
    "default_config",
    "ShortTermMemory",
    "STMMessage",
    "LongTermMemory",
    "LTMMemory",
    "KnowledgeLibrary",
    "KnowledgeEntry",
    "PersonaLibrary",
    "PersonaEntry",
    "MemoryRetrieval",
    "trim_and_summarize",
    "naive_llm_trim",
    "MemoryTools",
    "compute_priority",
    "rank_memories",
    "sigmoid",
    "zhipu_embed",
    "zhipu_embed_batch",
    "zhipu_embed_async",
    "deepseek_summarize",
    "deepseek_summarize_async",
]
