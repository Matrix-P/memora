"""统一检索接口。

工作流程：
1. 将查询向量化
2. 在各库中做向量搜索，召回候选
3. 用优先级公式对候选排序
4. 返回最终 top-k 结果，并更新引用计数
"""

import time
from .config import MemoryConfig, default_config
from .ltm import LongTermMemory
from .knowledge import KnowledgeLibrary
from .persona import PersonaLibrary
from .priority import rank_memories


class MemoryRetrieval:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config
        self.ltm = LongTermMemory(self.config)
        self.knowledge = KnowledgeLibrary(self.config)
        self.persona = PersonaLibrary(self.config)

    def search(
        self,
        query_embedding: list[float],
        memory_type: str = "all",
        top_k: int = None,
    ) -> dict:
        """跨库检索。

        Args:
            query_embedding: 查询的向量嵌入
            memory_type: "ltm" | "knowledge" | "persona" | "all"
            top_k: 最终返回数，默认使用配置中的 recall_final

        Returns:
            {"ltm": [...], "knowledge": [...], "persona": [...]}
        """
        if top_k is None:
            top_k = self.config.recall_final
        now = time.time()
        results = {}

        if memory_type in ("ltm", "all"):
            candidates = self.ltm.search_by_vector(
                query_embedding, self.config.recall_candidates
            )
            ranked = rank_memories(query_embedding, candidates, now, self.config)
            results["ltm"] = ranked[:top_k]

        if memory_type in ("knowledge", "all"):
            candidates = self.knowledge.search_by_vector(
                query_embedding, self.config.recall_candidates
            )
            ranked = rank_memories(query_embedding, candidates, now, self.config)
            results["knowledge"] = ranked[:top_k]

        if memory_type in ("persona", "all"):
            candidates = self.persona.search_by_vector(
                query_embedding, self.config.recall_candidates
            )
            ranked = rank_memories(query_embedding, candidates, now, self.config)
            results["persona"] = ranked[:top_k]

        return results

    def search_and_record(
        self,
        query_embedding: list[float],
        memory_type: str = "all",
        top_k: int = None,
    ) -> dict:
        """检索并自动更新引用计数。"""
        results = self.search(query_embedding, memory_type, top_k)

        for mem_type, items in results.items():
            for item in items:
                mem_id = item.get("id", "")
                if mem_type == "ltm":
                    self.ltm.increment_ref(mem_id)
                elif mem_type == "knowledge":
                    self.knowledge.increment_ref(mem_id)

        return results

    def build_context(self, results: dict) -> str:
        """将检索结果组装为 LLM 可读的上下文字符串。"""
        parts = []

        if results.get("persona"):
            parts.append("## 用户画像 / AI 人设")
            for p in results["persona"]:
                parts.append(
                    f"- [{p.get('type', '')}] **{p.get('trait', '')}**: "
                    f"{p.get('description', '')} "
                    f"(置信度: {p.get('confidence', 0):.2f})"
                )
            parts.append("")

        if results.get("knowledge"):
            parts.append("## 相关知识 / 经验")
            for k in results["knowledge"]:
                parts.append(
                    f"- [{k.get('category', '')}] **{k.get('title', '')}**: "
                    f"{k.get('content', '')}"
                )
            parts.append("")

        if results.get("ltm"):
            parts.append("## 历史对话摘要")
            for m in results["ltm"]:
                parts.append(f"- (长期记忆 #{m.get('index', '?')}) {m.get('summary', '')}")
            parts.append("")

        return "\n".join(parts)
