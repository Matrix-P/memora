"""AI Agent 可调用的记忆系统 Tool/Skill 定义。

提供给 LangChain/LangGraph agent 的标准工具接口。
每个函数支持同步和异步两种调用方式。
"""

from .ltm import LongTermMemory
from .knowledge import KnowledgeLibrary
from .persona import PersonaLibrary
from .retrieval import MemoryRetrieval
from .config import MemoryConfig, default_config


class MemoryTools:
    """记忆系统的工具集，封装为 Agent 可调用的接口。"""

    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config
        self.ltm = LongTermMemory(self.config)
        self.knowledge = KnowledgeLibrary(self.config)
        self.persona = PersonaLibrary(self.config)
        self.retrieval = MemoryRetrieval(self.config)

    # ── add_knowledge ──────────────────────────────────────

    def add_knowledge(
        self,
        title: str,
        content: str,
        category: str = "general",
        source: str = "",
    ) -> str:
        """从对话中提取知识并存入知识库。

        Args:
            title: 知识条目标题
            content: 知识内容
            category: 分类标签（如 coding_style, tech_solution, concept）
            source: 来源标识（对话ID等）
        Returns:
            新条目的 ID
        """
        entry = self.knowledge.add(title, content, category, source)
        return entry.id

    # ── update_persona ─────────────────────────────────────

    def update_persona(
        self,
        type: str,
        trait: str,
        description: str,
        confidence: float = 0.5,
        evidence: str = "",
    ) -> str:
        """更新用户画像或 AI 人设。存在则合并证据、提高置信度。

        Args:
            type: "user" 或 "ai_persona"
            trait: 特征名称
            description: 特征描述
            confidence: 置信度 (0~1)
            evidence: 支撑证据（引用对话片段或记忆ID）
        Returns:
            条目的 ID
        """
        evidence_list = [evidence] if evidence else []
        entry = self.persona.upsert(type, trait, description, confidence, evidence_list)
        return entry.id

    # ── search_memory ──────────────────────────────────────

    def search_memory(
        self,
        query: str,
        memory_type: str = "all",
        top_k: int = 5,
    ) -> str:
        """跨库检索记忆，返回格式化的文本结果。

        Args:
            query: 搜索查询
            memory_type: "ltm" | "knowledge" | "persona" | "all"
            top_k: 返回数量
        Returns:
            格式化的检索结果文本
        """
        from .config import default_config
        embedding = self._embed(query)
        results = self.retrieval.search_and_record(embedding, memory_type, top_k)
        return self.retrieval.build_context(results)

    # ── rebuild_library ────────────────────────────────────

    def rebuild_library(self, library: str = "all") -> str:
        """触发库重构，去重、合并、清理。

        Args:
            library: "knowledge" | "persona" | "all"
        Returns:
            重构统计信息
        """
        parts = []

        if library in ("knowledge", "all"):
            kb_entries = self.knowledge.list_all()
            merged = self._merge_similar_knowledge(kb_entries)
            parts.append(f"知识库: 原有 {len(kb_entries)} 条 → 合并后 {merged} 条")

        if library in ("persona", "all"):
            stats = self.persona.rebuild()
            parts.append(f"人设库: 原有条目 → 合并 {stats['merged']} 条重复，保留 {stats['total']} 条")

        if library in ("ltm", "all"):
            count = len(self.ltm)
            parts.append(f"长期记忆库: {count} 条记忆（未变更）")

        return "\n".join(parts) if parts else "未执行重构"

    # ── 辅助方法 ───────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """获取文本的向量嵌入。需要外部注入 embedding 函数。

        默认返回空列表，应在初始化后设置 embed_fn。
        """
        if hasattr(self, "_embed_fn") and self._embed_fn:
            return self._embed_fn(text)
        return []

    def set_embed_fn(self, fn):
        """注入嵌入函数 fn(text: str) -> list[float]"""
        self._embed_fn = fn

    def _merge_similar_knowledge(self, entries) -> int:
        """简单的知识条目合并：同标题只保留最新一条。"""
        seen = {}
        to_remove = []
        for e in entries:
            if e.title in seen:
                existing = seen[e.title]
                if e.updated_at > existing.updated_at:
                    to_remove.append(existing.id)
                    seen[e.title] = e
                else:
                    to_remove.append(e.id)
            else:
                seen[e.title] = e
        for eid in to_remove:
            self.knowledge.delete(eid)
        return len(seen)

    # ── 诊断 ────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "ltm_count": len(self.ltm),
            "knowledge_count": len(self.knowledge),
            "knowledge_categories": self.knowledge.list_categories(),
            "persona_count": len(self.persona),
            "user_traits": len(self.persona.list_user_traits()),
            "ai_traits": len(self.persona.list_ai_persona()),
        }
