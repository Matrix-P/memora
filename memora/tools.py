"""AI Agent 可调用的记忆系统 Tool/Skill 定义。"""

import os
import time
from .ltm import LongTermMemory
from .knowledge import KnowledgeLibrary
from .persona import PersonaLibrary
from .retrieval import MemoryRetrieval
from .config import MemoryConfig, default_config
from . import wiki_utils as wu
from .priority import compute_priority


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
        self, title: str, content: str,
        category: str = "general", source: str = "",
    ) -> str:
        entry = self.knowledge.add(title, content, category, source)
        return entry.id

    # ── update_persona ─────────────────────────────────────

    def update_persona(
        self, type: str, trait: str, description: str,
        confidence: float = 0.5, evidence: str = "",
    ) -> str:
        evidence_list = [evidence] if evidence else []
        entry = self.persona.upsert(type, trait, description, confidence, evidence_list)
        return entry.id

    # ── search_memory ──────────────────────────────────────

    def search_memory(
        self, query: str, memory_type: str = "all", top_k: int = 5,
    ) -> str:
        embedding = self._embed(query)
        results = self.retrieval.search_and_record(embedding, memory_type, top_k)
        return self.retrieval.build_context(results)

    # ── rebuild_library ────────────────────────────────────

    def rebuild_library(self, library: str = "all") -> str:
        """整仓维护: 遗忘 + fixed 入库 + 去重。

        Args:
            library: "ltm" | "knowledge" | "persona" | "all"
        """
        parts = []
        now = time.time()

        if library in ("ltm", "all"):
            n_forgotten = self._forget_library("ltm", now)
            n_synced = self._sync_fixed("ltm")
            parts.append(
                f"长期记忆库: 遗忘 {n_forgotten} 条, fixed 同步 {n_synced} 条, "
                f"现存 {len(self.ltm)} 条"
            )

        if library in ("knowledge", "all"):
            n_forgotten = self._forget_library("knowledge", now)
            n_synced = self._sync_fixed("knowledge")
            kb_entries = self.knowledge.list_all()
            merged = self._merge_similar_knowledge(kb_entries)
            parts.append(
                f"知识库: 遗忘 {n_forgotten} 条, fixed 同步 {n_synced} 条, "
                f"合并后 {merged} 条"
            )

        if library in ("persona", "all"):
            n_forgotten = self._forget_library("persona", now)
            n_synced = self._sync_fixed("persona")
            stats = self.persona.rebuild()
            parts.append(
                f"人设库: 遗忘 {n_forgotten} 条, fixed 同步 {n_synced} 条, "
                f"去重合并 {stats['merged']} 条, 保留 {stats['total']} 条"
            )

        return "\n".join(parts) if parts else "未执行重构"

    # ── 遗忘 ───────────────────────────────────────────────

    def _forget_library(self, name: str, now: float) -> int:
        """遍历库中所有条目, 删除优先级低于阈值的。"""
        lib_map = {
            "ltm": self.ltm,
            "knowledge": self.knowledge,
            "persona": self.persona,
        }
        lib = lib_map[name]
        cfg = self.config
        to_delete = []

        for entry in lib.list_all():
            priority = compute_priority(
                similarity=0.5,  # 遗忘检查用中性相似度
                ref_count=entry.ref_count,
                created_at=entry.created_at,
                now=now,
                config=cfg,
            )
            if priority < cfg.forgetting_threshold:
                to_delete.append(entry.id)

        for eid in to_delete:
            lib.delete(eid)
        return len(to_delete)

    # ── fixed/ 同步 ────────────────────────────────────────

    def _sync_fixed(self, name: str) -> int:
        """扫描 fixed/{name}/ 下的 .md, 检测新增/修改 → 入库。"""
        cfg = self.config
        fixed_dirs = {
            "ltm": cfg.fixed_ltm_dir,
            "knowledge": cfg.fixed_knowledge_dir,
            "persona": cfg.fixed_persona_dir,
        }
        fixed_dir = fixed_dirs[name]
        if not os.path.isdir(fixed_dir):
            return 0

        # 加载历史哈希 (存在 peo/ 的元数据里或单独 track)
        known_hashes = self._load_fixed_hashes(name)
        synced = 0

        for fname in sorted(os.listdir(fixed_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(fixed_dir, fname)
            current_hash = wu.file_hash(fpath)
            if not current_hash:
                continue

            # 检查是否已处理过 (未变化)
            prev_id = known_hashes.get(fname)
            if prev_id and known_hashes.get(f"{fname}.hash") == current_hash:
                continue

            # 读内容
            meta, content = wu.parse_frontmatter(fpath)
            if not content.strip():
                continue

            processed = self._process_fixed_entry(name, meta, content, fname)
            if processed:
                # 记录 hash
                known_hashes[fname] = processed
                known_hashes[f"{fname}.hash"] = current_hash
                # 删除旧版本
                if prev_id and prev_id != processed:
                    lib = {"ltm": self.ltm, "knowledge": self.knowledge,
                           "persona": self.persona}[name]
                    try:
                        lib.delete(prev_id)
                    except Exception:
                        pass
                synced += 1

        self._save_fixed_hashes(name, known_hashes)
        return synced

    def _process_fixed_entry(self, name: str, meta: dict, content: str,
                             fname: str) -> str | None:
        """处理单条 fixed 文件: 嵌入 → 入库。返回 entry_id。"""
        if name == "ltm":
            title = meta.get("title", fname[:-3])
            # fixed LTM: 直接作为摘要入库
            emb = self._embed(content[:1000])
            mem = self.ltm.add(summary=content.strip(), embedding=emb)
            return mem.id

        elif name == "knowledge":
            title = meta.get("title", fname[:-3])
            category = meta.get("category", "general")
            source = meta.get("source", f"fixed/{fname}")
            emb = self._embed(f"{title} {content[:200]}")
            entry = self.knowledge.add(
                title=title, content=content.strip(),
                category=category, source=source, embedding=emb,
            )
            return entry.id

        elif name == "persona":
            etype = meta.get("type", "user")
            trait = meta.get("trait", fname[:-3])
            confidence = float(meta.get("confidence", 0.7))
            evidence = meta.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            emb = self._embed(f"{trait}: {content[:200]}")
            entry = self.persona.add(
                type=etype, trait=trait, description=content.strip(),
                confidence=confidence, evidence=evidence, embedding=emb,
            )
            return entry.id

        return None

    # ── fixed 哈希跟踪 ─────────────────────────────────────

    def _hash_file_path(self, name: str) -> str:
        return os.path.join(self.config.mem_dir, f".fixed_hashes_{name}.json")

    def _load_fixed_hashes(self, name: str) -> dict:
        import json
        fp = self._hash_file_path(name)
        if os.path.isfile(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_fixed_hashes(self, name: str, data: dict):
        import json
        fp = self._hash_file_path(name)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 辅助方法 ───────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        if hasattr(self, "_embed_fn") and self._embed_fn:
            return self._embed_fn(text)
        return []

    def set_embed_fn(self, fn):
        self._embed_fn = fn

    def _merge_similar_knowledge(self, entries) -> int:
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
