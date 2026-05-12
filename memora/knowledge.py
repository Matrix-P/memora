"""知识/经验库 (llmwiki #2)。

AI 主动从对话中提取的可复用知识和经验教训。
存储为带 frontmatter 的 .md 文件。
"""

import time
from .config import MemoryConfig, default_config
from . import wiki_utils as wu


class KnowledgeEntry:
    __slots__ = (
        "id", "title", "content", "category",
        "source", "embedding", "ref_count",
        "created_at", "updated_at",
    )

    def __init__(
        self,
        id: str = "",
        title: str = "",
        content: str = "",
        category: str = "general",
        source: str = "",
        embedding: list[float] = None,
        ref_count: int = None,
        created_at: float = None,
        updated_at: float = None,
    ):
        self.id = id
        self.title = title
        self.content = content
        self.category = category
        self.source = source
        self.embedding = embedding or []
        self.ref_count = ref_count if ref_count is not None else 0
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()

    def to_meta(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "source": self.source,
            "ref_count": self.ref_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding": self.embedding,
        }

    @classmethod
    def from_entry(cls, entry: dict) -> "KnowledgeEntry":
        return cls(
            id=entry.get("id", ""),
            title=entry.get("title", ""),
            content=entry.get("content", ""),
            category=entry.get("category", "general"),
            source=entry.get("source", ""),
            embedding=entry.get("embedding", []),
            ref_count=entry.get("ref_count", 0),
            created_at=entry.get("created_at"),
            updated_at=entry.get("updated_at"),
        )


class KnowledgeLibrary:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config
        self._dir = self.config.knowledge_dir

    def add(self, title: str, content: str, category: str = "general",
            source: str = "", embedding: list[float] = None) -> KnowledgeEntry:
        entry = KnowledgeEntry(
            id=wu.generate_id("k"),
            title=title,
            content=content,
            category=category,
            source=source,
            embedding=embedding or [],
            ref_count=self.config.ref_base,
        )
        meta = entry.to_meta()
        meta["tags"] = ["knowledge", category]
        formatted = _format_knowledge_content(entry)
        wu.write_entry(self._dir, entry.id, meta, formatted)
        return entry

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        entry = wu.read_entry(self._dir, entry_id)
        if entry is None:
            return None
        return KnowledgeEntry.from_entry(entry)

    def update(self, entry_id: str, **kwargs):
        entry = self.get(entry_id)
        if entry is None:
            raise FileNotFoundError(f"Knowledge entry {entry_id} not found")
        for field in ("title", "content", "category", "source", "embedding"):
            if field in kwargs:
                setattr(entry, field, kwargs[field])
        entry.updated_at = time.time()
        meta = entry.to_meta()
        meta["tags"] = ["knowledge", entry.category]
        wu.write_entry(self._dir, entry_id, meta, _format_knowledge_content(entry))

    def update_content(self, entry_id: str, content: str):
        entry = self.get(entry_id)
        if entry:
            entry.content = content
            entry.updated_at = time.time()
            meta = entry.to_meta()
            meta["tags"] = ["knowledge", entry.category]
            wu.write_entry(self._dir, entry_id, meta, _format_knowledge_content(entry))

    def list_all(self) -> list[KnowledgeEntry]:
        results = []
        for path in wu.scan_entries(self._dir):
            meta, content = wu.parse_frontmatter(path)
            meta["content"] = content
            results.append(KnowledgeEntry.from_entry(meta))
        results.sort(key=lambda e: e.updated_at, reverse=True)
        return results

    def list_by_category(self, category: str) -> list[KnowledgeEntry]:
        return [e for e in self.list_all() if e.category == category]

    def list_categories(self) -> list[str]:
        cats = set()
        for path in wu.scan_entries(self._dir):
            meta, _ = wu.parse_frontmatter(path)
            if "category" in meta:
                cats.add(meta["category"])
        return sorted(cats)

    def increment_ref(self, entry_id: str):
        entry = self.get(entry_id)
        if entry:
            entry.ref_count += 1
            wu.update_metadata(self._dir, entry_id, {"ref_count": entry.ref_count})

    def delete(self, entry_id: str):
        wu.delete_entry_file(self._dir, entry_id)

    def search_by_vector(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        from .priority import cosine_similarity
        candidates = []
        for e in self.list_all():
            sim = cosine_similarity(query_embedding, e.embedding)
            candidates.append({
                "id": e.id,
                "title": e.title,
                "content": e.content,
                "category": e.category,
                "embedding": e.embedding,
                "ref_count": e.ref_count,
                "created_at": e.created_at,
                "_similarity": sim,
            })
        candidates.sort(key=lambda x: x["_similarity"], reverse=True)
        return candidates[:top_k]

    def __len__(self):
        return len(wu.scan_entries(self._dir))


def _format_knowledge_content(entry: KnowledgeEntry) -> str:
    import time as _time
    ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(entry.created_at))
    return (
        f"# {entry.title}\n\n"
        f"{entry.content}\n\n"
        f"---\n\n"
        f"*分类: {entry.category} | 来源: {entry.source or 'unknown'} | 创建: {ts} | 引用: {entry.ref_count}*"
    )
