"""知识/经验库。

AI 主动从对话中提取的可复用知识和经验教训。
双份存储：peo/.md (人读) + mac/.json (机读)。
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
        }

    def to_mac(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
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
        self._peo = self.config.knowledge_peo_dir
        self._mac = self.config.knowledge_mac_dir

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
        wu.write_entry(self._peo, self._mac, entry.id, meta, formatted, entry.to_mac())
        return entry

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        entry = wu.read_entry(self._peo, entry_id)
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
        wu.write_entry(self._peo, self._mac, entry_id, meta,
                      _format_knowledge_content(entry), entry.to_mac())

    def update_content(self, entry_id: str, content: str):
        entry = self.get(entry_id)
        if entry:
            entry.content = content
            entry.updated_at = time.time()
            self.update(entry_id, content=content)

    def list_all(self) -> list[KnowledgeEntry]:
        results = []
        for data in wu.scan_mac(self._mac):
            results.append(KnowledgeEntry.from_entry(data))
        results.sort(key=lambda e: e.updated_at, reverse=True)
        return results

    def list_by_category(self, category: str) -> list[KnowledgeEntry]:
        return [e for e in self.list_all() if e.category == category]

    def list_categories(self) -> list[str]:
        cats = set()
        for data in wu.scan_mac(self._mac):
            if "category" in data:
                cats.add(data["category"])
        return sorted(cats)

    def increment_ref(self, entry_id: str):
        entry = self.get(entry_id)
        if entry:
            entry.ref_count += 1
            wu.update_entry(self._peo, self._mac, entry_id,
                           {"ref_count": entry.ref_count},
                           mac_updates={"ref_count": entry.ref_count})

    def delete(self, entry_id: str):
        wu.delete_entry(self._peo, self._mac, entry_id)

    def search_by_vector(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        from .priority import cosine_similarity
        candidates = []
        for data in wu.scan_mac(self._mac):
            sim = cosine_similarity(query_embedding, data.get("embedding", []))
            candidates.append({
                "id": data.get("id", ""),
                "title": data.get("title", ""),
                "content": data.get("content", ""),
                "category": data.get("category", "general"),
                "embedding": data.get("embedding", []),
                "ref_count": data.get("ref_count", 0),
                "created_at": data.get("created_at", 0),
                "_similarity": sim,
            })
        candidates.sort(key=lambda x: x["_similarity"], reverse=True)
        return candidates[:top_k]

    def __len__(self):
        return len(wu.scan_mac(self._mac))


def _format_knowledge_content(entry: KnowledgeEntry) -> str:
    import time as _time
    ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(entry.created_at))
    return (
        f"# {entry.title}\n\n"
        f"{entry.content}\n\n"
        f"---\n\n"
        f"*分类: {entry.category} | 来源: {entry.source or 'unknown'} | 创建: {ts} | 引用: {entry.ref_count}*"
    )
