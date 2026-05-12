"""长期记忆库 (llmwiki #1)。

存储由短期记忆裁剪后经 LLM 生成的摘要。
每条记忆携带：序号、摘要、原始token数、时间范围、引用计数、向量嵌入。
"""

import time
from .config import MemoryConfig, default_config
from . import wiki_utils as wu


class LTMMemory:
    __slots__ = (
        "id", "index", "summary", "original_tokens",
        "created_at", "time_start", "time_end",
        "ref_count", "embedding",
    )

    def __init__(
        self,
        id: str = "",
        index: int = 0,
        summary: str = "",
        original_tokens: int = 0,
        created_at: float = None,
        time_start: float = None,
        time_end: float = None,
        ref_count: int = None,
        embedding: list[float] = None,
    ):
        self.id = id
        self.index = index
        self.summary = summary
        self.original_tokens = original_tokens
        self.created_at = created_at or time.time()
        self.time_start = time_start
        self.time_end = time_end
        self.ref_count = ref_count if ref_count is not None else 0
        self.embedding = embedding or []

    def to_meta(self) -> dict:
        return {
            "id": self.id,
            "index": self.index,
            "summary": self.summary,
            "original_tokens": self.original_tokens,
            "created_at": self.created_at,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "ref_count": self.ref_count,
            "embedding": self.embedding,
        }

    @classmethod
    def from_entry(cls, entry: dict) -> "LTMMemory":
        return cls(
            id=entry.get("id", ""),
            index=entry.get("index", 0),
            summary=entry.get("summary") or entry.get("content", ""),
            original_tokens=entry.get("original_tokens", 0),
            created_at=entry.get("created_at"),
            time_start=entry.get("time_start"),
            time_end=entry.get("time_end"),
            ref_count=entry.get("ref_count", 0),
            embedding=entry.get("embedding", []),
        )


class LongTermMemory:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config
        self._dir = self.config.ltm_dir
        self._next_index = self._load_next_index()

    def _load_next_index(self) -> int:
        entries = wu.scan_entries(self._dir)
        max_idx = 0
        for path in entries:
            meta, _ = wu.parse_frontmatter(path)
            max_idx = max(max_idx, meta.get("index", 0))
        return max_idx + 1

    def add(self, summary: str, original_tokens: int = 0,
            time_start: float = None, time_end: float = None,
            embedding: list[float] = None) -> LTMMemory:
        import time as _time
        mem = LTMMemory(
            id=wu.generate_id("ltm"),
            index=self._next_index,
            summary=summary,
            original_tokens=original_tokens,
            time_start=time_start,
            time_end=time_end,
            ref_count=self.config.ref_base,
            embedding=embedding or [],
        )
        meta = mem.to_meta()
        meta["tags"] = ["memory", "ltm"]
        content = _format_ltm_content(mem)
        wu.write_entry(self._dir, mem.id, meta, content)
        self._next_index += 1
        return mem

    def get(self, entry_id: str) -> LTMMemory | None:
        entry = wu.read_entry(self._dir, entry_id)
        if entry is None:
            return None
        return LTMMemory.from_entry(entry)

    def list_all(self) -> list[LTMMemory]:
        results = []
        for path in wu.scan_entries(self._dir):
            meta, content = wu.parse_frontmatter(path)
            meta["content"] = content
            results.append(LTMMemory.from_entry(meta))
        results.sort(key=lambda m: m.index)
        return results

    def increment_ref(self, entry_id: str):
        mem = self.get(entry_id)
        if mem:
            mem.ref_count += 1
            wu.update_metadata(self._dir, entry_id, {"ref_count": mem.ref_count})

    def update_embedding(self, entry_id: str, embedding: list[float]):
        wu.update_metadata(self._dir, entry_id, {"embedding": embedding})

    def delete(self, entry_id: str):
        wu.delete_entry_file(self._dir, entry_id)

    def search_by_vector(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        """简易向量搜索：遍历所有记忆计算余弦相似度。"""
        from .priority import cosine_similarity
        candidates = []
        for mem in self.list_all():
            sim = cosine_similarity(query_embedding, mem.embedding)
            candidates.append({
                "id": mem.id,
                "summary": mem.summary,
                "embedding": mem.embedding,
                "ref_count": mem.ref_count,
                "created_at": mem.created_at,
                "_similarity": sim,
            })
        candidates.sort(key=lambda x: x["_similarity"], reverse=True)
        return candidates[:top_k]

    def __len__(self):
        return len(wu.scan_entries(self._dir))


def _format_ltm_content(mem: LTMMemory) -> str:
    import time as _time
    ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mem.created_at))
    return (
        f"# 长期记忆 #{mem.index}\n\n"
        f"{mem.summary}\n\n"
        f"---\n\n"
        f"*创建: {ts} | 引用: {mem.ref_count} | 原始 token: {mem.original_tokens}*"
    )
