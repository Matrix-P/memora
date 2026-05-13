"""长期记忆库。

存储由短期记忆裁剪后经 LLM 生成的摘要。
每条记忆携带：序号、摘要、时间范围、引用计数、向量嵌入。
双份存储：peo/.md (人读) + mac/.json (机读)。
"""

import time
from .config import MemoryConfig, default_config
from . import wiki_utils as wu


class LTMMemory:
    __slots__ = (
        "id", "index", "summary", "created_at",
        "time_start", "time_end", "ref_count", "embedding",
    )

    def __init__(
        self,
        id: str = "",
        index: int = 0,
        summary: str = "",
        created_at: float = None,
        time_start: float = None,
        time_end: float = None,
        ref_count: int = None,
        embedding: list[float] = None,
    ):
        self.id = id
        self.index = index
        self.summary = summary
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
            "created_at": self.created_at,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "ref_count": self.ref_count,
        }

    def to_mac(self) -> dict:
        d = self.to_meta()
        d["embedding"] = self.embedding
        return d

    @classmethod
    def from_entry(cls, entry: dict) -> "LTMMemory":
        return cls(
            id=entry.get("id", ""),
            index=entry.get("index", 0),
            summary=entry.get("summary") or entry.get("content", ""),
            created_at=entry.get("created_at"),
            time_start=entry.get("time_start"),
            time_end=entry.get("time_end"),
            ref_count=entry.get("ref_count", 0),
            embedding=entry.get("embedding", []),
        )


class LongTermMemory:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config
        self._peo = self.config.ltm_peo_dir
        self._mac = self.config.ltm_mac_dir
        self._next_index = self._load_next_index()

    def _load_next_index(self) -> int:
        max_idx = 0
        for data in wu.scan_mac(self._mac):
            max_idx = max(max_idx, data.get("index", 0))
        return max_idx + 1

    def add(self, summary: str,
            time_start: float = None, time_end: float = None,
            embedding: list[float] = None) -> LTMMemory:
        import time as _time
        mem = LTMMemory(
            id=wu.generate_id("ltm"),
            index=self._next_index,
            summary=summary,
            time_start=time_start,
            time_end=time_end,
            ref_count=self.config.ref_base,
            embedding=embedding or [],
        )
        meta = mem.to_meta()
        meta["tags"] = ["memory", "ltm"]
        content = _format_ltm_content(mem)
        wu.write_entry(self._peo, self._mac, mem.id, meta, content, mem.to_mac())
        self._next_index += 1
        return mem

    def get(self, entry_id: str) -> LTMMemory | None:
        entry = wu.read_entry(self._peo, entry_id)
        if entry is None:
            return None
        return LTMMemory.from_entry(entry)

    def list_all(self) -> list[LTMMemory]:
        results = []
        for data in wu.scan_mac(self._mac):
            results.append(LTMMemory.from_entry(data))
        results.sort(key=lambda m: m.index)
        return results

    def increment_ref(self, entry_id: str):
        mem = self.get(entry_id)
        if mem:
            mem.ref_count += 1
            wu.update_entry(self._peo, self._mac, entry_id,
                           {"ref_count": mem.ref_count},
                           mac_updates={"ref_count": mem.ref_count})

    def update_embedding(self, entry_id: str, embedding: list[float]):
        wu.update_entry(self._peo, self._mac, entry_id,
                       {"embedding": embedding},
                       mac_updates={"embedding": embedding})

    def delete(self, entry_id: str):
        wu.delete_entry(self._peo, self._mac, entry_id)

    def search_by_vector(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        """向量搜索: 直接从 mac/ JSON 读取 (无需解析 YAML)。"""
        from .priority import cosine_similarity
        candidates = []
        for data in wu.scan_mac(self._mac):
            sim = cosine_similarity(query_embedding, data.get("embedding", []))
            candidates.append({
                "id": data.get("id", ""),
                "summary": data.get("summary", ""),
                "embedding": data.get("embedding", []),
                "ref_count": data.get("ref_count", 0),
                "created_at": data.get("created_at", 0),
                "_similarity": sim,
            })
        candidates.sort(key=lambda x: x["_similarity"], reverse=True)
        return candidates[:top_k]

    def __len__(self):
        return len(wu.scan_mac(self._mac))


def _format_ltm_content(mem: LTMMemory) -> str:
    import time as _time
    ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mem.created_at))
    return (
        f"# 长期记忆 #{mem.index}\n\n"
        f"{mem.summary}\n\n"
        f"---\n\n"
        f"*创建: {ts} | 引用: {mem.ref_count}*"
    )
