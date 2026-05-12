"""人设/用户库 (llmwiki #3)。

用户画像：偏好、习惯、背景、沟通风格。
AI 人设：角色定位、性格特征、行为准则、回答风格。

存储为带 frontmatter 的 .md 文件。
"""

import time
from .config import MemoryConfig, default_config
from . import wiki_utils as wu


class PersonaEntry:
    __slots__ = (
        "id", "type", "trait", "description",
        "confidence", "evidence", "embedding",
        "created_at", "updated_at",
    )

    def __init__(
        self,
        id: str = "",
        type: str = "",         # "user" | "ai_persona"
        trait: str = "",
        description: str = "",
        confidence: float = 0.5,
        evidence: list[str] = None,
        embedding: list[float] = None,
        created_at: float = None,
        updated_at: float = None,
    ):
        self.id = id
        self.type = type
        self.trait = trait
        self.description = description
        self.confidence = confidence
        self.evidence = evidence or []
        self.embedding = embedding or []
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()

    def to_meta(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "trait": self.trait,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding": self.embedding,
        }

    @classmethod
    def from_entry(cls, entry: dict) -> "PersonaEntry":
        return cls(
            id=entry.get("id", ""),
            type=entry.get("type", ""),
            trait=entry.get("trait", ""),
            description=entry.get("content", ""),
            confidence=entry.get("confidence", 0.5),
            evidence=entry.get("evidence", []),
            embedding=entry.get("embedding", []),
            created_at=entry.get("created_at"),
            updated_at=entry.get("updated_at"),
        )


class PersonaLibrary:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config
        self._dir = self.config.persona_dir

    def add(self, type: str, trait: str, description: str,
            confidence: float = 0.5, evidence: list[str] = None,
            embedding: list[float] = None) -> PersonaEntry:
        entry = PersonaEntry(
            id=wu.generate_id("p"),
            type=type,
            trait=trait,
            description=description,
            confidence=confidence,
            evidence=evidence or [],
            embedding=embedding or [],
        )
        meta = entry.to_meta()
        meta["tags"] = ["persona", type]
        formatted = _format_persona_content(entry)
        wu.write_entry(self._dir, entry.id, meta, formatted)
        return entry

    def get(self, entry_id: str) -> PersonaEntry | None:
        e = wu.read_entry(self._dir, entry_id)
        if e is None:
            return None
        return PersonaEntry.from_entry(e)

    def update(self, entry_id: str, **kwargs):
        entry = self.get(entry_id)
        if entry is None:
            raise FileNotFoundError(f"Persona entry {entry_id} not found")

        if "description" in kwargs:
            entry.description = kwargs.pop("description")
        for field in ("type", "trait", "confidence", "evidence", "embedding"):
            if field in kwargs:
                setattr(entry, field, kwargs[field])
        entry.updated_at = time.time()

        meta = entry.to_meta()
        meta["tags"] = ["persona", entry.type]
        wu.write_entry(self._dir, entry_id, meta, _format_persona_content(entry))

    def upsert(self, type: str, trait: str, description: str,
               confidence: float = 0.5, evidence: list[str] = None) -> PersonaEntry:
        """存在则更新置信度+证据，不存在则新建。"""
        existing = self.find_by_trait(type, trait)
        if existing:
            new_conf = min(1.0, existing.confidence + confidence * 0.1)
            merged_evidence = list(set(existing.evidence + (evidence or [])))
            self.update(existing.id,
                        description=description,
                        confidence=new_conf,
                        evidence=merged_evidence)
            return self.get(existing.id)
        return self.add(type, trait, description, confidence, evidence)

    def find_by_trait(self, type: str, trait: str) -> PersonaEntry | None:
        for e in self.list_all():
            if e.type == type and e.trait == trait:
                return e
        return None

    def list_all(self) -> list[PersonaEntry]:
        results = []
        for path in wu.scan_entries(self._dir):
            meta, content = wu.parse_frontmatter(path)
            meta["content"] = content
            results.append(PersonaEntry.from_entry(meta))
        results.sort(key=lambda e: e.updated_at, reverse=True)
        return results

    def list_by_type(self, type: str) -> list[PersonaEntry]:
        return [e for e in self.list_all() if e.type == type]

    def list_user_traits(self) -> list[PersonaEntry]:
        return self.list_by_type("user")

    def list_ai_persona(self) -> list[PersonaEntry]:
        return self.list_by_type("ai_persona")

    def delete(self, entry_id: str):
        wu.delete_entry_file(self._dir, entry_id)

    def search_by_vector(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        from .priority import cosine_similarity
        candidates = []
        for e in self.list_all():
            sim = cosine_similarity(query_embedding, e.embedding)
            candidates.append({
                "id": e.id,
                "type": e.type,
                "trait": e.trait,
                "description": e.description,
                "embedding": e.embedding,
                "ref_count": 0,
                "created_at": e.created_at,
                "_similarity": sim,
            })
        candidates.sort(key=lambda x: x["_similarity"], reverse=True)
        return candidates[:top_k]

    def rebuild(self) -> dict:
        """重构人设库：合并重复特征，清理低置信度条目。"""
        all_entries = self.list_all()
        merged = {}
        removed = 0

        for e in all_entries:
            key = (e.type, e.trait)
            if key in merged:
                existing = merged[key]
                existing.confidence = max(existing.confidence, e.confidence)
                existing.evidence = list(set(existing.evidence + e.evidence))
                self.delete(e.id)
                removed += 1
            else:
                merged[key] = e

        for (typ, trait), entry in merged.items():
            self.update(entry.id,
                        confidence=entry.confidence,
                        evidence=entry.evidence)

        return {"total": len(merged), "merged": removed}

    def __len__(self):
        return len(wu.scan_entries(self._dir))


def _format_persona_content(entry: PersonaEntry) -> str:
    import time as _time
    ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(entry.updated_at))
    type_label = "用户画像" if entry.type == "user" else "AI 人设"
    evidence_bullets = "\n".join(f"- {e}" for e in (entry.evidence or []))
    return (
        f"# [{type_label}] {entry.trait}\n\n"
        f"{entry.description}\n\n"
        f"## 置信度\n\n"
        f"{'█' * int(entry.confidence * 10)}{'░' * (10 - int(entry.confidence * 10))} "
        f"{entry.confidence:.0%}\n\n"
        f"## 证据\n\n"
        f"{evidence_bullets or '*（暂无）*'}\n\n"
        f"---\n\n"
        f"*更新: {ts}*"
    )
