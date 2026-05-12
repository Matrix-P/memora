import math
from .config import MemoryConfig, default_config


def sigmoid(x: float) -> float:
    """Sigmoid 函数，将任意实数映射到 (0, 1)。"""
    return 1.0 / (1.0 + math.exp(-x))


def ref_count_priority(ref_count: int, base: int = 1, tau: float = 5.0) -> float:
    """引用计数优先级，通过 sigmoid 平滑。

    Args:
        ref_count: 当前引用次数
        base: 基数，新记忆从 base 开始，避免优先级过低
        tau: 温度参数，越大曲线越平缓
    """
    return sigmoid((base + ref_count) / tau)


def recency_priority(created_at: float, now: float, tau: float = 86400.0) -> float:
    """时间近度优先级，越近越高。使用反向 sigmoid。

    Args:
        created_at: 创建时间戳
        now: 当前时间戳
        tau: 温度参数（秒），默认约1天
    """
    age = now - created_at
    return 1.0 - sigmoid(age / tau)


def compute_priority(
    similarity: float,
    ref_count: int,
    created_at: float,
    now: float,
    config: MemoryConfig = None,
) -> float:
    """综合优先级计算。

    priority = sim * w_sim + σ(ref) * w_ref + recency(age) * w_time
    """
    cfg = config or default_config
    score = (
        similarity * cfg.weight_similarity
        + ref_count_priority(ref_count, cfg.ref_base, cfg.tau_ref)
        * cfg.weight_ref_count
        + recency_priority(created_at, now, cfg.tau_time)
        * cfg.weight_recency
    )
    return score


def rank_memories(
    query_embedding: list[float],
    candidates: list[dict],
    now: float = None,
    config: MemoryConfig = None,
) -> list[dict]:
    """对候选记忆列表计算优先级并排序，返回排序后的列表。

    每个 candidate 是 dict，需包含 embedding, ref_count, created_at。
    """
    import time
    cfg = config or default_config
    if now is None:
        now = time.time()

    scored = []
    for c in candidates:
        sim = cosine_similarity(query_embedding, c.get("embedding", []))
        priority = compute_priority(
            similarity=sim,
            ref_count=c.get("ref_count", 0),
            created_at=c.get("created_at", now),
            now=now,
            config=cfg,
        )
        c_copy = dict(c)
        c_copy["_similarity"] = sim
        c_copy["_priority"] = priority
        scored.append(c_copy)

    scored.sort(key=lambda x: x["_priority"], reverse=True)
    return scored


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
