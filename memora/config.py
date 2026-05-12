from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    # 短期记忆裁剪
    stm_max_tokens: int = 8000   # M
    stm_trim_tokens: int = 4000  # N

    # 长期记忆检索
    recall_candidates: int = 20
    recall_final: int = 5

    # 优先级权重
    weight_similarity: float = 0.5
    weight_ref_count: float = 0.2
    weight_recency: float = 0.3

    # Sigmoid 温度
    tau_ref: float = 5.0
    tau_time: float = 86400.0   # 1天（秒）

    # 引用计数基数
    ref_base: int = 1

    # 嵌入模型 (智谱)
    embedding_model: str = "embedding-2"
    embedding_dim: int = 1024
    embedding_api_key: str = ""
    embedding_api_base: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    # 摘要模型 (DeepSeek)
    summary_model: str = "deepseek-v4-flash"
    summary_api_key: str = ""
    summary_api_base: str = "https://api.deepseek.com/v1/chat/completions"

    # 存储路径（相对于运行目录）
    wiki_dir: str = "wiki"
    ltm_dir: str = "wiki/ltm"
    knowledge_dir: str = "wiki/knowledge"
    persona_dir: str = "wiki/persona"


default_config = MemoryConfig()
