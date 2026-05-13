from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    # 短期记忆裁剪
    stm_max_tokens: int = 8000   # M: 触发裁剪阈值
    stm_trim_tokens: int = 4000  # N: 裁剪后保留量

    # 长期记忆检索
    recall_candidates: int = 20
    recall_final: int = 5

    # 优先级权重
    weight_similarity: float = 0.5
    weight_ref_count: float = 0.2
    weight_recency: float = 0.3

    # Sigmoid 温度
    tau_ref: float = 5.0
    tau_time: float = 86400.0   # 1 天 (秒)

    # 引用计数基数
    ref_base: int = 1

    # 遗忘阈值 (priority < 此值 → 删除)
    forgetting_threshold: float = 0.15

    # 嵌入模型 (智谱)
    embedding_model: str = "embedding-2"
    embedding_dim: int = 1024
    embedding_api_key: str = ""
    embedding_api_base: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    # 摘要模型 (DeepSeek)
    summary_model: str = "deepseek-v4-flash"
    summary_api_key: str = ""
    summary_api_base: str = "https://api.deepseek.com/v1/chat/completions"

    # 存储路径
    mem_dir: str = "mem"

    @property
    def fixed_dir(self) -> str:
        return f"{self.mem_dir}/fixed"

    @property
    def memory_dir(self) -> str:
        return f"{self.mem_dir}/memory"

    @property
    def ltm_peo_dir(self) -> str:
        return f"{self.memory_dir}/ltm/peo"

    @property
    def ltm_mac_dir(self) -> str:
        return f"{self.memory_dir}/ltm/mac"

    @property
    def knowledge_peo_dir(self) -> str:
        return f"{self.memory_dir}/knowledge/peo"

    @property
    def knowledge_mac_dir(self) -> str:
        return f"{self.memory_dir}/knowledge/mac"

    @property
    def persona_peo_dir(self) -> str:
        return f"{self.memory_dir}/persona/peo"

    @property
    def persona_mac_dir(self) -> str:
        return f"{self.memory_dir}/persona/mac"

    @property
    def fixed_ltm_dir(self) -> str:
        return f"{self.fixed_dir}/ltm"

    @property
    def fixed_knowledge_dir(self) -> str:
        return f"{self.fixed_dir}/knowledge"

    @property
    def fixed_persona_dir(self) -> str:
        return f"{self.fixed_dir}/persona"


default_config = MemoryConfig()
