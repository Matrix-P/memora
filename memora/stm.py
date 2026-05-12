import time
import tiktoken
from dataclasses import dataclass, field
from .config import MemoryConfig, default_config


@dataclass
class STMMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "STMMessage":
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", time.time()),
            token_count=d.get("token_count", 0),
        )


class ShortTermMemory:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config
        self.messages: list[STMMessage] = []
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def add(self, role: str, content: str) -> STMMessage:
        token_count = self._estimate_tokens(content)
        msg = STMMessage(role=role, content=content, token_count=token_count)
        self.messages.append(msg)
        return msg

    def needs_trim(self) -> bool:
        return self.total_tokens > self.config.stm_max_tokens

    def get_trim_candidates(self) -> tuple[list[STMMessage], list[STMMessage]]:
        """返回 (保留部分, 裁剪部分)，保留最近约 N token的消息。"""
        if not self.needs_trim():
            return self.messages[:], []

        keep: list[STMMessage] = []
        discard: list[STMMessage] = []
        accumulated = 0

        for msg in reversed(self.messages):
            if accumulated + msg.token_count <= self.config.stm_trim_tokens:
                keep.insert(0, msg)
                accumulated += msg.token_count
            else:
                discard.insert(0, msg)

        return keep, discard

    def apply_trim(self, keep: list[STMMessage], discard: list[STMMessage]):
        self.messages = keep
        return discard

    def trim(self) -> list[STMMessage] | None:
        """执行裁剪，返回被丢弃的消息列表，不修改自身（由外部决定摘要后提交）。"""
        if not self.needs_trim():
            return None
        keep, discard = self.get_trim_candidates()
        self.apply_trim(keep, discard)
        return discard

    def clear(self):
        self.messages.clear()

    def _estimate_tokens(self, text: str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        return len(text) // 2

    def format_with_time(self) -> str:
        lines = []
        for m in self.messages:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp))
            lines.append(f"[{ts}] {m.role}: {m.content}")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]

    def __len__(self):
        return len(self.messages)
