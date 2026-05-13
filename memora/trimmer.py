"""对话裁剪与 LLM 摘要生成。

当短期记忆 token 超过 M 时：
1. 保留最近 N token 的消息
2. 将被裁剪的消息发送给 LLM 生成摘要
3. 摘要写入长期记忆库
"""

from .stm import ShortTermMemory, STMMessage
from .ltm import LongTermMemory
from .config import MemoryConfig, default_config


def build_trim_prompt(messages: list[STMMessage]) -> str:
    """构造给 LLM 的摘要提示词。"""
    conversation = ""
    for m in messages:
        conversation += f"[{m.role}]: {m.content}\n"

    return f"""请将以下对话片段总结为一段简洁的摘要（不超过200字），保留关键事实、决定和上下文：

{conversation}

摘要："""


async def trim_and_summarize(
    stm: ShortTermMemory,
    ltm: LongTermMemory,
    llm_call,
    embed_fn = None,
) -> LongTermMemory | None:
    """检查并执行裁剪+摘要流程。

    Args:
        stm: 短期记忆实例
        ltm: 长期记忆库实例
        llm_call: 异步函数，签名为 async def(llm, prompt: str) -> str
        embed_fn: 可选，异步函数 async def(text: str) -> list[float]

    Returns:
        新创建的 LTMMemory，如果不需要裁剪则返回 None
    """
    discard = stm.trim()
    if not discard:
        return None

    prompt = build_trim_prompt(discard)
    summary = await llm_call(prompt)
    if summary is None:
        return None

    time_start = discard[0].timestamp if discard else None
    time_end = discard[-1].timestamp if discard else None

    embedding = None
    if embed_fn:
        embedding = await embed_fn(summary)

    memory = ltm.add(
        summary=summary.strip(),
        time_start=time_start,
        time_end=time_end,
        embedding=embedding,
    )
    return memory


def naive_llm_trim(stm: ShortTermMemory) -> list[STMMessage] | None:
    """简化版裁剪（不生成摘要），直接丢弃早期消息。
    在 LLM 不可用时的降级方案。
    """
    return stm.trim()
