"""智谱嵌入 + DeepSeek 摘要 API 客户端。

提供同步和异步两种接口，方便在不同场景（测试、Agent 循环）中注入使用。
"""

import json
from urllib.request import Request, urlopen
from urllib.error import URLError
from .config import MemoryConfig, default_config


def zhipu_embed(text: str, config: MemoryConfig = None) -> list[float]:
    """调用智谱 Embedding API 获取文本向量。

    Args:
        text: 待嵌入的文本
        config: MemoryConfig 实例，需配置 embedding_api_key

    Returns:
        1024 维浮点向量列表
    """
    cfg = config or default_config
    if not cfg.embedding_api_key:
        raise ValueError("embedding_api_key 未设置，请在 MemoryConfig 中配置智谱 API Key")

    body = json.dumps({
        "model": cfg.embedding_model,
        "input": text,
    }).encode("utf-8")

    req = Request(
        cfg.embedding_api_base,
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.embedding_api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"][0]["embedding"]
    except URLError as e:
        raise RuntimeError(f"智谱 Embedding API 请求失败: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"智谱 Embedding API 返回格式异常: {e}")


def zhipu_embed_batch(texts: list[str], config: MemoryConfig = None) -> list[list[float]]:
    """批量嵌入（逐个调用，智谱未提供 batch 接口）。"""
    return [zhipu_embed(t, config) for t in texts]


async def zhipu_embed_async(text: str, config: MemoryConfig = None) -> list[float]:
    """异步版本：调用智谱 Embedding API。"""
    cfg = config or default_config
    if not cfg.embedding_api_key:
        raise ValueError("embedding_api_key 未设置")

    import httpx

    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        resp = await client.post(
            cfg.embedding_api_base,
            json={"model": cfg.embedding_model, "input": text},
            headers={
                "Authorization": f"Bearer {cfg.embedding_api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data["data"][0]["embedding"]


def deepseek_summarize(prompt: str, config: MemoryConfig = None, max_tokens: int = 300) -> str:
    """调用 DeepSeek API（同步）对对话片段生成摘要。

    Args:
        prompt: 摘要提示词（含待摘要的对话内容）
        config: MemoryConfig 实例，需配置 summary_api_key
        max_tokens: 摘要最大 token 数

    Returns:
        摘要文本
    """
    cfg = config or default_config
    if not cfg.summary_api_key:
        raise ValueError("summary_api_key 未设置，请在 MemoryConfig 中配置 DeepSeek API Key")

    body = json.dumps({
        "model": cfg.summary_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")

    req = Request(
        cfg.summary_api_base,
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.summary_api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except URLError as e:
        raise RuntimeError(f"DeepSeek API 请求失败: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"DeepSeek API 返回格式异常: {e}")


async def deepseek_summarize_async(prompt: str, config: MemoryConfig = None,
                                    max_tokens: int = 300) -> str:
    """异步版本：调用 DeepSeek API 生成摘要。

    签名为 async def(prompt: str) -> str，可直接注入 trim_and_summarize 的 llm_call 参数。
    """
    cfg = config or default_config
    if not cfg.summary_api_key:
        raise ValueError("summary_api_key 未设置")

    import httpx

    async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
        resp = await client.post(
            cfg.summary_api_base,
            json={
                "model": cfg.summary_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            headers={
                "Authorization": f"Bearer {cfg.summary_api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]
