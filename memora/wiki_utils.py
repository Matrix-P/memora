"""Wiki .md 文件的通用读写工具。

每个 wiki 条目存储为一个 .md 文件，包含 YAML frontmatter 头部和 Markdown 正文。
"""

import os
import json
import yaml
from datetime import datetime


def parse_frontmatter(filepath: str) -> tuple[dict, str]:
    """解析 .md 文件的 YAML frontmatter。

    Returns:
        (metadata: dict, content: str)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    meta = yaml.safe_load(parts[1]) or {}
    content = parts[2].strip()
    return meta, content


def write_frontmatter(filepath: str, metadata: dict, content: str):
    """写入带 YAML frontmatter 的 .md 文件。"""
    meta_str = yaml.dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False)
    text = f"---\n{meta_str}---\n\n{content}\n"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def scan_entries(directory: str) -> list[str]:
    """扫描目录下所有 .md 文件，返回文件路径列表。"""
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".md")
    ])


def read_entry(directory: str, entry_id: str) -> dict | None:
    """读取单个条目，返回 {**metadata, "content": ..., "_path": ...} 或 None。"""
    filepath = os.path.join(directory, f"{entry_id}.md")
    if not os.path.isfile(filepath):
        return None
    meta, content = parse_frontmatter(filepath)
    meta["content"] = content
    meta["_path"] = filepath
    return meta


def write_entry(directory: str, entry_id: str, metadata: dict, content: str):
    """写入单个条目到 .md 文件。"""
    filepath = os.path.join(directory, f"{entry_id}.md")
    write_frontmatter(filepath, metadata, content)


def update_metadata(directory: str, entry_id: str, updates: dict):
    """更新条目的元数据（不改内容）。"""
    entry = read_entry(directory, entry_id)
    if entry is None:
        raise FileNotFoundError(f"Entry {entry_id} not found in {directory}")
    content = entry.pop("content", "")
    entry.pop("_path", None)
    entry.update(updates)
    write_entry(directory, entry_id, entry, content)


def delete_entry_file(directory: str, entry_id: str):
    filepath = os.path.join(directory, f"{entry_id}.md")
    if os.path.isfile(filepath):
        os.remove(filepath)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
