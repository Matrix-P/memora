"""Wiki .md + mac .json 双格式读写工具。

每个条目同步维护两份文件:
  - peo/{id}.md  人可读 (YAML frontmatter + 正文)
  - mac/{id}.json 机器读 (JSON: 向量 + 全部元数据)
"""

import os
import json
import hashlib
import yaml
from datetime import datetime


# ── YAML frontmatter (.md) ──────────────────────────────────

def parse_frontmatter(filepath: str) -> tuple[dict, str]:
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
    meta_str = yaml.dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False)
    text = f"---\n{meta_str}---\n\n{content}\n"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


# ── 双格式读写 ──────────────────────────────────────────────

def write_entry(peo_dir: str, mac_dir: str, entry_id: str,
                metadata: dict, content: str, mac_data: dict = None):
    """同时写 peo/.md 和 mac/.json。"""
    peo_path = os.path.join(peo_dir, f"{entry_id}.md")
    write_frontmatter(peo_path, metadata, content)
    mac_path = os.path.join(mac_dir, f"{entry_id}.json")
    os.makedirs(os.path.dirname(mac_path), exist_ok=True)
    if mac_data is None:
        mac_data = dict(metadata)
    with open(mac_path, "w", encoding="utf-8") as f:
        json.dump(mac_data, f, ensure_ascii=False, indent=2)


def read_entry(peo_dir: str, entry_id: str) -> dict | None:
    """从 peo/.md 读取条目。"""
    filepath = os.path.join(peo_dir, f"{entry_id}.md")
    if not os.path.isfile(filepath):
        return None
    meta, content = parse_frontmatter(filepath)
    meta["content"] = content
    meta["_path"] = filepath
    return meta


def read_mac(mac_dir: str, entry_id: str) -> dict | None:
    """从 mac/.json 读取机器数据。"""
    filepath = os.path.join(mac_dir, f"{entry_id}.json")
    if not os.path.isfile(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_entries(peo_dir: str) -> list[str]:
    """扫描 peo 目录下所有 .md 文件。"""
    if not os.path.isdir(peo_dir):
        return []
    return sorted(
        os.path.join(peo_dir, f)
        for f in os.listdir(peo_dir)
        if f.endswith(".md")
    )


def scan_mac(mac_dir: str) -> list[dict]:
    """扫描 mac 目录下所有 .json, 返回已解析 dict 列表 (用于向量搜索)。"""
    if not os.path.isdir(mac_dir):
        return []
    results = []
    for f in sorted(os.listdir(mac_dir)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(mac_dir, f), "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


def delete_entry(peo_dir: str, mac_dir: str, entry_id: str):
    """同时删 peo/.md 和 mac/.json。"""
    for d, ext in [(peo_dir, ".md"), (mac_dir, ".json")]:
        fp = os.path.join(d, f"{entry_id}{ext}")
        if os.path.isfile(fp):
            os.remove(fp)


def update_entry(peo_dir: str, mac_dir: str, entry_id: str,
                 metadata_updates: dict, content: str = None,
                 mac_updates: dict = None):
    """更新条目: 合并现有内容 + 更新 → 重写 peo + mac。"""
    existing = read_entry(peo_dir, entry_id)
    if existing is None:
        raise FileNotFoundError(f"Entry {entry_id} not found")
    new_content = content if content is not None else existing.get("content", "")
    existing.pop("_path", None)
    existing.pop("content", None)
    existing.update({k: v for k, v in metadata_updates.items() if v is not None})

    mac = read_mac(mac_dir, entry_id) or {}
    mac_updates = mac_updates or {}
    mac.update(mac_updates)
    mac.update({k: v for k, v in metadata_updates.items() if v is not None})

    write_entry(peo_dir, mac_dir, entry_id, existing, new_content, mac)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def file_hash(filepath: str) -> str | None:
    """计算文件内容 SHA256 哈希 (前 16 位 hex)。"""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]
