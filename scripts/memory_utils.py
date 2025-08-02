# scripts/memory_utils.py
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import List, Dict, Tuple

def get_session_id(username: str = "default") -> str:
    # 简单可重复的会话 id
    return hashlib.md5(username.encode("utf-8")).hexdigest()[:10]

def load_history(store_file: Path, session_id: str) -> List[Dict]:
    if store_file.exists():
        db = json.loads(store_file.read_text(encoding="utf-8"))
        return db.get(session_id, [])
    return []

def save_history(store_file: Path, session_id: str, history: List[Dict]):
    store_file.parent.mkdir(parents=True, exist_ok=True)
    db = {}
    if store_file.exists():
        db = json.loads(store_file.read_text(encoding="utf-8"))
    db[session_id] = history[-200:]  # 最多保留200轮，防爆
    store_file.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

def format_chat_history(history: List[Dict], max_rounds: int = 6) -> str:
    """将最近 n 轮拼接为纯文本，供 Prompt 注入"""
    hist = history[-max_rounds:]
    lines = []
    for turn in hist:
        q = turn.get("question","").strip()
        a = turn.get("answer","").strip()
        if q: lines.append(f"用户：{q}")
        if a: lines.append(f"助手：{a}")
    return "\n".join(lines)

def concat_or_summarize(history_text: str, llm, max_tokens_hint: int = 256) -> str:
    """可选：让本地LLM把较长历史压缩为摘要，降低上下文长度"""
    if not history_text:
        return ""
    if len(history_text) < 1500:
        return history_text
    prompt = (
        "以下是多轮对话内容，请压缩为不丢关键信息的中文摘要（不超过约{}字），"
        "突出用户的意图、限定条件、已给出的结论与未决问题：\n\n{}"
    ).format(max_tokens_hint, history_text)
    try:
        summary = llm(prompt) if callable(llm) else llm.invoke(prompt)  # 兼容 HF/LLM
        return summary if isinstance(summary, str) else getattr(summary, "content", str(summary))
    except Exception:
        return history_text[:1500]
