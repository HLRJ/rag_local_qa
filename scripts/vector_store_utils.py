import json
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
)

from .config import DATA_DIR, FAISS_DIR, RECORD_FILE, EMBED_MODEL
import streamlit as st

# ---- 加载向量库（缓存）----
@st.cache_resource
def load_vector_store():
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)

# ---- 记录文件 ----
def load_indexed_files() -> List[str]:
    if not RECORD_FILE.exists():
        return []
    try:
        return json.loads(RECORD_FILE.read_text(encoding="utf-8")).get("indexed_files", [])
    except Exception:
        return []

def save_indexed_files(files: List[str]):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    RECORD_FILE.write_text(json.dumps({"indexed_files": files}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 文档加载器 ----
def _loader_for_path(path: Path):
    suf = path.suffix.lower()
    if suf == ".pdf":  return PyMuPDFLoader(str(path))
    if suf == ".docx": return UnstructuredWordDocumentLoader(str(path))
    if suf == ".xlsx": return UnstructuredExcelLoader(str(path))
    return None

# ---- 工具函数 ----
def human_size(num_bytes: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f}PB"

def list_data_files() -> List[Dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file():
            stat = f.stat()
            rows.append({
                "文件": str(f.relative_to(DATA_DIR)),
                "大小": human_size(stat.st_size),
                "修改时间": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "后缀": f.suffix.lower(),
            })
    return rows

def safe_save_upload(uploaded_file, target_dir: Path) -> Path:
    """保存上传文件，若同名则自动加 _1、_2 避免覆盖"""
    target_dir.mkdir(parents=True, exist_ok=True)   # ✅ 确保子目录存在
    base = Path(uploaded_file.name).stem
    suffix = Path(uploaded_file.name).suffix
    candidate = target_dir / (base + suffix)
    idx = 1
    while candidate.exists():
        candidate = target_dir / f"{base}_{idx}{suffix}"
        idx += 1
    with open(candidate, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return candidate


# ---- 增量构建 ----
def incremental_build_faiss(progress_cb: Callable[[int,int,str], None] | None = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    indexed = set(load_indexed_files())
    new_docs = []
    new_files = []

    all_files = [p for p in DATA_DIR.rglob("*") if p.is_file()]
    total = len(all_files)
    done = 0

    for f in all_files:
        rel = str(f.relative_to(DATA_DIR))
        done += 1
        if rel in indexed:
            if progress_cb: progress_cb(done, total, f"跳过已索引：{rel}")
            continue
        loader = _loader_for_path(f)
        if not loader:
            if progress_cb: progress_cb(done, total, f"不支持格式：{rel}")
            continue
        try:
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = rel
            new_docs.extend(docs)
            new_files.append(rel)
            if progress_cb: progress_cb(done, total, f"加载完成：{rel}（{len(docs)} 段）")
        except Exception as e:
            if progress_cb: progress_cb(done, total, f"❌ 加载失败：{rel} - {e}")

    if not new_docs:
        return {"added_files": [], "chunks": 0, "merged": False}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(new_docs)
    if progress_cb:
        progress_cb(total, total, f"切分完成，共 {len(chunks)} 块，开始向量化...")

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db_new = FAISS.from_documents(chunks, embed)

    index_path = FAISS_DIR / "index.faiss"
    if index_path.exists():
        db_exist = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
        db_exist.merge_from(db_new)
        db_exist.save_local(str(FAISS_DIR))
        merged = True
    else:
        db_new.save_local(str(FAISS_DIR))
        merged = False

    indexed = list(indexed) + new_files
    save_indexed_files(indexed)
    return {"added_files": new_files, "chunks": len(chunks), "merged": merged}

# ---- 清理/重建 ----
def clear_vector_store():
    removed = []
    if FAISS_DIR.exists():
        for p in FAISS_DIR.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                    removed.append(p.name)
                except Exception:
                    pass
    return removed

def rebuild_vector_store(progress_cb: Callable[[int,int,str], None] | None = None):
    _ = clear_vector_store()
    save_indexed_files([])
    return incremental_build_faiss(progress_cb=progress_cb)
