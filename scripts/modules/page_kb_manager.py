# scripts/modules/page_kb_manager.py
import re
import time
import hashlib
from pathlib import Path
import streamlit as st
from scripts.config import DATA_DIR
from scripts.vector_store_utils import (
    list_data_files, safe_save_upload,
    incremental_build_faiss, clear_vector_store, rebuild_vector_store
)

# ---------------- Utils ----------------

def safe_rerun():
    """兼容不同版本的 Streamlit 重刷页面"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _existing_subdirs(root: Path) -> list[str]:
    """列出现有子目录（相对 root），用于下拉选择"""
    root.mkdir(parents=True, exist_ok=True)
    # 只取第一层与多层目录（拍平显示），去重排序
    subs: set[str] = set()
    for p in root.rglob("*"):
        if p.is_dir():
            subs.add(str(p.relative_to(root)).replace("\\", "/"))
    return sorted([s for s in subs if s and s != "."])

def _sanitize_subdir(s: str) -> str:
    """
    规范化并校验用户输入的子目录：
    - 去除前后空白
    - 将反斜杠替换为斜杠
    - 禁止绝对路径、盘符、.. 回退
    - 允许多级路径，如 '监控类/摄像机'
    """
    s = (s or "").strip()
    s = s.replace("\\", "/")
    if not s:
        return ""
    # 禁止绝对/盘符/父级跳转
    if s.startswith("/") or re.match(r"^[a-zA-Z]:", s) or ".." in s.split("/"):
        return ""
    # 去除重复分隔符
    parts = [p for p in s.split("/") if p and p != "."]
    return "/".join(parts)

def _uploads_signature(files, subdir: str) -> str:
    """根据当前选择的上传文件 + 目标子目录生成一次性签名，用于会话内防重复提交"""
    if not files:
        return ""
    h = hashlib.md5()
    h.update(f"[subdir]{subdir}".encode("utf-8"))
    for f in files:
        name = getattr(f, "name", "")
        size = getattr(f, "size", 0)
        h.update(f"{name}::{size}".encode("utf-8"))
    return h.hexdigest()

# ---------------- Page ----------------

def render():
    st.subheader("📂 知识库管理")

    # 顶部文件列表
    files = list_data_files()
    if files:
        st.dataframe(files, use_container_width=True, hide_index=True)
    else:
        st.info("data/ 目录当前为空。")

    st.markdown("---")
    st.subheader("⬆️ 上传文件（pdf / docx / xlsx）")

    # 现有子目录供选择
    subdir_options = _existing_subdirs(DATA_DIR)
    with st.form("kb_upload_form", clear_on_submit=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            selected_subdir = st.selectbox(
                "📁 选择已有子目录（可留空）",
                options=[""] + subdir_options,
                index=0,
                help="从 data/ 下已有目录中选择；留空则上传到 data/ 根目录。"
            )
        with c2:
            input_subdir = st.text_input(
                "🆕 或新建/输入子目录（相对 data/，可多级，如 监控类/摄像机）",
                value=""
            )

        uploads = st.file_uploader(
            "选择文件后点击下方“开始上传”按钮",
            type=["pdf", "docx", "xlsx"],
            accept_multiple_files=True,
            key="kb_uploader"
        )
        submitted = st.form_submit_button("开始上传")

    # 选择优先级：若输入了新目录，用输入；否则用下拉选择
    target_subdir_raw = input_subdir.strip() or selected_subdir.strip()
    target_subdir = _sanitize_subdir(target_subdir_raw)

    if "last_upload_sig" not in st.session_state:
        st.session_state["last_upload_sig"] = ""

    if submitted and uploads:
        cur_sig = _uploads_signature(uploads, target_subdir)
        if cur_sig and cur_sig == st.session_state["last_upload_sig"]:
            st.info("这批文件刚刚已处理过，无需重复上传。")
        else:
            st.session_state["last_upload_sig"] = cur_sig
            target_dir = DATA_DIR / target_subdir if target_subdir else DATA_DIR
            saved = []
            for uf in uploads:
                target = safe_save_upload(uf, target_dir)   # 自动创建子目录
                saved.append(str(target.relative_to(DATA_DIR)))
            where = target_subdir if target_subdir else "根目录"
            st.success(f"已上传到 **{where}**：{saved}")
            try:
                st.toast("上传完成", icon="✅")
            except Exception:
                pass
            time.sleep(0.2)
            safe_rerun()

    st.markdown("---")
    st.subheader("🗑 删除文件")
    all_names = [row["文件"] for row in files]
    to_delete = st.multiselect(
        "选择要删除的文件（仅删除 data/ 中文件；向量库不自动清理）",
        options=all_names
    )
    if st.button("删除选中文件", disabled=len(to_delete) == 0):
        deleted = []
        for name in to_delete:
            path = DATA_DIR / name
            if path.exists():
                try:
                    path.unlink()
                    deleted.append(name)
                except Exception as e:
                    st.error(f"删除失败：{name} - {e}")
        if deleted:
            st.success(f"已删除：{deleted}")
            try:
                st.toast("删除完成", icon="🗑")
            except Exception:
                pass
            time.sleep(0.3)
            safe_rerun()

    st.markdown("---")
    st.subheader("🔄 向量库维护（FAISS）")

    ph = st.empty()
    bar = st.progress(0)
    def prog(done, total, msg):
        pct = int(done / max(total, 1) * 100)
        bar.progress(pct)
        ph.info(f"{pct}% - {msg}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("增量构建向量库"):
            bar.progress(0); ph.info("准备中…")
            with st.spinner("向量库增量构建中..."):
                result = incremental_build_faiss(progress_cb=prog)
            bar.progress(100)
            if not result["added_files"]:
                st.info("没有发现新文件需要索引。")
            else:
                merged_note = "（已合并到现有索引）" if result["merged"] else "（首次创建索引）"
                st.success(f"新增 {len(result['added_files'])} 个文件，切分 {result['chunks']} 块 {merged_note}")
            try:
                st.toast("向量库增量更新完成", icon="🎉")
            except Exception:
                pass
            safe_rerun()

    with c2:
        if st.button("清理向量库（删除索引文件）"):
            removed = clear_vector_store()
            if removed:
                st.warning("已删除：\n\n- " + "\n- ".join(removed))
            else:
                st.info("没有发现可删除的索引文件。")
            try:
                st.toast("清理完成", icon="🧹")
            except Exception:
                pass

    with c3:
        if st.button("完全重建向量库（清理后重建）"):
            bar.progress(0); ph.info("准备中…")
            with st.spinner("清理并重建向量库中..."):
                result = rebuild_vector_store(progress_cb=prog)
            bar.progress(100)
            if not result["added_files"]:
                st.info("data/ 中没有可用文件，未创建索引。")
            else:
                st.success(f"重建完成：{len(result['added_files'])} 个文件，切分 {result['chunks']} 块（已生成新索引）")
            try:
                st.toast("向量库重建完成", icon="✅")
            except Exception:
                pass
            safe_rerun()
