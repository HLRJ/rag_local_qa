# scripts/app.py
import streamlit as st
# --- 确保项目根在 sys.path 中 ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # .../rag_local_qa
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modules import page_rag_chat, page_kb_manager, page_kg
def main():
    st.set_page_config(page_title="RAG + Graph 问答系统", layout="wide")
    st.title("🤖 维助通 WeHelpOps")

    page = st.sidebar.radio("🛠 功能模块", ["📘 RAG问答", "🕸️ 图谱交互", "📂 知识库管理"])
    if page == "📘 RAG问答":
        page_rag_chat.render()
    elif page == "🕸️ 图谱交互":
        page_kg.render()
    else:
        page_kb_manager.render()

if __name__ == "__main__":
    main()
