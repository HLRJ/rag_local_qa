# 文件：scripts/run_web_ui.py
# import os
# from pathlib import Path
# current_dir = Path(__file__).resolve().parent
# target_script = current_dir / "query_rag.py"
# os.system(f"streamlit run {target_script}")
import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
vector_store_file = current_dir.parent / "embeddings/faiss_store/index.faiss"
query_script = current_dir / "query_rag_with_graph.py" # query_rag_with_graph.py query_rag_mixed.py

if not vector_store_file.exists():
    print("❌ 未检测到向量库 embeddings/faiss_store/index.faiss")
    print("请先执行：python scripts/build_vector_store.py 来生成知识库。")
    sys.exit(1)

print("🚀 检测到向量库，准备启动 Streamlit...")
os.system(f"streamlit run {query_script}")
