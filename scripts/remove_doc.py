# 文件: scripts/remove_doc.py

import os
import json
import numpy as np
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

EMBEDDING_DIR = "../embeddings/faiss_store"
RECORD_FILE = "../embeddings/faiss_store/record.json"
EMBED_MODEL = "BAAI/bge-small-zh"

def load_indexed_files():
    if not os.path.exists(RECORD_FILE):
        return []
    with open(RECORD_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("indexed_files", [])

def save_indexed_files(file_list):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump({"indexed_files": file_list}, f, ensure_ascii=False, indent=2)

def main():
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vector_store_path = Path(EMBEDDING_DIR)
    if not vector_store_path.exists() or not Path(EMBEDDING_DIR, "index.faiss").exists():
        print("❌ 未找到向量库，请先执行 build_vector_store.py")
        return

    file_to_remove = input("请输入要移除的文件名（例如 运维手册.pdf）: ").strip()
    indexed_files = load_indexed_files()
    if file_to_remove not in indexed_files:
        print(f"⚠️ 文件 {file_to_remove} 不在已索引记录中，无需移除。")
        return

    print("🔗 加载现有向量库...")
    db = FAISS.load_local(EMBEDDING_DIR, embed_model, allow_dangerous_deserialization=True)
    print("✅ 加载完成，准备移除向量...")

    # 获取现有 docstore 中所有文档
    all_docs = list(db.docstore._dict.items())
    print(f"📊 当前向量库文档总数：{len(all_docs)}")

    # 找出需要保留的文档
    keep_doc_items = []
    keep_vectors = []
    for doc_id, doc in all_docs:
        source_file = doc.metadata.get("source", "")
        if source_file != file_to_remove:
            keep_doc_items.append((doc_id, doc))
            # 从向量库中重构该文档的向量
            vec = db.index.reconstruct(int(doc_id))
            keep_vectors.append(vec)

    print(f"✅ 保留文档数：{len(keep_doc_items)} （已移除 {len(all_docs) - len(keep_doc_items)} 条）")

    # 如果一个都不保留，直接清空
    if not keep_doc_items:
        print("🚨 警告：已移除所有向量库条目，向量库将被清空。")
        # 新建一个空向量库
        new_db = FAISS.from_documents([], embed_model)
        new_db.save_local(EMBEDDING_DIR)

        # 更新 record.json
        indexed_files.remove(file_to_remove)
        save_indexed_files(indexed_files)
        print(f"🎉 已清空向量库，并更新索引记录文件。")
        return

    # 重建索引
    dim = db.index.d
    new_index = faiss.IndexFlatL2(dim)
    new_index.add(np.array(keep_vectors).astype(np.float32))

    # 重建 docstore
    new_docstore = {}
    for doc_id, doc in keep_doc_items:
        new_docstore[doc_id] = doc

    # 重建 db
    new_db = FAISS(
        index=new_index,
        docstore=db.docstore.__class__(new_docstore),
        index_to_docstore_id={i: doc_id for i, (doc_id, _) in enumerate(keep_doc_items)},
        embedding_function=embed_model
    )

    new_db.save_local(EMBEDDING_DIR)

    # 更新 record.json
    indexed_files.remove(file_to_remove)
    save_indexed_files(indexed_files)
    print(f"🎉 已成功移除 {file_to_remove} 对应的向量，并更新向量库与记录文件。")

if __name__ == "__main__":
    main()
