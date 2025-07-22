# scripts/build_vector_stroe.py
from pathlib import Path
import os
import json
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 获取当前脚本所在目录 scripts/
SCRIPT_DIR = Path(__file__).resolve().parent

# 项目根目录
PROJECT_ROOT = SCRIPT_DIR.parent

# 正确的 data 路径
SOURCE_DIR = PROJECT_ROOT / "data"

EMBEDDING_DIR = PROJECT_ROOT / "embeddings/faiss_store"
RECORD_FILE = EMBEDDING_DIR / "record.json"
EMBED_MODEL = "BAAI/bge-large-zh"
# EMBED_MODEL = "BAAI/bge-small-zh"

def load_indexed_files():
    if not RECORD_FILE.exists():
        return []
    with open(RECORD_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("indexed_files", [])

def save_indexed_files(file_list):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump({"indexed_files": file_list}, f, ensure_ascii=False, indent=2)

def load_new_documents(indexed_files):
    new_docs = []
    new_files = []

    files = list(SOURCE_DIR.glob("*"))
    print("📁 当前 data 文件夹内容:", [str(f) for f in files])
    print("绝对路径:", SOURCE_DIR.resolve())

    for f in files:
        print("正在处理文件:", f)
        fname = f.name

        if fname in indexed_files:
            print(f"✅ 文件已存在索引，跳过: {fname}")
            continue

        loader = None
        try:
            if f.suffix.lower() == ".pdf":
                loader = PyMuPDFLoader(str(f))
            elif f.suffix.lower() == ".docx" : # or f.suffix.lower() == ".doc"
                loader = UnstructuredWordDocumentLoader(str(f))
            elif f.suffix.lower() == ".xlsx" : # or f.suffix.lower() == "xls"
                loader = UnstructuredExcelLoader(str(f))

            if loader:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = fname
                    new_docs.append(doc)
                new_files.append(fname)
                print(f"✅ 成功加载文档 {fname}，共 {len(docs)} 段文本。")
            else:
                print(f"⚠️ 文件 {fname} 格式不被支持，已跳过。")

        except Exception as e:
            print(f"❌ 加载文件 {fname} 发生错误：{str(e)}")

    return new_docs, new_files

def main():
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    indexed_files = load_indexed_files()
    print(f"✅ 已索引文件: {indexed_files}")

    new_docs, new_files = load_new_documents(indexed_files)
    if not new_docs:
        print("✅ 没有发现新的文件需要索引，已跳过。")
        return

    print(f"📄 本次新增 {len(new_files)} 文件: {new_files}")
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    new_chunks = splitter.split_documents(new_docs)
    print(f"✂️ 切分后得到 {len(new_chunks)} 个文本块")

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db_new = FAISS.from_documents(new_chunks, embed)

    if (EMBEDDING_DIR / "index.faiss").exists():
        print("🔗 加载已有向量库并合并...")
        db_existing = FAISS.load_local(str(EMBEDDING_DIR), embed, allow_dangerous_deserialization=True)
        db_existing.merge_from(db_new)
        db_existing.save_local(str(EMBEDDING_DIR))
    else:
        print("🚀 首次生成向量库...")
        db_new.save_local(str(EMBEDDING_DIR))

    indexed_files += new_files
    save_indexed_files(indexed_files)
    print("🎉 本次新增知识已索引完成！")

if __name__ == "__main__":
    main()
