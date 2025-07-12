# scripts/query_rag.py
import os
import streamlit as st
from pathlib import Path
import json
# 全部修正后的 imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


EMBED_MODEL = "BAAI/bge-large-zh" # "BAAI/bge-large-zh" "BAAI/bge-small-zh"

BASE_DIR = Path(__file__).resolve().parent.parent
CHAT_HISTORY_FILE = BASE_DIR / "chat_history.json"
# 中文提示模板
PROMPT_TEMPLATE = """
你是一个资深的运维专家，请根据以下知识内容回答用户的问题。
若无法直接从中找到答案，请基于常识合理推断。

知识内容：
{context}

问题：{question}

请用清晰简洁的中文回答。
"""

# prompt = PromptTemplate(
#     template=PROMPT_TEMPLATE,
#     input_variables=["context", "question"]
# )

MODEL_LIST = {
    "llama-2-7b.Q2_K": {
        "path": str(BASE_DIR / "models" / "llama" / "llama-2-7b.Q2_K.gguf"),
        "model_type": "llama"
    },
    "MiniCPM-2B-128k-Q2_K": {
        "path": str(BASE_DIR / "models" / "minicpm" / "MiniCPM-2B-128k-Q2_K.gguf"),
        "model_type": "llama"
    },
    "minicpm-2b-dpo-fp32.Q5_K_M": {
        "path": str(BASE_DIR / "models" / "minicpm" / "minicpm-2b-dpo-fp32.Q5_K_M.gguf"),
        "model_type": "llama"
    }
}


@st.cache_resource
def load_vector_store():
    model = HuggingFaceEmbeddings(model_name=EMBED_MODEL) # "BAAI/bge-small-zh"
    return FAISS.load_local(
        "embeddings/faiss_store",
        model,
        allow_dangerous_deserialization=True
    )
@st.cache_resource
def load_llm(model_cfg):
    return CTransformers(
        model=model_cfg["path"],
        model_type=model_cfg["model_type"],
        config={"max_new_tokens":512, "temperature":0.7,
                # "gpu_layers": 10,
                # "context_length": 2048
                }
    )

def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def main():
    st.set_page_config(page_title="中文RAG问答", layout="wide")
    st.title("📘 本地中文运维智能问答")
    db = load_vector_store()
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context","question"])
    history = load_chat_history()

    with st.sidebar:
        model_name = st.selectbox("选择模型", list(MODEL_LIST.keys()))
        query = st.text_area("输入你的问题", "", height=120)
        do = st.button("🔍 提问")
        if st.button("🗑 清空历史"):
            history = []
            save_chat_history(history)
            st.rerun()


    if do and query:
        with st.spinner("处理中..."):
            llm = load_llm(MODEL_LIST[model_name])
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":8})
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever,
                return_source_documents=True, chain_type_kwargs={"prompt":prompt}
            )
            res = qa.invoke(query)
            answer = res["result"]

            history.append({
                "question": query,
                "answer": answer,
                "sources": [
                    {"source": doc.metadata.get("source", ""), "content": doc.page_content[:300]+"..."}
                    for doc in res["source_documents"]
                ]
            })
            save_chat_history(history)
            st.rerun()


    st.subheader("💬 历史问答")
    if history:
        for idx, chat in enumerate(reversed(history), 1):
            st.markdown(f"**{idx}. 用户问题:** {chat['question']}")
            st.markdown(f"**🤖 回答:** {chat['answer']}")
            if chat["sources"]:
                with st.expander("查看参考片段"):
                    for sidx, s in enumerate(chat["sources"], 1):
                        st.markdown(f"**片段{sidx}: {s['source']}**")
                        st.write(s["content"])
    else:
        st.info("暂无历史记录。")

if __name__ == '__main__':
    main()