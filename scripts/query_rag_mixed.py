# 文件：scripts/query_rag_mixed.py
import os
import json
from pathlib import Path
import streamlit as st
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ========== 基础路径 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_PATH = BASE_DIR / "embeddings/faiss_store"
CHAT_HISTORY_FILE = BASE_DIR / "chat_history.json"

# ========== 模型列表 ==========
MODEL_CONFIG = {
    "llama-2-7b.Q2_K": {
        "type": "gguf",
        "model_path": str(BASE_DIR / "models" / "llama" / "llama-2-7b.Q2_K.gguf"),
        "model_type": "llama"
    },
    "Qwen-1.8B-SAFETENSORS": {
        "type": "hf",
        "model_path": BASE_DIR / "models/Qwen/Qwen1.5-1.8B",  # huggingface路径或本地路径
    }
}

# ========== PROMPT 模板 ==========
PROMPT_TEMPLATE = """
已知信息如下：
----------------
{context}
----------------

用户提问：
{question}

### 回答：
"""


# ========== 向量库加载 ==========
@st.cache_resource
def load_vector_store():
    embed = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    return FAISS.load_local(str(EMBEDDING_PATH), embed, allow_dangerous_deserialization=True)

# ========== 模型加载逻辑 ==========
@st.cache_resource
def load_llm(model_key):
    config = MODEL_CONFIG[model_key]
    if config["type"] == "gguf":
        return CTransformers(
            model=str(config["model_path"]),
            model_type=config["model_type"],
            config={"max_new_tokens": 512, "temperature": 0.7}
        )
    elif config["type"] == "hf":
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True  # 自动适配低显存
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.1
        )
        return HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError("不支持的模型类型")

# ========== 历史问答 ==========
def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ========== 主界面 ==========
def main():
    st.set_page_config(page_title="RAG 运维问答（GGUF + safetensors）", layout="wide")
    st.title("📘 本地中文运维智能问答")

    db = load_vector_store()
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    history = load_chat_history()

    with st.sidebar:
        model_choice = st.selectbox("选择模型：", list(MODEL_CONFIG.keys()))
        query = st.text_area("输入你的问题：", "", height=150)
        do = st.button("🔍 提问")
        if st.button("🧹 清空历史记录"):
            history = []
            save_chat_history(history)
            st.rerun()

    if do and query.strip():
        with st.spinner("🔄 正在处理..."):
            llm = load_llm(model_choice)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 8, "score_threshold": 0.3})
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            res = qa.invoke(query)
            answer = res["result"]

            history.append({
                "question": query,
                "answer": answer,
                "sources": [
                    {"source": doc.metadata.get("source", ""), "content": doc.page_content[:300] + "..."}
                    for doc in res["source_documents"]
                ]
            })
            save_chat_history(history)
            st.rerun()

    st.subheader("💬 历史问答记录")
    if history:
        for idx, chat in enumerate(reversed(history), 1):
            st.markdown(f"**{idx}. 用户提问：** {chat['question']}")
            st.markdown(f"**🤖 回答：** {chat['answer']}")
            if chat["sources"]:
                with st.expander("📄 查看参考片段"):
                    for i, s in enumerate(chat["sources"], 1):
                        st.markdown(f"**片段{i}：{s['source']}**")
                        st.write(s["content"])
    else:
        st.info("暂无历史记录")

if __name__ == "__main__":
    main()
