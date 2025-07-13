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
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig


# ========== 基础路径 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_PATH = BASE_DIR / "embeddings/faiss_store"
CHAT_HISTORY_FILE = BASE_DIR / "chat_history.json"

# ========== 模型列表 ==========
MODEL_CONFIGS = {
    "Qwen-1.8B-SAFETENSORS": {
        "type": "transformers",
        "model_path": BASE_DIR / "models/Qwen/Qwen1.5-1.8B",  # huggingface路径或本地路径
    },
    # "Baichuan2-7B-Chat": {
    #     "type": "transformers",
    #     "model_path": BASE_DIR / "models/Baichuan/Baichuan2-7B-Chat-4bits",
    # },
    "MiniCPM4-0.5B": {
        "type": "transformers",
        "model_path": str(BASE_DIR / "models" / "openbmb" / "MiniCPM4-0.5B"),
    },
    # "Yi-1.5-6B-Chat": {
    #     "type": "transformers",
    #     "model_path": BASE_DIR / "models/01-ai/Yi-1.5-6B-Chat",
    # },
    "TinyLlama-1.1B-Chat-v1.0": {
        "type": "transformers",
        "model_path": str(BASE_DIR / "models" / "TinyLlama" / "TinyLlama-1.1B-Chat-v1.0"),
    },
    # "phi-2.Q4_K_M.gguf": {
    #     "type": "gguf",
    #     "model_path": str(BASE_DIR / "models" / "phi-2" / "phi-2.Q4_K_M.gguf"),
    #     "model_type": "phi"
    # },
    "glm-edge-1.5b-chat": {
        "type": "transformers",
        "model_path": str(BASE_DIR / "models" / "THUDM" / "glm-edge-1.5b-chat"),
    },
    "llama-2-7b.Q4_K_M": {
        "type": "gguf",
        "model_path": str(BASE_DIR / "models" / "llama" / "llama-2-7b.Q4_K_M.gguf"),
        "model_type": "llama"
    },
}

# ========== PROMPT 模板 ==========
PROMPT_TEMPLATE = """
请仅基于下列信息，回答用户的问题，不要重复、不要生成多个版本：

----------------
{context}
----------------

问题：{question}

请用中文直接回答，简洁明了，不要重复提示、不要复述问题。只回答一次。
"""



# ========== 向量库加载 ==========
@st.cache_resource
def load_vector_store():
    embed = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    return FAISS.load_local(str(EMBEDDING_PATH), embed, allow_dangerous_deserialization=True)

# ========== 模型加载逻辑 ==========
@st.cache_resource
def load_llm(model_key):
    config = MODEL_CONFIGS[model_key]
    if config["type"] == "gguf":
        return CTransformers(
            model=str(config["model_path"]),
            model_type=config["model_type"],
            config={"max_new_tokens": 512, "temperature": 0.7, "gpu_layers": 15},
            repetition_penalty=1.1,
            stop=["\nUser:"],
        )
    elif config["type"] == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            trust_remote_code=True,
            use_fast=False,
        )


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # # ✅ 手动设置 device_map，部分模块卸载到 CPU
        # device_map = {
        #     "model.embed_tokens": "cuda:0",
        #     "model.layers.0": "cuda:0",
        #     "model.layers.1": "cpu",  # 显存不足，从第 1 层卸载到 CPU
        #     "model.layers.2": "cpu",
        #     "model.layers.3": "cpu",
        #     "model.layers.4": "cpu",
        #     "model.norm": "cpu",
        #     "lm_head": "cuda:0",
        # }
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            device_map="auto",
            torch_dtype=torch.float16,
            # load_in_8bit=True,  # 8bit量化
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        # ✅ 设置 stop_token
        eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 2

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.1,
            eos_token_id = eos_token_id,  # ✅ 控制生成终止
            return_full_text = False  # ✅ 只返回回答部分
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
        model_choice = st.selectbox("选择模型：", list(MODEL_CONFIGS.keys()))
        query = st.text_area("输入你的问题：", "", height=150)
        do = st.button("🔍 提问")
        if st.button("🧹 清空历史记录"):
            history = []
            save_chat_history(history)
            st.rerun()
    with st.expander("📜 历史问答记录", expanded=False):
        if history:
            for idx, chat in enumerate(reversed(history), 1):
                st.markdown(f"**{idx}. 用户问题：** {chat['question']}")
                st.markdown(f"**🤖 回答：** {chat['answer']}")
                if chat["sources"]:
                    for i, s in enumerate(chat["sources"], 1):
                        st.markdown(f"📄 **片段{i}：{s['source']}**")
                        st.caption(s["content"])
        else:
            st.info("暂无历史记录")
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

    # ✅ 仅展示最新一次回答
    if history:
        st.subheader("💬 当前回答")
        chat = history[-1]
        st.markdown(f"**🧾 问题：** {chat['question']}")
        st.markdown(f"**🤖 回答：** {chat['answer']}")
        if chat["sources"]:
            with st.expander("📄 查看参考片段"):
                for i, s in enumerate(chat["sources"], 1):
                    st.markdown(f"**片段{i}：{s['source']}**")
                    st.write(s["content"])

if __name__ == "__main__":
    main()
