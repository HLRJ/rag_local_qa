# 文件：scripts/query_rag_with_graph.py
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
from neo4j import GraphDatabase

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_PATH = BASE_DIR / "embeddings/faiss_store"
CHAT_HISTORY_FILE = BASE_DIR / "chat_history_graph_mix.json"

# 模型配置
MODEL_CONFIGS = {
    "Qwen-1.8B-SAFETENSORS": {
        "type": "transformers",
        "model_path": BASE_DIR / "models/Qwen/Qwen1.5-1.8B",
    },
    "MiniCPM4-0.5B": {
        "type": "transformers",
        "model_path": str(BASE_DIR / "models" / "openbmb" / "MiniCPM4-0.5B"),
    },
    "TinyLlama-1.1B-Chat-v1.0": {
        "type": "transformers",
        "model_path": str(BASE_DIR / "models" / "TinyLlama" / "TinyLlama-1.1B-Chat-v1.0"),
    },
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

# 图谱查询

def search_neo4j_triples(query_text):
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "12345678")
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        cypher = f"""
        MATCH (n:Entity) 
        WHERE n.name CONTAINS '{query_text}'
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.name AS source, type(r) AS relation, m.name AS target
        LIMIT 10
        """
        with driver.session() as session:
            results = session.run(cypher)
            triples = [f"{r['source']} --{r['relation']}--> {r['target']}" for r in results]
        return "\n".join(triples)
    except Exception as e:
        return ""

# PROMPT 模板，包含图谱内容
PROMPT_TEMPLATE = """
请只基于下列信息，回答用户的问题：

[Graph]
{graph_info}

[Docs]
{context}

问题：{question}

请用中文简洁回答。
"""

@st.cache_resource
def load_vector_store():
    embed = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    return FAISS.load_local(str(EMBEDDING_PATH), embed, allow_dangerous_deserialization=True)

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
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 2
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.1,
            eos_token_id=eos_token_id,
            return_full_text=False
        )
        return HuggingFacePipeline(pipeline=pipe)

# 历史问答

def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 主界面

def main():
    st.set_page_config(page_title="RAG + Graph 问答系统", layout="wide")
    st.title("🤖 维助通WeHelpOps")
    tool = st.sidebar.radio("🛠 功能模块", ["📘 RAG问答", "🕸️ 图谱交互"])

    if tool == "📘 RAG问答":
        db = load_vector_store()
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["graph_info", "context", "question"])
        history = load_chat_history()

        with st.sidebar:
            model_choice = st.selectbox("🤖 选择模型：", list(MODEL_CONFIGS.keys()))
            query = st.text_area("💬 输入你的问题：", "", height=150)
            do = st.button("🔍 提问")
            if st.button("🚹 清空历史记录"):
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
                graph_info = search_neo4j_triples(query) or "暂无相关图谱信息"

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt.partial(graph_info=graph_info)}
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

        if history:
            st.subheader("📌 当前回答")
            chat = history[-1]
            st.markdown(f"**📃 问题：** {chat['question']}")
            st.markdown(f"**🤖 回答：** {chat['answer']}")
            if chat["sources"]:
                with st.expander("📄 查看参考片段"):
                    for i, s in enumerate(chat["sources"], 1):
                        st.markdown(f"**片段{i}：{s['source']}**")
                        st.write(s["content"])

    elif tool == "🕸️ 图谱交互":
        from scripts.neo4j_vis import show_neo4j_graph
        show_neo4j_graph()

if __name__ == "__main__":
    main()