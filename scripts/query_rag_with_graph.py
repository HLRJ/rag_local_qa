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

# 导入记忆工具
from scripts.memory_utils import format_chat_history, concat_or_summarize

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_PATH = BASE_DIR / "embeddings/faiss_store"
SESSIONS_FILE = BASE_DIR / "chat_sessions.json"

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

# ===== 图谱查询 =====
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
    except Exception:
        return ""

# ===== Prompt 模板 =====
PROMPT_TEMPLATE = """
以下是用户与助手的历史对话，请结合上下文回答问题：

[History]
{chat_history}

[Graph]
{graph_info}

[Docs]
{context}

当前问题：{question}

请用中文简洁回答。
"""

# ===== 向量库 & 模型加载 =====
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

# ===== 会话管理 =====
def load_sessions():
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    return {}

def save_sessions(sessions):
    SESSIONS_FILE.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")

# ===== 主界面 =====
def main():
    st.set_page_config(page_title="RAG + Graph 问答系统", layout="wide")
    st.title("🤖 维助通 WeHelpOps")

    tool = st.sidebar.radio("🛠 功能模块", ["📘 RAG问答", "🕸️ 图谱交互"])
    sessions = load_sessions()

    # --- 会话管理 ---
    st.sidebar.subheader("💬 对话列表")
    if not sessions:
        sessions["默认会话"] = []
        save_sessions(sessions)

    selected = st.sidebar.selectbox("选择对话", options=list(sessions.keys()))

    # 新建会话
    if st.sidebar.button("➕ 新建对话"):
        new_name = f"新对话{len(sessions)+1}"
        sessions[new_name] = []
        save_sessions(sessions)
        st.rerun()

    # 删除会话
    if st.sidebar.button("🗑 删除当前对话"):
        if selected in sessions:
            sessions.pop(selected)
            save_sessions(sessions)
            st.rerun()

    # 重命名会话
    new_name = st.sidebar.text_input("✏️ 重命名当前对话", value=selected)
    if st.sidebar.button("💾 保存名称") and new_name.strip():
        if new_name != selected and new_name not in sessions:
            sessions[new_name] = sessions.pop(selected)
            save_sessions(sessions)
            st.rerun()
        elif new_name in sessions and new_name != selected:
            st.sidebar.warning("⚠️ 已存在同名对话，请换一个名字。")

    history = sessions.get(new_name if new_name in sessions else selected, [])

    # --- RAG 问答 ---
    if tool == "📘 RAG问答":
        db = load_vector_store()
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["graph_info", "context", "question", "chat_history"]
        )
        model_choice = st.sidebar.selectbox("🤖 选择模型：", list(MODEL_CONFIGS.keys()))

        # ✅ 聊天记录回放
        for chat in history:
            with st.chat_message("user"):
                st.markdown(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(chat["answer"])
                if chat["sources"]:
                    with st.expander("📄 参考片段", expanded=False):
                        for i, s in enumerate(chat["sources"], 1):
                            st.markdown(f"**片段{i}：{s['source']}**")
                            st.caption(s["content"][:300])

        # ✅ 输入框
        query = st.chat_input("请输入你的问题...")
        if query:
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("🤖 思考中..."):
                llm = load_llm(model_choice)
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 8, "score_threshold": 0.3})
                graph_info = search_neo4j_triples(query) or "暂无相关图谱信息"

                # 拼接历史对话
                chat_history_text = format_chat_history(history, max_rounds=8)
                chat_history_text = concat_or_summarize(chat_history_text, llm, max_tokens_hint=256)

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt.partial(graph_info=graph_info, chat_history=chat_history_text)}
                )
                res = qa.invoke(query)
                answer = res["result"]

            with st.chat_message("assistant"):
                st.markdown(answer)
                if res["source_documents"]:
                    with st.expander("📄 参考片段", expanded=False):
                        for i, s in enumerate(res["source_documents"], 1):
                            st.markdown(f"**片段{i}：{s.metadata.get('source','')}**")
                            st.caption(s.page_content[:300])

            # 保存到当前会话
            history.append({
                "question": query,
                "answer": answer,
                "sources": [
                    {"source": doc.metadata.get("source", ""), "content": doc.page_content[:300] + "..."}
                    for doc in res["source_documents"]
                ]
            })
            sessions[new_name if new_name in sessions else selected] = history
            save_sessions(sessions)
            st.rerun()

    elif tool == "🕸️ 图谱交互":
        from scripts.neo4j_vis import show_neo4j_graph
        show_neo4j_graph()


if __name__ == "__main__":
    main()
