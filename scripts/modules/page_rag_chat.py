import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from scripts.config import PROMPT_TEMPLATE, SESSIONS_FILE, MODEL_CONFIGS
from scripts.vector_store_utils import load_vector_store
from scripts.llm_utils import load_llm
from scripts.graph_utils import search_neo4j_triples
from scripts.memory_utils import format_chat_history, concat_or_summarize
import json

# ---- 会话读写 ----
def load_sessions():
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    return {}

def save_sessions(sessions):
    SESSIONS_FILE.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")

def render():
    st.subheader("📘 RAG 问答")

    # 会话管理
    sessions = load_sessions()
    if not sessions:
        sessions["默认会话"] = []
        save_sessions(sessions)

    st.sidebar.subheader("💬 对话列表")
    selected = st.sidebar.selectbox("选择对话", options=list(sessions.keys()))

    if st.sidebar.button("➕ 新建对话"):
        new_name = f"新对话{len(sessions)+1}"
        sessions[new_name] = []
        save_sessions(sessions); st.rerun()

    if st.sidebar.button("🗑 删除当前对话"):
        if selected in sessions:
            sessions.pop(selected)
            save_sessions(sessions); st.rerun()

    new_name = st.sidebar.text_input("✏️ 重命名当前对话", value=selected)
    if st.sidebar.button("💾 保存名称") and new_name.strip():
        if new_name != selected and new_name not in sessions:
            sessions[new_name] = sessions.pop(selected)
            save_sessions(sessions); st.rerun()
        elif new_name in sessions and new_name != selected:
            st.sidebar.warning("⚠️ 已存在同名对话，请换一个名字。")

    history = sessions.get(new_name if new_name in sessions else selected, [])

    # 向量库
    try:
        db = load_vector_store()
    except Exception:
        st.warning("未检测到向量库或加载失败，请先在『📂 知识库管理』中构建。")
        db = None

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["graph_info", "context", "question", "chat_history"]
    )
    model_choice = st.sidebar.selectbox("🤖 选择模型：", list(MODEL_CONFIGS.keys()))

    # 历史回放
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

    # 输入
    query = st.chat_input("请输入你的问题...")
    if query and db is not None:
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("🤖 思考中..."):
            llm = load_llm(model_choice)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 8, "score_threshold": 0.3})
            graph_info = search_neo4j_triples(query) or "暂无相关图谱信息"

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

        # 保存
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
