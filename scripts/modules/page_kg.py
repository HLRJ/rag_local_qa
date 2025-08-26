import streamlit as st

def render():
    st.subheader("🕸️ 图谱交互")
    # 复用你已有的可视化模块
    from scripts.neo4j_vis import show_neo4j_graph
    show_neo4j_graph()
