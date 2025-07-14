import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# 🧠 配置连接参数
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # 请根据你的设置替换

# 🧪 查询图谱数据
def run_query(cypher):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        results = session.run(cypher)
        records = results.data()
    return records

# 🔄 转换为 DataFrame 格式
def parse_graph_data(records):
    if not records:
        return pd.DataFrame(columns=["source", "relation", "target"])

    # 尝试获取 source, relation, target 三列
    try:
        rows = []
        for r in records:
            rows.append([
                r.get("source", ""),
                r.get("relation", ""),
                r.get("target", "")
            ])
        df = pd.DataFrame(rows, columns=["source", "relation", "target"])
        return df
    except Exception as e:
        st.error(f"数据解析错误：{e}")
        return pd.DataFrame(columns=["source", "relation", "target"])

# 🧭 构建 NetworkX 图谱并用 PyVis 渲染
def draw_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["source"], row["target"], label=row["relation"])

    nt = Network(height="600px", width="100%", directed=True)
    nt.from_nx(G)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        nt.save_graph(tmp_file.name)
        tmp_path = tmp_file.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=650, scrolling=True)
    os.unlink(tmp_path)

# 🚀 主展示函数
def show_neo4j_graph():
    st.title("🕸️ 图谱关系可视化（Neo4j）")

    default_query = """
MATCH (n)-[r]->(m)
RETURN n.name AS source, type(r) AS relation, m.name AS target
LIMIT 100
    """.strip()

    query = st.text_area("输入Cypher查询语句：", value=default_query, height=150)

    if st.button("🔍 查询图谱"):
        try:
            with st.spinner("执行中..."):
                records = run_query(query)
                df = parse_graph_data(records)

                if df.empty:
                    st.warning("查询结果为空或格式不符合要求。请确保包含 source、relation、target 三列。")
                else:
                    st.success(f"查询成功，共有 {len(df)} 条关系。")
                    draw_graph(df)
        except Exception as e:
            st.error(f"Neo4j 查询失败：{e}")
