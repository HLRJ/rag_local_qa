# 文件：scripts/build_graph_from_doc.py
import os
import json
from pathlib import Path
from typing import List, Tuple
from neo4j import GraphDatabase
import spacy
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RECORD_FILE = BASE_DIR / "embeddings/graph_store/record.json"
nlp = spacy.load("zh_core_web_sm")

def normalize_entity(entity: str) -> str:
    return entity.strip().lower()

def extract_triples_spacy(text: str) -> List[Tuple[str, str, str]]:
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        try:
            subj, verb, obj = None, None, None
            for token in sent:
                if token.dep_ in ("nsubj", "nsubj:pass"):
                    subj = token.text
                elif token.dep_ == "ROOT":
                    verb = token.text
                elif token.dep_ in ("dobj", "obj"):
                    obj = token.text
            if subj and verb and obj:
                triples.append((
                    normalize_entity(subj),
                    normalize_entity(verb),
                    normalize_entity(obj)
                ))
        except Exception as e:
            continue  # 忽略异常
    return triples


def insert_triples_to_neo4j(triples: List[Tuple[str, str, str]], filename: str):
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "12345678")
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as session:
        for h, r, t in triples:
            session.run("""
                MERGE (a:Entity {name: $h})
                MERGE (b:Entity {name: $t})
                MERGE (a)-[:REL {name: $r, source: $s}]->(b)
            """, h=h, r=r, t=t, s=filename)
    driver.close()

def load_indexed_files() -> List[str]:
    if not RECORD_FILE.exists():
        return []
    with open(RECORD_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("indexed_files", [])

def save_indexed_files(files: List[str]):
    RECORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump({"indexed_files": files}, f, ensure_ascii=False, indent=2)

def load_documents():
    docs = []
    indexed = set(load_indexed_files())
    for file in DATA_DIR.rglob("*"):
        if not file.is_file():
            continue
        relative_path = str(file.relative_to(DATA_DIR))
        if relative_path in indexed:
            continue
        try:
            if file.suffix == ".pdf":
                loader = PyMuPDFLoader(str(file))
            elif file.suffix == ".docx":
                loader = UnstructuredWordDocumentLoader(str(file))
            elif file.suffix == ".xlsx":
                loader = UnstructuredExcelLoader(str(file))
            else:
                continue
            chunks = loader.load()
            docs.append((relative_path, chunks))
        except Exception as e:
            print(f"❌ 加载失败：{file} - {e}")
    return docs


def build_graph():
    indexed = load_indexed_files()
    docs = load_documents()
    for fname, chunks in docs:
        text = "\n".join([c.page_content for c in chunks])

        # 分块处理整个文档内容，避免一次性爆内存
        chunk_size = 1000
        text_blocks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        all_triples = []
        for i, block in enumerate(text_blocks):
            triples = extract_triples_spacy(block)
            all_triples.extend(triples)

        if all_triples:
            insert_triples_to_neo4j(all_triples, fname)
            print(f"✅ 写入图谱：{fname}，共 {len(all_triples)} 条")
        else:
            print(f"⚠️ 无有效三元组：{fname}")

        indexed.append(fname)
        save_indexed_files(indexed)


if __name__ == "__main__":
    build_graph()
