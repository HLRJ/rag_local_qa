from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "embeddings/faiss_store"
RECORD_FILE = FAISS_DIR / "record.json"
SESSIONS_FILE = BASE_DIR / "chat_sessions.json"

# 嵌入模型
EMBED_MODEL = "BAAI/bge-large-zh"

# 本地 LLM 模型配置
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

# Prompt 模板
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
