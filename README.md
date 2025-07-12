# 运维智能问答平台（RAG + 中文本地模型）

## 🧠 功能
- 多模型中文LLM问答（MiniCPM、Alpaca、Yi、BGE）
- 中文向量化检索（BGE-small）
- Word/PDF/Excel上传构建知识库
- Streamlit交互界面

## 🚀 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 构建向量库
python scripts/build_vector_store.py

# 启动问答界面
python scripts/run_web_ui.py
```

## 📂 模型下载建议（GGUF量化版）
- MiniCPM: [Q4_K_M](https://huggingface.co/openbmb/MiniCPM-2B-dpo-GGUF)
- Chinese-Alpaca: [Q4_0](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b-GGUF)
- Yi-6B: [Q4_K_M](https://huggingface.co/01-ai/Yi-6B-Chat-GGUF)
- BGE-Llama: [Q4_K_M](https://huggingface.co/FlagOpen/flagembedding-llama2-zh-GGUF)

将模型下载放入 `models/` 目录下。