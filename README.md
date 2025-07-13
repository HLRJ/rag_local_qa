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

## 📂 模型下载建议（支持GGUF量化版和safetensors格式）
- Qwen: [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main) (下载config.json、model.safetensors、tokenizer.json、tokenizer_config.json、vocab.json、merges.txt、generation_config.json文件)

将模型下载放入 `models/` 目录下。