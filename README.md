# 运维智能问答平台（RAG + 中文本地模型）

## 🧠 功能
- 多模型中文LLM问答（Qwen、Baichuan）
- 中文向量化检索（BAAI/bge-large-zh）
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
- THUDM: [glm-edge-1.5b-chat](https://huggingface.co/THUDM/glm-edge-1.5b-chat/tree/main)
- openbmb: [MiniCPM4-0.5B](https://huggingface.co/openbmb/MiniCPM4-0.5B/tree/main)
- TinyLlama: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main)
- [llama-2-7b.Q4_K_M](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/tree/main)

将模型下载放入 `models/` 目录下。

## 实践
- Baichuan2-7B-Chat、Yi-1.5-6B-Chat在移动版3060爆显存
- 有些依赖必须在Linux环境下，Windows系统无法安装
- MiniCPM-2B-sft-bf16、MiniCPM3-4B-GGUF简单调试后发现无法适配
- llama-2-7b.Q4_K_M模型推理如果设置"gpu_layers"参数的话，可能会导致回答效果降低
- 综合对比还是qwen效果好