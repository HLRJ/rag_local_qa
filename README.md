# 运维智能问答平台（RAG + 中文本地模型）

## 🧠 功能
- 多模型中文LLM问答（Qwen、Baichuan）
- 中文向量化检索（BAAI/bge-large-zh）
- Word/PDF/Excel上传构建知识库
- Streamlit交互界面

## 🧰 环境要求
需要安装(neo4j数据库)[https://neo4j.com/]
已在win10/nvidia geforce rtx3060laptop成功运行，如需迁移至linux系统，请注意修改目录路径的斜线

## 🚀 快速开始
```bash
# windows
cd \path\to\rag_local_qa  # 进入这个项目目录下
set PYTHONPATH=.


#linux/macos
cd /path/to/rag_local_qa  # 进入这个项目目录下
PYTHONPATH=. 



# 安装依赖
pip install -r requirements.txt
python -m spacy download zh_core_web_sm

# 构建知识图谱
python scripts/build_graph_from_doc.py

# 构建向量库
python scripts/build_vector_store.py

# 启动问答界面
python scripts/run_web_ui.py
```

## 📦 模型下载建议（支持GGUF量化版和safetensors格式）
- Qwen: [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main)
- THUDM: [glm-edge-1.5b-chat](https://huggingface.co/THUDM/glm-edge-1.5b-chat/tree/main)
- openbmb: [MiniCPM4-0.5B](https://huggingface.co/openbmb/MiniCPM4-0.5B/tree/main)
- TinyLlama: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main)
- [llama-2-7b.Q4_K_M](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/tree/main)

✅ 下载后统一放置到 models/ 目录下对应子文件夹中。

## 📁 项目目录结构

```text
│  chat_history.json
│  README.md
│  requirements.txt
│  test.py
│
│
├─data
│      比如我这里放的是Linux常用命令手册.pdf    
│
├─embeddings
│  └─faiss_store
│          index.faiss
│          index.pkl
│          record.json
│
├─models
│  ├─llama
│  │      llama-2-7b.Q4_K_M.gguf
│  │
│  ├─openbmb
│  │  └─MiniCPM4-0.5B
│  │          added_tokens.json
│  │          config.json
│  │          configuration_minicpm.py
│  │          generation_config.json
│  │          model.safetensors
│  │          modeling_minicpm.py
│  │          special_tokens_map.json
│  │          tokenizer.json
│  │          tokenizer.model
│  │          tokenizer_config.json
│  │
│  ├─Qwen
│  │  └─Qwen1.5-1.8B
│  │          config.json
│  │          generation_config.json
│  │          merges.txt
│  │          model.safetensors
│  │          tokenizer.json
│  │          tokenizer_config.json
│  │          vocab.json
│  │
│  ├─THUDM
│  │  └─glm-edge-1.5b-chat
│  │          config.json
│  │          generation_config.json
│  │          model.safetensors
│  │          special_tokens_map.json
│  │          tokenizer.json
│  │          tokenizer_config.json
│  │
│  └─TinyLlama
│      └─TinyLlama-1.1B-Chat-v1.0
│              config.json
│              eval_results.json
│              generation_config.json
│              model.safetensors
│              special_tokens_map.json
│              tokenizer.json
│              tokenizer.model
│              tokenizer_config.json
│
├─scripts
│      build_vector_store.py
│      query_rag.py
│      query_rag_mixed.py
│      remove_doc.py
│      run_web_ui.py
│
└─tools
        read_gguf_header.py
        test_ctransformers.py

```

## 🔍 实践
- ❗ Baichuan2-7B-Chat、Yi-1.5-6B-Chat在移动版3060爆显存
- 🔧 有些依赖必须在Linux环境下，Windows系统无法安装
- ❌ MiniCPM-2B-sft-bf16、MiniCPM3-4B-GGUF简单调试后发现无法适配
- ⚙️ llama-2-7b.Q4_K_M模型推理如果设置"gpu_layers"参数的话，可能会导致回答效果降低
- ✅ 综合对比还是qwen效果好