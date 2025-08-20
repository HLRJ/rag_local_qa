import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import CTransformers
from .config import MODEL_CONFIGS

@st.cache_resource
def load_llm(model_key: str):
    config = MODEL_CONFIGS[model_key]
    if config["type"] == "gguf":
        return CTransformers(
            model=str(config["model_path"]),
            model_type=config["model_type"],
            config={"max_new_tokens": 512, "temperature": 0.7, "gpu_layers": 15},
            repetition_penalty=1.1,
            stop=["\nUser:"],
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], trust_remote_code=True, use_fast=False,
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
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe)
