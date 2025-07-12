from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    r"/models/llama/llama-2-7b.Q2_K.gguf",
    model_type="llama",
    gpu_layers=1000,
    max_new_tokens=20
)

response = llm("请用中文回答：你是谁？")
print("模型回答:", response)
