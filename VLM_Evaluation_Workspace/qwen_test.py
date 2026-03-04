import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen-VL-Chat', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)

print("Preparing input...")
query = tok.from_list_format([
    {'text': 'Hello, who are you?'},
])
response, _ = model.chat(tok, query=query, history=None)
print("Response:", response)
