import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import sys, os, time
os.environ["HF_TOKEN"] = ""
import transformers
if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
    transformers.PreTrainedModel.all_tied_weights_keys = {}

model_id = "vikhyatk/moondream2"
print("Loading model...")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision="2024-08-26")
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16, revision="2024-08-26"
).to("cuda")

from datasets_loader import load_rsvlmqa_data
data = load_rsvlmqa_data(max_samples=2)

for i, item in enumerate(data):
    img = Image.open(item["image_path"]).convert("RGB")
    question = item["question"]
    print(f"\n--- Q{i+1} ---")
    print(f"Q: {question}")
    
    enc = model.encode_image(img)
    answer = model.answer_question(enc, question, tok)
    print(f"A: {answer}")
