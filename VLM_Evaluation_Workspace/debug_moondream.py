import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import sys, os, time

# Auto-load variables from .env file (which is gitignored)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip('"\'')

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # Uses loaded token or blank
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
