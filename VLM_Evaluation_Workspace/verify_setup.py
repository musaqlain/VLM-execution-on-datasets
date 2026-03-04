#!/usr/bin/env python3
"""
verify_setup.py
===============
Pre-flight check: tests EVERY dependency, import, dataset path, and
does a single-image inference with the smallest model (moondream2).

Run this BEFORE the overnight job:
    python verify_setup.py

If this script prints  ✅ ALL CHECKS PASSED  you are safe to launch run_all.sh.
"""

import sys, os, time
os.environ["HF_TOKEN"] = ""
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

PASS = 0
FAIL = 0

def check(label, fn):
    global PASS, FAIL
    try:
        result = fn()
        print(f"  ✅  {label}")
        PASS += 1
        return result
    except Exception as e:
        print(f"  ❌  {label}  →  {e}")
        FAIL += 1
        return None


# ── 1. Python version ────────────────────────
print("\n1. PYTHON")
check("Python >= 3.9", lambda: None if sys.version_info >= (3, 9) else (_ for _ in ()).throw(RuntimeError(f"Got {sys.version}")))

# ── 2. Core libraries ───────────────────────
print("\n2. LIBRARIES")
check("torch",        lambda: __import__("torch"))
check("transformers", lambda: __import__("transformers"))
check("PIL (Pillow)", lambda: __import__("PIL"))
check("nltk",         lambda: __import__("nltk"))
check("rouge_score",  lambda: __import__("rouge_score"))
check("accelerate",   lambda: __import__("accelerate"))
check("bitsandbytes", lambda: __import__("bitsandbytes"))

# ── 3. Transformers model classes ────────────
print("\n3. MODEL CLASSES (transformers v5 compatibility)")
check("AutoModelForCausalLM",         lambda: getattr(__import__("transformers"), "AutoModelForCausalLM"))
check("AutoProcessor",                lambda: getattr(__import__("transformers"), "AutoProcessor"))
check("AutoTokenizer",                lambda: getattr(__import__("transformers"), "AutoTokenizer"))
check("Idefics2ForConditionalGeneration", lambda: getattr(__import__("transformers"), "Idefics2ForConditionalGeneration"))
check("LlavaForConditionalGeneration", lambda: getattr(__import__("transformers"), "LlavaForConditionalGeneration"))

# Check for BLIP model classes (changed in transformers v5)
def check_blip_imports():
    try:
        from transformers import AutoModelForVision2Seq
        return "AutoModelForVision2Seq"
    except ImportError:
        pass
    try:
        from transformers import Blip2ForConditionalGeneration
        return "Blip2ForConditionalGeneration"
    except ImportError:
        pass
    raise ImportError("Neither AutoModelForVision2Seq nor Blip2ForConditionalGeneration found!")

blip_class = check("BLIP-2 model class", check_blip_imports)
if blip_class:
    print(f"       → Will use: {blip_class}")

# ── 4. GPU ───────────────────────────────────
print("\n4. GPU")
import torch
check("CUDA available", lambda: None if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError("No CUDA")))
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"       → {name}  ({mem:.1f} GB)")

# ── 5. Dataset paths ────────────────────────
print("\n5. DATASETS")
BASE = "/home/aipmu/Datasets for VLM/Raw dataset files"

check("RSVLM-QA.jsonl exists",
      lambda: None if os.path.exists(f"{BASE}/RSVLM-QA.jsonl") else (_ for _ in ()).throw(FileNotFoundError()))
check("RSVLM-QA image dir exists",
      lambda: None if os.path.isdir(f"{BASE}/RSVLM-QA") else (_ for _ in ()).throw(FileNotFoundError()))
check("DisasterM3 questions exist",
      lambda: None if os.path.exists(f"{BASE}/DisasterM3/vqa_format/all_questions.json") else (_ for _ in ()).throw(FileNotFoundError()))
check("DisasterM3 answers exist",
      lambda: None if os.path.exists(f"{BASE}/DisasterM3/vqa_format/all_answers.json") else (_ for _ in ()).throw(FileNotFoundError()))
check("DisasterM3 train_images dir",
      lambda: None if os.path.isdir(f"{BASE}/DisasterM3/train_images") else (_ for _ in ()).throw(FileNotFoundError()))

# Count a sample image
rsvlm_sample = None
for subdir in ["INRIA-Aerial-Image-Labeling", "LoveDA", "WHU", "iSAID"]:
    candidate = os.path.join(BASE, "RSVLM-QA", subdir)
    if os.path.isdir(candidate):
        # walk to find first image
        for root, dirs, files in os.walk(candidate):
            for f in files:
                if f.endswith(('.tif', '.png', '.jpg')):
                    rsvlm_sample = os.path.join(root, f)
                    break
            if rsvlm_sample:
                break
    if rsvlm_sample:
        break

if rsvlm_sample:
    print(f"       → Sample RSVLM-QA image: {os.path.basename(rsvlm_sample)}")

dm3_images = os.listdir(f"{BASE}/DisasterM3/train_images")[:3]
print(f"       → Sample DisasterM3 images: {dm3_images}")

# ── 6. Data loader test ─────────────────────
print("\n6. DATA LOADERS")
from datasets_loader import load_rsvlmqa_data, load_disasterm3_data
rsvlm_data = check("Load 3 RSVLM-QA samples", lambda: load_rsvlmqa_data(max_samples=3))
dm3_data   = check("Load 3 DisasterM3 samples", lambda: load_disasterm3_data(max_samples=3))

if rsvlm_data:
    s = rsvlm_data[0]
    print(f"       → Q: {s['question'][:70]}…")
    print(f"       → A: {s['answer'][:70]}…")
if dm3_data:
    s = dm3_data[0]
    print(f"       → Q: {s['question'][:70]}…")
    print(f"       → A: {s['answer'][:70]}…")

# ── 7. Evaluation test ──────────────────────
print("\n7. EVALUATION METRICS")
from evaluation import evaluate_predictions
dummy = [{"prediction": "yes", "ground_truth": "yes"},
         {"prediction": "flooding detected", "ground_truth": "flooding"}]
metrics = check("Compute metrics on dummy data", lambda: evaluate_predictions(dummy))
if metrics:
    print(f"       → {metrics}")

# ── 8. Model inference (moondream2) ──────────
print("\n8. SINGLE-IMAGE INFERENCE TEST  (moondream2)")
if rsvlm_data and torch.cuda.is_available():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        # Patch for moondream2 + transformers v5 compat
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            transformers.PreTrainedModel.all_tied_weights_keys = {}
        print("  ⏳ Downloading moondream2 (~3.6 GB) …")
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True,
            torch_dtype=torch.float16, revision="2024-08-26").to("cuda")
        tok = AutoTokenizer.from_pretrained("vikhyatk/moondream2", trust_remote_code=True, revision="2024-08-26")

        from PIL import Image
        img = Image.open(rsvlm_data[0]["image_path"]).convert("RGB")
        question = rsvlm_data[0]["question"]

        print(f"  ⏳ Asking: \"{question[:60]}…\"")
        t0 = time.time()
        enc = model.encode_image(img)
        answer = model.answer_question(enc, question, tok)
        elapsed = time.time() - t0

        print(f"  ✅  Model answered in {elapsed:.1f}s: \"{answer[:80]}\"")
        PASS += 1

        # Clean up
        del model, tok, enc
        import gc; gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ❌  Inference failed: {e}")
        FAIL += 1
else:
    print("  ⚠  Skipped (no data or no GPU)")

# ── Summary ──────────────────────────────────
print("\n" + "=" * 55)
if FAIL == 0:
    print(f"  ✅  ALL {PASS} CHECKS PASSED — safe to run overnight!")
else:
    print(f"  ⚠  {PASS} passed, {FAIL} FAILED — fix before running!")
print("=" * 55)
