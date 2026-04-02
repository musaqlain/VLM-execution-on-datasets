#!/usr/bin/env python3
"""
predownload_models.py
=====================
Downloads all 8 VLM models SEQUENTIALLY into ~/.cache/huggingface/hub.

WHY THIS EXISTS:
  The parallel runner (multi_gpu_runner.py) launches 32 jobs at once.
  If models are not cached, those 32 jobs all try to download simultaneously
  → HuggingFace rate-limits + network congestion + cache file corruption.

  Run this ONCE, BEFORE run_all.sh, to pre-cache everything. After that,
  the parallel runner works instantly (no downloads needed).

HOW TO RUN:
  python -u predownload_models.py 2>&1 | tee predownload_log.txt

ESTIMATED TIME:
  ~30-60 min total depending on internet speed (models range 3-17 GB each).
"""

import sys, os, time, gc
import torch

MODELS = [
    ("moondream2",           "vikhyatk/moondream2",                   "2024-08-26"),
    ("blip2-opt-2.7b",       "Salesforce/blip2-opt-2.7b",             None),
    ("llava-1.5-7b",         "llava-hf/llava-1.5-7b-hf",             None),
    ("qwen-vl-chat",         "Qwen/Qwen-VL-Chat",                     None),
    ("instructblip-vicuna",  "Salesforce/instructblip-vicuna-7b",     None),
    ("idefics2-8b",          "HuggingFaceM4/idefics2-8b",             None),
    ("internvl2-4b",         "OpenGVLab/InternVL2-4B",                None),
    ("llava-next-llama3",    "llava-hf/llama3-llava-next-8b-hf",      None),
]

def download_model(name, hf_id, revision=None):
    """Download a model to HF cache without loading it to GPU."""
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

    print(f"\n[{name}] Downloading {hf_id} (revision={revision or 'latest'}) ...")
    t0 = time.time()

    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision

    try:
        # Download processor/tokenizer first (small, fast)
        try:
            proc = AutoProcessor.from_pretrained(hf_id, use_fast=False, **kwargs)
        except Exception:
            try:
                proc = AutoTokenizer.from_pretrained(hf_id, **kwargs)
            except Exception as e:
                print(f"  [WARN] Processor/tokenizer download failed: {e}")

        # Download model weights to CPU (no GPU needed for caching)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="cpu",        # CPU only — avoids VRAM issues
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # stream weights, don't load all at once
            **kwargs,
        )

        elapsed = time.time() - t0
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  [OK] Downloaded {name} ({param_count:.1f}B params) in {elapsed/60:.1f} min")

        # Free memory immediately
        del model, proc
        gc.collect()

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {name} after {elapsed/60:.1f} min: {e}")


def main():
    print("="*60)
    print("  VLM MODEL PRE-DOWNLOAD")
    print(f"  Models: {len(MODELS)}  |  Device: CPU (cache only)")
    print(f"  HF Cache: {os.path.expanduser('~/.cache/huggingface/hub')}")
    print("="*60)

    t_total = time.time()
    for idx, (name, hf_id, revision) in enumerate(MODELS, 1):
        print(f"\n[{idx}/{len(MODELS)}] {name}")
        download_model(name, hf_id, revision)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  DONE — All models cached in {total_min:.1f} min")
    print(f"  You can now run: bash run_all.sh")
    print("="*60)


if __name__ == "__main__":
    main()
