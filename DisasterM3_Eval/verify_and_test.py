#!/usr/bin/env python3
"""
verify_and_test.py
==================
Pre-flight check + mini smoke test for the DisasterM3 evaluation pipeline.

What it checks:
  1. Python version
  2. CUDA / GPU availability and VRAM
  3. All required packages installed
  4. Dataset exists and can be loaded
  5. Each VLM model can be downloaded (weights check)
  6. (Optional) Runs a 2-sample smoke test with the smallest model

Usage:
  python verify_and_test.py                    # Check everything, no inference
  python verify_and_test.py --smoke-test       # Check + run 2 samples with moondream2
  python verify_and_test.py --download-all     # Download all 5 model weights
"""

import argparse
import sys
import os
import importlib
import time

def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def check(label, ok, detail=""):
    status = "OK" if ok else "FAIL"
    print(f"  [{status:4s}] {label}")
    if detail:
        print(f"         {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run 2 samples with moondream2 after checks")
    ap.add_argument("--download-all", action="store_true",
                    help="Pre-download all 5 model weights (no inference)")
    args = ap.parse_args()

    all_ok = True

    # ── 1. Python Version ──
    section("1. Python Version")
    py_ver = sys.version
    py_ok = sys.version_info >= (3, 9)
    all_ok &= check(f"Python {py_ver.split()[0]}", py_ok,
                     "Need Python >= 3.9" if not py_ok else "")

    # ── 2. CUDA / GPU ──
    section("2. CUDA / GPU")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        all_ok &= check("PyTorch installed", True, f"v{torch.__version__}")
        all_ok &= check("CUDA available", cuda_ok)
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            check(f"GPU: {gpu_name}", True,
                  f"{free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
            if free / 1e9 < 3.0:
                check("VRAM warning", False, "Less than 3 GB free — close other GPU processes")
                all_ok = False
        else:
            check("No GPU detected", False, "VLM inference requires CUDA GPU")
            all_ok = False
    except ImportError:
        check("PyTorch", False, "Not installed. Run setup_env.ps1 first.")
        all_ok = False

    # ── 3. Required Packages ──
    section("3. Required Packages")
    packages = {
        "transformers": "transformers",
        "accelerate": "accelerate",
        "PIL": "Pillow",
        "rouge_score": "rouge-score",
        "nltk": "nltk",
        "qwen_vl_utils": "qwen-vl-utils",
        "tqdm": "tqdm",
        "dotenv": "python-dotenv",
        "einops": "einops",
        "sentencepiece": "sentencepiece",
    }
    for mod_name, pip_name in packages.items():
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "ok")
            check(f"{pip_name}", True, f"v{ver}")
        except ImportError:
            check(f"{pip_name}", False, f"pip install {pip_name}")
            all_ok = False

    # Check transformers version specifically
    try:
        import transformers
        tv = transformers.__version__
        tv_ok = tuple(int(x) for x in tv.split(".")[:2]) >= (4, 49)
        if not tv_ok:
            check("transformers version", False,
                  f"v{tv} is too old. Need >= 4.49.0 for Qwen2.5-VL")
            all_ok = False
    except Exception:
        pass

    # ── 4. Dataset ──
    section("4. Dataset")
    try:
        from config import BENCH_JSON, RAW_DATA_DIR
        check("config.py loaded", True)
        check(f"RAW_DATA_DIR exists", os.path.isdir(RAW_DATA_DIR), RAW_DATA_DIR)
        check(f"benchmark_release.json exists", os.path.isfile(BENCH_JSON), BENCH_JSON)

        if os.path.isfile(BENCH_JSON):
            import json
            with open(BENCH_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            check(f"Dataset entries", True, f"{len(data)} total entries")

            # Check a few images exist
            imgs_found = 0
            imgs_checked = 0
            for entry in data[:20]:
                for key in ["pre_image_path", "post_image_path", "image_path"]:
                    rel = entry.get(key, "")
                    if rel:
                        rel = rel.replace("\\", "/")
                        full = os.path.join(RAW_DATA_DIR, rel)
                        imgs_checked += 1
                        if os.path.exists(full):
                            imgs_found += 1
            check(f"Image files accessible", imgs_found > 0,
                  f"{imgs_found}/{imgs_checked} images found in first 20 entries")
            if imgs_found == 0:
                all_ok = False
        else:
            all_ok = False
    except Exception as e:
        check("Dataset loading", False, str(e))
        all_ok = False

    # ── 5. NLTK Data (for METEOR) ──
    section("5. NLTK Data")
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score as ms
        test_score = ms([["hello", "world"]], ["hello", "earth"])
        check("METEOR scoring", True, f"test score = {test_score:.3f}")
    except Exception as e:
        check("METEOR scoring", False,
              f"{e} — run: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")

    # ── 6. Model Weights ──
    section("6. Model Weights")
    from config import MODELS

    if args.download_all:
        print("  Downloading all model weights (this may take a while)...")
        for key, hf_id in MODELS.items():
            print(f"\n  Downloading: {key} ({hf_id})")
            try:
                from huggingface_hub import snapshot_download
                token = os.environ.get("HF_TOKEN", None)
                if key == "moondream2":
                    snapshot_download(hf_id, revision="2024-08-26", token=token)
                else:
                    snapshot_download(hf_id, token=token)
                check(f"{key} downloaded", True)
            except Exception as e:
                check(f"{key} download", False, str(e)[:100])
    else:
        print("  (Use --download-all to pre-download model weights)")
        for key, hf_id in MODELS.items():
            try:
                from huggingface_hub import try_to_load_from_cache
                # Just check if config.json exists in cache
                cached = try_to_load_from_cache(hf_id, "config.json")
                if cached and os.path.exists(str(cached)):
                    check(f"{key}", True, "cached")
                else:
                    check(f"{key}", False, "not cached — will download on first run")
            except Exception:
                check(f"{key}", False, "not cached — will download on first run")

    # ── 7. Smoke Test ──
    if args.smoke_test:
        section("7. Smoke Test (2 samples with moondream2)")
        try:
            from dataset_loader import load_disasterm3_bench
            from vlm_registry import load_vlm, ask_vlm, unload_model
            from prompt_templates import get_prompt_and_images
            from evaluation import evaluate_sample

            data = load_disasterm3_bench(max_samples=2, stratified=True)
            check("Loaded 2 stratified samples", len(data) >= 1)

            model, proc = load_vlm("moondream2", MODELS["moondream2"])
            check("Moondream2 loaded", True)

            for i, item in enumerate(data):
                prompted = get_prompt_and_images(item)
                pred = ask_vlm(
                    model, proc,
                    prompt_text=prompted["prompt_text"],
                    image_paths=prompted["image_paths"],
                    needs_dual_image=prompted["needs_dual_image"],
                    model_key="moondream2",
                    max_new_tokens=64,
                )
                print(f"\n  Sample {i + 1}:")
                print(f"    Task:  {item['task_type']}")
                print(f"    Pred:  {pred[:150]}")
                print(f"    GT:    {item['answer'][:100]}")
                check(f"Inference sample {i + 1}", len(pred) > 0)

            unload_model(model, proc)
            check("Smoke test passed", True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check("Smoke test", False, str(e)[:200])
            all_ok = False
    else:
        print("\n  (Use --smoke-test to run 2 samples with moondream2)")

    # ── Summary ──
    section("SUMMARY")
    if all_ok:
        print("  All checks passed! You are ready to run.\n")
        print("  Quick test (50 samples, 1 model):")
        print("    python run_test.py --model moondream2 --max_samples 50 --stratified\n")
        print("  Full dataset (all samples, 1 model):")
        print("    python run_test.py --model moondream2 --stratified\n")
        print("  All 5 models:")
        print("    powershell -ExecutionPolicy Bypass -File run_all_models.ps1\n")
    else:
        print("  Some checks FAILED. Fix the issues above before running.\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
