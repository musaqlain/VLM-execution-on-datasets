#!/usr/bin/env python3
"""
run_test.py
===========
Run a single VLM on N samples from DisasterM3_Bench with task-specific prompts.

Usage:
    python run_test.py --model moondream2 --max_samples 50 --stratified
    python run_test.py --model qwen2.5-vl-7b --max_samples 10
"""

import argparse
import json
import os
import time
import sys
import traceback
from datetime import datetime, timezone

import torch

from config import MODELS, RESULTS_DIR, MAX_NEW_TOKENS
from dataset_loader import load_disasterm3_bench
from vlm_registry import load_vlm, ask_vlm, unload_model
from evaluation import evaluate_sample, evaluate_all
from prompt_templates import get_prompt_and_images


def main():
    ap = argparse.ArgumentParser(description="Run a VLM on DisasterM3 samples")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--stratified", action="store_true",
                    help="Stratified sampling across task types")
    ap.add_argument("--output_prefix", type=str, default="eval",
                    help="Prefix for output filename")
    args = ap.parse_args()

    hf_id = MODELS[args.model]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = os.path.join(RESULTS_DIR, f"{args.output_prefix}_{args.model}.json")

    # ── Environment diagnostics ──
    token = os.environ.get("HF_TOKEN", "")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    vram_total = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
    vram_free = f"{torch.cuda.mem_get_info()[0] / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"

    print(f"\n{'=' * 60}")
    print(f"  Model:      {args.model}")
    print(f"  HF ID:      {hf_id}")
    print(f"  Samples:    {args.max_samples} ({'stratified' if args.stratified else 'sequential'})")
    print(f"  HF_TOKEN:   {'set (' + token[:8] + '...)' if token else 'NOT SET'}")
    print(f"  GPU:        {gpu_name}")
    print(f"  VRAM:       {vram_free} free / {vram_total} total")
    print(f"  Max tokens: {MAX_NEW_TOKENS}")
    print(f"{'=' * 60}")

    # ── Load dataset ──
    data = load_disasterm3_bench(
        max_samples=args.max_samples,
        stratified=args.stratified,
        skip_segmentation=True,
    )
    if not data:
        print("No data loaded. Check dataset_loader and paths.")
        sys.exit(1)

    # ── Load model ──
    model, proc = load_vlm(args.model, hf_id)

    # ── Run inference with task-specific prompts ──
    predictions = []
    errors = 0
    t0 = time.time()

    for i, item in enumerate(data):
        # Build task-specific prompt (from DisasterM3 author templates)
        prompted = get_prompt_and_images(item)

        # Choose max_new_tokens based on task track
        if item["eval_track"] == "free_text":
            max_tokens = MAX_NEW_TOKENS  # 256 for free-text (reports, recovery advice)
        else:
            max_tokens = 64  # MCQ answers are 1-5 tokens — saves time

        # Run inference
        try:
            pred_raw = ask_vlm(
                model, proc,
                prompt_text=prompted["prompt_text"],
                image_paths=prompted["image_paths"],
                needs_dual_image=prompted["needs_dual_image"],
                model_key=args.model,
                max_new_tokens=max_tokens,
            )
        except Exception as e:
            pred_raw = f"[ERROR: {e}]"
            errors += 1
            traceback.print_exc()

        # Per-sample evaluation
        num_options = len(item["options_list"]) if item["options_list"] else 26
        per_sample = evaluate_sample(
            prediction_raw=pred_raw,
            ground_truth_options=item["ground_truth_options"],
            ground_truth_text=item["ground_truth_text"],
            eval_track=item["eval_track"],
            options_count=num_options,
        )

        record = {
            "sample_id": item["sample_id"],
            "task_type": item["task_type"],
            "eval_track": item["eval_track"],
            "image_path": item["primary_image_path"],
            "image_type": item["image_type"],
            "prompt_template_used": True,
            "prompt": prompted["prompt_text"][:500],
            "needs_dual_image": prompted["needs_dual_image"],
            "num_images_provided": len(prompted["image_paths"]),
            "options": item["options_str"],
            "ground_truth_text": item["ground_truth_text"],
            "ground_truth_options": item["ground_truth_options"],
            "prediction_raw": pred_raw,
            "prediction_parsed": per_sample.get("prediction_parsed", []),
            "per_sample_metrics": {k: v for k, v in per_sample.items()
                                   if k not in ("eval_track", "prediction_parsed")},
        }
        predictions.append(record)

        # Console output
        print(f"\n--- Sample {i + 1}/{len(data)} [{item['eval_track']}] ---")
        print(f"  Image: {os.path.basename(item['primary_image_path'])}")
        print(f"  Task:  {item['task_type']}")
        print(f"  Dual:  {prompted['needs_dual_image']} ({len(prompted['image_paths'])} images)")
        print(f"  Pred:  {pred_raw[:200]}")
        if item['eval_track'] in ('single_label_mcq', 'multi_label_mcq'):
            print(f"  Parsed: {per_sample.get('prediction_parsed', [])}")
            print(f"  GT:     {item['ground_truth_options']}")
        else:
            print(f"  GT:    {item['answer'][:120]}")
        # Show key metric
        m = record["per_sample_metrics"]
        if "accuracy" in m:
            print(f"  -> accuracy: {m['accuracy']:.1f}")
        elif "f1" in m:
            print(f"  -> P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
        elif "rougeL" in m:
            print(f"  -> BLEU1={m['bleu1']:.3f} ROUGE-L={m['rougeL']:.3f} METEOR={m.get('meteor', 0):.3f}")

    elapsed = time.time() - t0
    print(f"\n  Inference done: {len(predictions)} samples in {elapsed:.1f}s ({errors} errors)")

    # ── Aggregate metrics ──
    for p in predictions:
        p["eval_track"] = p.get("eval_track", "unknown")
        p["task_type"] = p.get("task_type", "unknown")

    agg = evaluate_all(predictions)

    # ── Print summary ──
    print(f"\n{'_' * 50}")
    print(f"  Overall Metrics for {args.model}:")
    print(f"{'_' * 50}")
    for k, v in agg["overall"].items():
        print(f"  {k:35s}: {v:.4f}")
    print(f"\n  Per-Task Breakdown:")
    for task, tm in agg["by_task"].items():
        track = tm["track"]
        n = tm["count"]
        if track == "single_label_mcq":
            print(f"    {task:40s} [{n:3d}] acc={tm['accuracy']:.3f}")
        elif track == "multi_label_mcq":
            print(f"    {task:40s} [{n:3d}] P={tm['precision']:.3f} R={tm['recall']:.3f} F1={tm['f1']:.3f}")
        elif track == "free_text":
            print(f"    {task:40s} [{n:3d}] B1={tm['bleu1']:.3f} RL={tm['rougeL']:.3f} MET={tm.get('meteor', 0):.3f}")

    # ── Save structured JSON ──
    output = {
        "metadata": {
            "model_name": args.model,
            "hf_id": hf_id,
            "dataset": "DisasterM3_Bench",
            "total_samples": len(predictions),
            "errors": errors,
            "sampling": "stratified" if args.stratified else "sequential",
            "prompt_templates": "DisasterM3_author_templates",
            "timestamp": timestamp,
            "gpu": gpu_name,
            "inference_seconds": round(elapsed, 1),
        },
        "metrics_overall": agg["overall"],
        "metrics_by_task": agg["by_task"],
        "predictions": predictions,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved -> {out_file}")

    # ── Cleanup VRAM ──
    unload_model(model, proc)


if __name__ == "__main__":
    main()
