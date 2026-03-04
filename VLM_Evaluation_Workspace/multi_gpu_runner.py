#!/usr/bin/env python3
"""
multi_gpu_runner.py
===================
Orchestrator for running VLM evaluations across multiple GPUs in parallel.

Designed for enterprise GPU servers (e.g. 16× NVIDIA V100-32GB).
Each (model, dataset) combination runs as an isolated subprocess with its
own CUDA_VISIBLE_DEVICES assignment, ensuring even GPU utilisation.

Usage
-----
  # Full run (all 8 models × 4 datasets = 32 jobs across 16 GPUs)
  python multi_gpu_runner.py --max_samples 5000

  # Quick test (10 samples, auto-detect GPUs)
  python multi_gpu_runner.py --max_samples 10

  # Specify GPU count manually
  python multi_gpu_runner.py --num_gpus 16 --max_samples 5000

  # Run only specific datasets or models
  python multi_gpu_runner.py --datasets rsvlmqa disasterm3 --models moondream2 blip2-opt-2.7b

  # Run with nohup for overnight execution
  nohup python multi_gpu_runner.py --max_samples 5000 > master_log.txt 2>&1 &
"""

import argparse
import os
import subprocess
import sys
import time
import json
from datetime import datetime, timedelta
from collections import deque


# ── Configuration ───────────────────────────
MODELS = [
    "moondream2",
    "blip2-opt-2.7b",
    "llava-1.5-7b",
    "qwen-vl-chat",
    "instructblip-vicuna",
    "idefics2-8b",
    "internvl2-4b",
    "llava-next-llama3",
]

DATASET_SCRIPTS = {
    "rsvlmqa":     "run_rsvlmqa.py",
    "disasterm3":  "run_disasterm3.py",
    "rsvqa_hr":    "run_rsvqa_hr.py",
    "earthvqa":    "run_earthvqa.py",
}


def detect_gpu_count():
    """Auto-detect the number of available NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split("\n"))
            return gpu_count
    except Exception:
        pass
    return 1


def build_job_list(models, datasets, max_samples, results_dir):
    """Create the list of jobs to run. Each job = (dataset, model, command)."""
    jobs = []
    workspace = os.path.dirname(os.path.abspath(__file__))

    for dataset_name in datasets:
        script = DATASET_SCRIPTS[dataset_name]
        script_path = os.path.join(workspace, script)

        if not os.path.exists(script_path):
            print(f"  ⚠  Script not found: {script_path}, skipping {dataset_name}")
            continue

        for model in models:
            cmd = [
                sys.executable, script_path,
                "--model", model,
                "--results_dir", results_dir,
            ]
            if max_samples is not None:
                cmd.extend(["--max_samples", str(max_samples)])

            jobs.append({
                "dataset": dataset_name,
                "model": model,
                "cmd": cmd,
                "status": "pending",
                "gpu_id": None,
                "process": None,
                "log_file": None,
                "start_time": None,
                "end_time": None,
            })

    return jobs


def run_jobs_parallel(jobs, num_gpus, results_dir):
    """
    Execute jobs across GPUs using subprocess isolation.

    Strategy:
    - Maintain a pool of available GPU IDs [0, 1, ..., num_gpus-1]
    - When a GPU becomes free, assign the next pending job to it
    - Each job gets CUDA_VISIBLE_DEVICES=<gpu_id> so the model loads on exactly one GPU
    - Logs are saved per-job for debugging
    """
    available_gpus = deque(range(num_gpus))
    pending = deque([i for i in range(len(jobs))])
    running = {}  # gpu_id -> job_index
    completed = 0
    failed = 0
    total = len(jobs)

    print(f"\n{'=' * 70}")
    print(f"  MULTI-GPU VLM BENCHMARK")
    print(f"  GPUs: {num_gpus}  |  Jobs: {total}  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}\n")

    os.makedirs(results_dir, exist_ok=True)

    while completed + failed < total:
        # Launch jobs on available GPUs
        while available_gpus and pending:
            gpu_id = available_gpus.popleft()
            job_idx = pending.popleft()
            job = jobs[job_idx]

            # Set up environment with specific GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Log file for this job
            log_name = f"log_{job['dataset']}_{job['model']}.txt"
            log_path = os.path.join(results_dir, log_name)
            log_fh = open(log_path, "w")

            print(f"  🚀 GPU {gpu_id:2d}  ←  {job['dataset']:12s} × {job['model']:22s}  "
                  f"[{completed + failed + len(running) + 1}/{total}]")

            proc = subprocess.Popen(
                job["cmd"],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

            job["status"] = "running"
            job["gpu_id"] = gpu_id
            job["process"] = proc
            job["log_file"] = log_fh
            job["start_time"] = time.time()
            running[gpu_id] = job_idx

        # Poll running jobs for completion
        finished_gpus = []
        for gpu_id, job_idx in running.items():
            job = jobs[job_idx]
            ret = job["process"].poll()
            if ret is not None:
                # Job finished
                job["end_time"] = time.time()
                elapsed = job["end_time"] - job["start_time"]
                job["log_file"].close()

                if ret == 0:
                    job["status"] = "done"
                    completed += 1
                    emoji = "✅"
                else:
                    job["status"] = f"failed (exit={ret})"
                    failed += 1
                    emoji = "❌"

                print(f"  {emoji} GPU {gpu_id:2d}  →  {job['dataset']:12s} × {job['model']:22s}  "
                      f"{elapsed/60:.1f} min  [{completed + failed}/{total}]")

                finished_gpus.append(gpu_id)

        # Return finished GPUs to the pool
        for gpu_id in finished_gpus:
            del running[gpu_id]
            available_gpus.append(gpu_id)

        # Brief sleep to avoid busy-waiting
        if not finished_gpus:
            time.sleep(5)

    return completed, failed


def print_summary(jobs, total_time):
    """Print a summary table of all jobs."""
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE  •  Total time: {total_time/3600:.1f} hours")
    print(f"{'=' * 70}")
    print(f"\n  {'Dataset':<14s}  {'Model':<24s}  {'Status':<16s}  {'Time':>8s}  {'GPU':>4s}")
    print(f"  {'─'*14}  {'─'*24}  {'─'*16}  {'─'*8}  {'─'*4}")

    for job in jobs:
        elapsed = ""
        if job["start_time"] and job["end_time"]:
            elapsed = f"{(job['end_time'] - job['start_time'])/60:.0f}m"
        gpu = str(job["gpu_id"]) if job["gpu_id"] is not None else "-"
        status = job["status"]

        print(f"  {job['dataset']:<14s}  {job['model']:<24s}  {status:<16s}  {elapsed:>8s}  {gpu:>4s}")

    done = sum(1 for j in jobs if j["status"] == "done")
    fail = sum(1 for j in jobs if "failed" in j["status"])
    print(f"\n  Summary: {done} succeeded, {fail} failed, {len(jobs)} total")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Multi-GPU VLM evaluation orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--num_gpus", type=int, default=None,
                    help="Number of GPUs to use (default: auto-detect)")
    ap.add_argument("--max_samples", type=int, default=5000,
                    help="Max QA samples per dataset per model (default: 5000)")
    ap.add_argument("--results_dir", default="results",
                    help="Directory to save results JSON and logs")
    ap.add_argument("--models", nargs="+", default=None,
                    choices=MODELS,
                    help="Subset of models to run (default: all 8)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    choices=list(DATASET_SCRIPTS.keys()),
                    help="Subset of datasets to run (default: all 4)")
    args = ap.parse_args()

    # Determine GPU count
    num_gpus = args.num_gpus or detect_gpu_count()
    print(f"  🖥  Detected / using {num_gpus} GPU(s)")

    # Determine which models and datasets to run
    models = args.models or MODELS
    datasets = args.datasets or list(DATASET_SCRIPTS.keys())

    print(f"  📊 Datasets: {', '.join(datasets)}")
    print(f"  🤖 Models:   {', '.join(models)}")
    print(f"  📏 Max samples per dataset: {args.max_samples}")

    # Build job list
    jobs = build_job_list(models, datasets, args.max_samples, args.results_dir)
    if not jobs:
        print("❌ No jobs to run!"); sys.exit(1)

    print(f"  📋 Total jobs: {len(jobs)}  "
          f"({len(datasets)} datasets × {len(models)} models)")

    # Determine parallel capacity
    # Each ~7B model uses ~14-16 GB VRAM in fp16 → fits in one V100-32GB
    effective_gpus = min(num_gpus, len(jobs))

    # Run all jobs
    t0 = time.time()
    completed, failed = run_jobs_parallel(jobs, effective_gpus, args.results_dir)
    total_time = time.time() - t0

    # Print summary
    print_summary(jobs, total_time)

    # Save run metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "num_gpus": num_gpus,
        "max_samples": args.max_samples,
        "total_time_seconds": total_time,
        "jobs_completed": completed,
        "jobs_failed": failed,
        "jobs_total": len(jobs),
        "datasets": datasets,
        "models": models,
        "job_details": [
            {
                "dataset": j["dataset"],
                "model": j["model"],
                "status": j["status"],
                "gpu_id": j["gpu_id"],
                "elapsed_seconds": (j["end_time"] - j["start_time"])
                    if j["start_time"] and j["end_time"] else None,
            }
            for j in jobs
        ],
    }
    meta_path = os.path.join(args.results_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  💾 Run metadata → {meta_path}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
