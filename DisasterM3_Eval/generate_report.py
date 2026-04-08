#!/usr/bin/env python3
"""
generate_report.py
==================
Reads all eval_*.json result files and produces:
  1. A consolidated CSV table (models × metrics) for research papers.
  2. A per-task-type comparison CSV.
  3. A summary printed to console.

Usage:
    python generate_report.py
"""

import json
import os
import csv
from config import RESULTS_DIR, MODELS


def load_results(prefix="eval"):
    """Load all eval_*.json files from results/."""
    results = {}
    for model_name in MODELS:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{model_name}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                results[model_name] = json.load(f)
    return results


def generate_overall_csv(results, output_path):
    """Create a CSV with models as rows and overall metrics as columns."""
    if not results:
        print("⚠ No result files found.")
        return

    # Collect all metric keys
    all_metrics = set()
    for r in results.values():
        all_metrics.update(r.get("metrics_overall", {}).keys())
    all_metrics = sorted(all_metrics)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "samples", "errors", "inference_sec"] + all_metrics)

        for model_name, r in sorted(results.items()):
            meta = r.get("metadata", {})
            overall = r.get("metrics_overall", {})
            row = [
                model_name,
                meta.get("total_samples", 0),
                meta.get("errors", 0),
                meta.get("inference_seconds", 0),
            ]
            for m in all_metrics:
                row.append(f"{overall.get(m, 0):.4f}")
            writer.writerow(row)

    print(f"📊 Overall metrics CSV → {output_path}")


def generate_pertask_csv(results, output_path):
    """Create a CSV with one row per (model, task_type) combination."""
    if not results:
        return

    rows = []
    for model_name, r in sorted(results.items()):
        by_task = r.get("metrics_by_task", {})
        for task, metrics in sorted(by_task.items()):
            row = {
                "model": model_name,
                "task_type": task,
                "track": metrics.get("track", ""),
                "count": metrics.get("count", 0),
            }
            # Add all numeric metrics
            for k, v in metrics.items():
                if k not in ("track", "count") and isinstance(v, (int, float)):
                    row[k] = f"{v:.4f}"
            rows.append(row)

    if not rows:
        return

    # Get all columns
    columns = ["model", "task_type", "track", "count"]
    extra_cols = set()
    for row in rows:
        extra_cols.update(k for k in row if k not in columns)
    columns.extend(sorted(extra_cols))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"📊 Per-task metrics CSV → {output_path}")


def print_summary(results):
    """Print a formatted comparison table to console."""
    if not results:
        print("No results to summarize.")
        return

    print(f"\n{'='*80}")
    print(f"  DisasterM3 Benchmark — VLM Comparison ({len(results)} models)")
    print(f"{'='*80}")

    # Header
    models = sorted(results.keys())
    header = f"{'Metric':<35s}" + "".join(f"{m:>12s}" for m in models)
    print(header)
    print("─" * len(header))

    # Collect all overall metrics
    all_metrics = set()
    for r in results.values():
        all_metrics.update(r.get("metrics_overall", {}).keys())

    for metric in sorted(all_metrics):
        row = f"  {metric:<33s}"
        for m in models:
            val = results[m].get("metrics_overall", {}).get(metric, 0)
            row += f"{val:12.4f}"
        print(row)

    # Per-task breakdown
    print(f"\n{'='*80}")
    print(f"  Per-Task Breakdown:")
    print(f"{'='*80}")

    # Collect all tasks
    all_tasks = set()
    for r in results.values():
        all_tasks.update(r.get("metrics_by_task", {}).keys())

    for task in sorted(all_tasks):
        # Find the relevant metric for this task's track
        sample_track = None
        for r in results.values():
            tm = r.get("metrics_by_task", {}).get(task, {})
            if tm:
                sample_track = tm.get("track", "")
                break

        if sample_track == "single_label_mcq":
            key = "accuracy"
        elif sample_track == "multi_label_mcq":
            key = "f1"
        elif sample_track == "free_text":
            key = "rougeL"
        else:
            continue

        row = f"  {task:<33s}"
        for m in models:
            tm = results[m].get("metrics_by_task", {}).get(task, {})
            val = tm.get(key, 0)
            row += f"{val:12.4f}"
        print(f"{row}  ({key})")

    print()


def main():
    results = load_results()
    if not results:
        print("⚠ No eval_*.json files found in results/. Run run_all_test.sh first.")
        return

    overall_csv = os.path.join(RESULTS_DIR, "disasterm3_overall_comparison.csv")
    pertask_csv = os.path.join(RESULTS_DIR, "disasterm3_pertask_comparison.csv")

    generate_overall_csv(results, overall_csv)
    generate_pertask_csv(results, pertask_csv)
    print_summary(results)


if __name__ == "__main__":
    main()
