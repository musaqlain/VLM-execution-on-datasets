#!/usr/bin/env python3
"""Inspect DisasterM3 benchmark schema: task types, answer formats, field coverage."""
import json
import os
from collections import Counter, defaultdict

BENCH_JSON = "/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Bench/benchmark_release.json"

with open(BENCH_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
print(f"\n=== ALL FIELDS in first entry ===")
for k, v in data[0].items():
    print(f"  {k}: {repr(v)[:120]}")

# Task type distribution
tasks = Counter(e.get("task", "MISSING") for e in data)
print(f"\n=== TASK TYPE DISTRIBUTION ({len(tasks)} types) ===")
for task, count in tasks.most_common():
    print(f"  {count:5d}  {task}")

# Answer format analysis
print(f"\n=== ANSWER FORMAT ANALYSIS ===")
has_gt_option = sum(1 for e in data if e.get("ground_truth_option"))
has_gt        = sum(1 for e in data if e.get("ground_truth"))
has_options   = sum(1 for e in data if e.get("options"))
has_opts_str  = sum(1 for e in data if e.get("options_str"))
has_pre_img   = sum(1 for e in data if e.get("pre_image_path"))
has_post_img  = sum(1 for e in data if e.get("post_image_path"))
print(f"  ground_truth_option: {has_gt_option}/{len(data)}")
print(f"  ground_truth:        {has_gt}/{len(data)}")
print(f"  options (list):      {has_options}/{len(data)}")
print(f"  options_str:         {has_opts_str}/{len(data)}")
print(f"  pre_image_path:      {has_pre_img}/{len(data)}")
print(f"  post_image_path:     {has_post_img}/{len(data)}")

# Sample GT answers per task type (to understand multi-label vs single-label)
print(f"\n=== SAMPLE GT ANSWERS PER TASK TYPE ===")
by_task = defaultdict(list)
for e in data:
    by_task[e.get("task", "MISSING")].append(e)

for task, entries in sorted(by_task.items()):
    print(f"\n  --- {task} ({len(entries)} entries) ---")
    for e in entries[:3]:
        gt_opt = e.get("ground_truth_option", "N/A")
        gt     = e.get("ground_truth", "N/A")
        print(f"    GT_option: {repr(gt_opt)[:80]}")
        print(f"    GT_text:   {repr(gt)[:80]}")
        print(f"    Prompt:    {repr(e.get('prompts',''))[:80]}")
        print()

# Check how many answers have multiple labels (commas in ground_truth_option)
multi_label = sum(1 for e in data if "," in str(e.get("ground_truth_option", "")))
single_label = has_gt_option - multi_label
print(f"\n=== LABEL CARDINALITY ===")
print(f"  Single-label answers: {single_label}")
print(f"  Multi-label answers:  {multi_label}")
print(f"  No option answer:     {len(data) - has_gt_option}")
