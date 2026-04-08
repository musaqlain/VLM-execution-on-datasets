#!/usr/bin/env python3
"""Inspect the DisasterM3_Bench benchmark JSON schema."""
import json, os

BENCH_JSON = "/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Bench/benchmark_release.json"

with open(BENCH_JSON) as f:
    data = json.load(f)

print(f"Type: {type(data).__name__}")
if isinstance(data, list):
    print(f"Total entries: {len(data)}")
    if data:
        print(f"First entry keys: {list(data[0].keys())}")
        print("\n--- FIRST 3 ENTRIES ---")
        for i, entry in enumerate(data[:3]):
            print(f"\n[Entry {i}]")
            print(json.dumps(entry, indent=2, ensure_ascii=False))
        
        # Check for unique tasks/types
        tasks = set()
        has_pre = 0
        has_post = 0
        has_options = 0
        for e in data:
            if 'task' in e: tasks.add(e['task'])
            if 'pre_image_path' in e: has_pre += 1
            if 'post_image_path' in e: has_post += 1
            if 'options' in e: has_options += 1
        
        print(f"\n--- STATISTICS ---")
        print(f"Entries with pre_image_path: {has_pre}/{len(data)}")
        print(f"Entries with post_image_path: {has_post}/{len(data)}")
        print(f"Entries with options: {has_options}/{len(data)}")
        print(f"\nUnique tasks ({len(tasks)}):")
        for t in sorted(tasks):
            count = sum(1 for e in data if e.get('task') == t)
            print(f"  {t}: {count}")

elif isinstance(data, dict):
    print(f"Top-level keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
            if v and isinstance(v[0], dict):
                print(f"    First item keys: {list(v[0].keys())}")
                print(json.dumps(v[0], indent=2, ensure_ascii=False))
        else:
            print(f"  {k}: {type(v).__name__} = {str(v)[:200]}")
