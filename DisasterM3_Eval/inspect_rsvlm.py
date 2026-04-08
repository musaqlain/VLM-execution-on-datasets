#!/usr/bin/env python3
"""Inspect RSVLM-QA annotation files to understand schema."""
import json
import os

BASE = "/home/aipmu/Datasets for VLM/Raw dataset files/RSVLM-QA"

json_files = [
    "INRIA-Aerial-Image-Labeling/train_annotations.json",
    "WHU/annotation/annotation/validation.json",
    "WHU/annotation/annotation/test.json",
    "iSAID/train/train/Annotations/iSAID_train.json",
    "iSAID/val/val/Annotations/iSAID_val.json",
]

for jf in json_files:
    path = os.path.join(BASE, jf)
    print(f"\n{'='*60}")
    print(f"FILE: {jf}")
    print(f"Size: {os.path.getsize(path) / 1e6:.1f} MB")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        print(f"Type: dict, Keys: {list(data.keys())[:10]}")
        for k in list(data.keys())[:5]:
            v = data[k]
            if isinstance(v, list):
                print(f"  {k}: list[{len(v)}], first item type: {type(v[0]).__name__ if v else 'empty'}")
                if v and isinstance(v[0], dict):
                    print(f"    Keys: {list(v[0].keys())[:10]}")
                    print(f"    Sample: {json.dumps(v[0], default=str)[:200]}")
            else:
                print(f"  {k}: {type(v).__name__} = {repr(v)[:100]}")
    elif isinstance(data, list):
        print(f"Type: list[{len(data)}], first item type: {type(data[0]).__name__ if data else 'empty'}")
        if data and isinstance(data[0], dict):
            print(f"  Keys: {list(data[0].keys())[:10]}")
            print(f"  Sample: {json.dumps(data[0], default=str)[:200]}")
    print()
