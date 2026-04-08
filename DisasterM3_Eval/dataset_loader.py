"""
dataset_loader.py
=================
Loads the DisasterM3 Benchmark dataset from benchmark_release.json.

Key design choices
------------------
* Handles the field name ``options_list`` (not ``options``).
* Dynamically builds an options string when ``options_str`` is missing
  (e.g. Relational Reasoning entries that only provide ``option_str``
  or ``options_list``).
* Flattens list-type prompts (Disaster Report / Restoration Advice)
  into a single string.
* Tags every sample with an ``eval_track`` derived from config.TASK_TRACKS.
* Provides a ``stratified_sample()`` helper that pulls proportionally
  from each task type — so a 50-sample mini-dataset covers all 8
  evaluable task types, not just the first disaster event.
"""

import os
import json
import random
from collections import defaultdict
from config import BENCH_JSON, RAW_DATA_DIR, TASK_TRACKS, EVALUABLE_TRACKS

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── helpers ──────────────────────────────────────────────────
def _build_options_str(options_list):
    """Convert ['Stadiums', 'Buildings', ...] → 'A. Stadiums, B. Buildings, ...'"""
    if not options_list or not isinstance(options_list, list):
        return ""
    return ", ".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options_list))


def _flatten_prompt(raw_prompt):
    """If the prompt field is a list (Disaster Report), join into one string."""
    if isinstance(raw_prompt, list):
        return " ".join(str(p) for p in raw_prompt)
    return str(raw_prompt)


def _parse_gt_options(gt_option_str):
    """
    Parse ground truth option string into a sorted list of uppercase letters.
    'B, E, F, G.' → ['B', 'E', 'F', 'G']
    'A'           → ['A']
    ''            → []
    """
    if not gt_option_str or gt_option_str == "N/A":
        return []
    parts = gt_option_str.replace(".", "").split(",")
    letters = [p.strip().upper() for p in parts if p.strip() and len(p.strip()) == 1]
    return sorted(letters)


# ── main loader ──────────────────────────────────────────────
def load_disasterm3_bench(max_samples=None, stratified=False, skip_segmentation=True, seed=42):
    """
    Load the DisasterM3 Benchmark dataset.

    Parameters
    ----------
    max_samples : int or None
        Cap on total number of samples to return.
    stratified : bool
        If True, sample proportionally from each task type.
        If False, take the first ``max_samples`` entries.
    skip_segmentation : bool
        If True, skip Referring Expression Segmentation entries
        (VLMs cannot generate mask images).
    seed : int
        Random seed for stratified sampling reproducibility.

    Returns
    -------
    list of dict — each with rich metadata for machine-readable output.
    """
    if not os.path.exists(BENCH_JSON):
        raise FileNotFoundError(f"Cannot find benchmark JSON at {BENCH_JSON}")

    print(f"[Dataset] Loading DisasterM3 Bench from {BENCH_JSON} ...")
    with open(BENCH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Build all valid samples ──────────────────────────────
    all_samples = []
    skipped_img = 0
    skipped_seg = 0

    for idx, entry in enumerate(data):
        task = entry.get("task", "unknown")
        track = TASK_TRACKS.get(task, "unknown")

        # Skip segmentation if requested
        if skip_segmentation and track == "segmentation":
            skipped_seg += 1
            continue

        # ── Resolve image paths ──────────────────────────────
        pre_rel = entry.get("pre_image_path", "")
        post_rel = entry.get("post_image_path", "")
        img_rel = entry.get("image_path", "")  # Relational Reasoning uses this

        if pre_rel:
            pre_rel = pre_rel.replace("\\", "/")
        if post_rel:
            post_rel = post_rel.replace("\\", "/")
        if img_rel:
            img_rel = img_rel.replace("\\", "/")

        pre_path = os.path.join(RAW_DATA_DIR, pre_rel) if pre_rel else None
        post_path = os.path.join(RAW_DATA_DIR, post_rel) if post_rel else None
        img_path = os.path.join(RAW_DATA_DIR, img_rel) if img_rel else None

        # Determine primary image (prefer post-disaster, fallback to pre, then image_path)
        primary_image = None
        if post_path and os.path.exists(post_path):
            primary_image = post_path
        elif pre_path and os.path.exists(pre_path):
            primary_image = pre_path
        elif img_path and os.path.exists(img_path):
            primary_image = img_path

        if not primary_image:
            skipped_img += 1
            continue

        # ── Build options string ─────────────────────────────
        options_str = entry.get("options_str", "")
        options_list = entry.get("options_list", [])
        # Relational Reasoning uses "option_str" (singular, no 's')
        if not options_str:
            options_str = entry.get("option_str", "")
        if not options_str and options_list:
            options_str = _build_options_str(options_list)

        # ── Build question ───────────────────────────────────
        prompt = _flatten_prompt(entry.get("prompts", ""))
        question = prompt
        if options_str and track in ("single_label_mcq", "multi_label_mcq"):
            question = (
                f"{prompt}\n"
                f"Options: {options_str}\n"
                f"Answer with the correct option letter(s)."
            )

        # ── Ground truth ─────────────────────────────────────
        gt_option_raw = str(entry.get("ground_truth_option", "")).strip()
        gt_text = str(entry.get("ground_truth", "")).strip()
        gt_parsed = _parse_gt_options(gt_option_raw)

        # For MCQ tasks, the "answer" is the option letters.
        # For free-text tasks, use the full text.
        if track in ("single_label_mcq", "multi_label_mcq") and gt_parsed:
            answer = ", ".join(gt_parsed)
        else:
            answer = gt_text

        sample = {
            "sample_id": idx,
            "task_type": task,
            "eval_track": track,
            "pre_image_path": pre_path,
            "post_image_path": post_path,
            "primary_image_path": primary_image,
            "image_type": entry.get("post_image_type", entry.get("image_type", "")),
            "prompt_raw": prompt,
            "question": question.strip(),
            "options_str": options_str,
            "options_list": options_list if options_list else [],
            "ground_truth_text": gt_text,
            "ground_truth_option_raw": gt_option_raw,
            "ground_truth_options": gt_parsed,
            "answer": answer,
        }
        all_samples.append(sample)

    print(f"[Dataset] Built {len(all_samples)} evaluable samples "
          f"(skipped {skipped_img} missing images, {skipped_seg} segmentation).")

    # ── Apply sampling ───────────────────────────────────────
    if max_samples and max_samples < len(all_samples):
        if stratified:
            all_samples = _stratified_sample(all_samples, max_samples, seed)
        else:
            all_samples = all_samples[:max_samples]

    print(f"[Dataset] Returning {len(all_samples)} samples.")
    return all_samples


def _stratified_sample(samples, n, seed=42):
    """
    Proportionally sample ``n`` items from each task type.
    Guarantees at least 1 sample per task type (if available).
    """
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for s in samples:
        by_task[s["task_type"]].append(s)

    total = len(samples)
    result = []

    # First pass: allocate proportionally, minimum 1 per task
    allocations = {}
    for task, items in by_task.items():
        allocations[task] = max(1, round(n * len(items) / total))

    # Adjust to hit exactly n
    allocated = sum(allocations.values())
    if allocated > n:
        # Trim from largest groups
        for task in sorted(allocations, key=lambda t: allocations[t], reverse=True):
            if allocated <= n:
                break
            reduce = min(allocations[task] - 1, allocated - n)
            allocations[task] -= reduce
            allocated -= reduce
    elif allocated < n:
        # Add to largest groups
        for task in sorted(allocations, key=lambda t: len(by_task[t]), reverse=True):
            if allocated >= n:
                break
            add = min(len(by_task[task]) - allocations[task], n - allocated)
            allocations[task] += add
            allocated += add

    for task, items in by_task.items():
        k = min(allocations.get(task, 1), len(items))
        result.extend(rng.sample(items, k))

    rng.shuffle(result)
    print(f"[Dataset] Stratified sample: {dict(allocations)}")
    return result


if __name__ == "__main__":
    items = load_disasterm3_bench(max_samples=10, stratified=True)
    for i, item in enumerate(items):
        print(f"\n--- Item {i} ---")
        print(f"  Track:   {item['eval_track']}")
        print(f"  Task:    {item['task_type']}")
        print(f"  Image:   {os.path.basename(item['primary_image_path'])}")
        print(f"  Q:       {item['question'][:120]}")
        print(f"  GT opts: {item['ground_truth_options']}")
        print(f"  Answer:  {item['answer'][:80]}")
