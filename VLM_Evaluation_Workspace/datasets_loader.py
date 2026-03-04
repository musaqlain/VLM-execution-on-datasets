"""
datasets_loader.py
==================
Loads the RSVLM-QA and DisasterM3 datasets into a flat list of
{image_path, question, answer, question_type} dicts, ready for VLM inference.

Verified against the actual file structures on disk:
  - RSVLM-QA: JSONL file where each line = one image with a `vqa_pairs` list.
  - DisasterM3: Separate `all_questions.json` and `all_answers.json` inside a `vqa_format/` directory.
"""

import os
import json

# ──────────────────────────────────────────────
# RSVLM-QA
# ──────────────────────────────────────────────
def load_rsvlmqa_data(
    base_dir="/home/aipmu/Datasets for VLM/Raw dataset files",
    max_samples=None
):
    """
    Loads the RSVLM-QA dataset.

    File layout on disk
    -------------------
    base_dir/
      RSVLM-QA.jsonl          <- metadata (one JSON object per line)
      RSVLM-QA/               <- image root
        INRIA-Aerial-Image-Labeling/train/images/austin11.tif
        LoveDA/...
        WHU/...
        iSAID/...

    Each JSONL line looks like:
      {
        "id": "0",
        "image": "RSVLM-QA/INRIA-Aerial-Image-Labeling/train/images/austin11.tif",
        "vqa_pairs": [
          {"question_id": "1", "question_type": "spatial",
           "question": "Where is the highway?", "answer": "In the center."},
          ...
        ]
      }

    We flatten the vqa_pairs so that every (image, question, answer) triple
    becomes one entry in the returned list.
    """
    jsonl_path = os.path.join(base_dir, "RSVLM-QA.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Cannot find {jsonl_path}")

    print(f"[RSVLM-QA] Loading from {jsonl_path} ...")
    dataset = []
    skipped_images = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and len(dataset) >= max_samples:
                break
            entry = json.loads(line.strip())

            # Image path is relative to base_dir (e.g. "RSVLM-QA/INRIA-.../austin11.tif")
            img_rel = entry.get("image", "")
            img_path = os.path.join(base_dir, img_rel)

            if not os.path.exists(img_path):
                skipped_images += 1
                continue

            for pair in entry.get("vqa_pairs", []):
                if max_samples and len(dataset) >= max_samples:
                    break
                dataset.append({
                    "image_id": entry["id"],
                    "image_path": img_path,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "question_type": pair.get("question_type", "unknown"),
                })

    print(f"[RSVLM-QA] Loaded {len(dataset)} QA pairs  (skipped {skipped_images} missing images).")
    return dataset


# ──────────────────────────────────────────────
# DisasterM3
# ──────────────────────────────────────────────
def load_disasterm3_data(
    base_dir="/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3",
    max_samples=None,
):
    """
    Loads DisasterM3 in its VQA format.

    File layout on disk
    -------------------
    base_dir/
      train_images/            <- PNG images  (e.g. bata_explosion_post_0.png)
      vqa_format/
        all_questions.json     <- {"questions": [{id, question, image_id, type}, ...]}
        all_answers.json       <- {"answers":   [{question_id, answer}, ...]}

    We merge questions and answers by id → question_id, and resolve image paths
    to train_images/<image_id>.
    """
    q_path = os.path.join(base_dir, "vqa_format", "all_questions.json")
    a_path = os.path.join(base_dir, "vqa_format", "all_answers.json")
    image_dir = os.path.join(base_dir, "train_images")

    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Cannot find {q_path}")
    if not os.path.exists(a_path):
        raise FileNotFoundError(f"Cannot find {a_path}")

    print(f"[DisasterM3] Loading questions from {q_path} ...")
    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    print(f"[DisasterM3] Loading answers   from {a_path} ...")
    with open(a_path, "r", encoding="utf-8") as f:
        answers_list = json.load(f)["answers"]

    # Build lookup: question_id → answer string
    answer_map = {a["question_id"]: a["answer"] for a in answers_list}

    dataset = []
    skipped = 0

    for q in questions:
        if max_samples and len(dataset) >= max_samples:
            break

        qid = q["id"]
        image_id = q["image_id"]            # e.g. "bata_explosion_post_0.png"
        img_path = os.path.join(image_dir, image_id)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        gt_answer = answer_map.get(qid, "")

        dataset.append({
            "question_id": qid,
            "image_id": image_id,
            "image_path": img_path,
            "question": q["question"],
            "answer": str(gt_answer),
            "question_type": q.get("type", "unknown"),
        })

    print(f"[DisasterM3] Loaded {len(dataset)} QA pairs  (skipped {skipped} missing images).")
    return dataset


# Quick sanity test
if __name__ == "__main__":
    print("=== RSVLM-QA (first 3) ===")
    for item in load_rsvlmqa_data(max_samples=3):
        print(f"  Q: {item['question'][:80]}  |  A: {item['answer'][:60]}")

    print("\n=== DisasterM3 (first 3) ===")
    for item in load_disasterm3_data(max_samples=3):
        print(f"  Q: {item['question'][:80]}  |  A: {item['answer'][:60]}")
