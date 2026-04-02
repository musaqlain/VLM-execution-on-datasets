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
    base_dir="/home/aiserver/Documents/opensource/VLM-execution-on-datasets/Raw dataset files",
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

            # Remap: JSONL says "RSVLM-QA/..." but actual dir is "RSVLM-QA/RSVL-VQA/..."
            if not os.path.exists(img_path) and img_rel.startswith("RSVLM-QA/"):
                alt_rel = img_rel.replace("RSVLM-QA/", "RSVLM-QA/RSVL-VQA/", 1)
                alt_path = os.path.join(base_dir, alt_rel)
                if os.path.exists(alt_path):
                    img_path = alt_path

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
    base_dir="/home/aiserver/Documents/opensource/VLM-execution-on-datasets/Raw dataset files/DisasterM3_Instruct",
    max_samples=None,
):
    """
    Loads DisasterM3 in its instruction-tuning format.

    File layout on disk (DisasterM3_Instruct from HuggingFace)
    -----------------------------------------------------------
    base_dir/
      train_images/            <- PNG images  (e.g. bata_explosion_post_0.png)
      train.json               <- list of dicts, each with:
        {
          "pre_image_path": "train_images\\name_pre_N.png",
          "post_image_path": "train_images\\name_post_N.png",
          "post_image_type": "Optical",
          "ground_truth": "Buildings, Roads, Forest, Farmland",
          "ground_truth_option": "B, E, F, G.",
          "options_list": ["Stadiums", "Buildings", ...],
          "options_str": "A. Stadiums, B. Buildings, ..."
        }

    We construct a VQA question from the options and use ground_truth as the answer.
    We use the post_image for VQA (the disaster-affected image).
    """
    train_json = os.path.join(base_dir, "train.json")
    image_dir = os.path.join(base_dir, "train_images")

    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Cannot find {train_json}")

    print(f"[DisasterM3] Loading from {train_json} ...")
    with open(train_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    dataset = []
    skipped = 0

    for idx, entry in enumerate(entries):
        if max_samples and len(dataset) >= max_samples:
            break

        # Use post image (the disaster image) — normalise backslashes
        post_rel = entry.get("post_image_path", "").replace("\\", "/")
        img_path = os.path.join(base_dir, post_rel)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Build question from the options
        options_str = entry.get("options_str", "")
        question = (
            f"Look at this remote sensing image taken after a disaster. "
            f"Which of the following land-cover or infrastructure categories "
            f"are visible? Options: {options_str}"
        )

        gt_answer = entry.get("ground_truth", "")
        post_type = entry.get("post_image_type", "Optical")

        dataset.append({
            "question_id": idx,
            "image_id": os.path.basename(post_rel),
            "image_path": img_path,
            "question": question,
            "answer": str(gt_answer),
            "question_type": f"disaster_classification_{post_type.lower()}",
        })

    print(f"[DisasterM3] Loaded {len(dataset)} QA pairs  (skipped {skipped} missing images).")
    return dataset


# ──────────────────────────────────────────────
# RSVQA-HR  (High Resolution)
# ──────────────────────────────────────────────
def load_rsvqa_hr_data(
    base_dir="/home/aiserver/Documents/opensource/VLM-execution-on-datasets/Raw dataset files/RSVQA-HR",
    max_samples=None,
    split="test",
):
    """
    Loads the RSVQA-HR dataset (Zenodo).

    File layout on disk
    -------------------
    base_dir/
      Data/                            <- TIF/PNG images  (e.g. 0.tif, 1.tif, ...)
      USGS_split_test_questions.json   <- {"questions": [{"id": 0, "active": True/False}, ...]}
      USGSquestions.json               <- master questions [{"id": 0, "question": "...", "img_id": 0, "answers_ids": [0]}, ...]
      USGSanswers.json                 <- master answers [{"id": 0, "question_id": 0, "answer": "yes"}, ...]
      ...

    The question types in RSVQA-HR include:
      - presence (yes/no)
      - comparison
      - count
      - rural_urban
      - area
    """
    master_q_path = os.path.join(base_dir, "USGSquestions.json")
    master_a_path = os.path.join(base_dir, "USGSanswers.json")
    split_q_path = os.path.join(base_dir, f"USGS_split_{split}_questions.json")
    image_dir = os.path.join(base_dir, "Data")

    if not os.path.exists(master_q_path):
        raise FileNotFoundError(f"Cannot find {master_q_path}")
    if not os.path.exists(split_q_path):
        raise FileNotFoundError(f"Cannot find {split_q_path}")

    print(f"[RSVQA-HR] Loading split flags from {split_q_path} ...")
    with open(split_q_path, "r", encoding="utf-8") as f:
        split_flags = json.load(f)["questions"]
    
    active_qids = {item["id"] for item in split_flags if item.get("active", False)}

    print(f"[RSVQA-HR] Loading master questions from {master_q_path} ...")
    with open(master_q_path, "r", encoding="utf-8") as f:
        master_questions = json.load(f)["questions"]

    print(f"[RSVQA-HR] Loading master answers from {master_a_path} ...")
    with open(master_a_path, "r", encoding="utf-8") as f:
        master_answers = json.load(f)["answers"]

    # Build lookup: question_id → raw string answer
    # If a question has multiple answers in the answers file (say by different people),
    # we'll just take the first one or the most frequent. Let's take the first found.
    answer_map = {}
    for a in master_answers:
        if a["question_id"] not in answer_map:
            answer_map[a["question_id"]] = a["answer"]

    dataset = []
    skipped = 0

    for q in master_questions:
        qid = q["id"]
        if qid not in active_qids:
            continue
            
        if max_samples and len(dataset) >= max_samples:
            break

        img_id = q["img_id"]

        # RSVQA-HR images are named by their numeric id (e.g. "Data/0.tif")
        img_filename = f"{img_id}.tif"
        img_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(img_path):
            img_filename = f"{img_id}.png"
            img_path = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_path):
                skipped += 1
                continue

        gt_answer = answer_map.get(qid, "")

        dataset.append({
            "question_id": qid,
            "image_id": str(img_id),
            "image_path": img_path,
            "question": q["question"],
            "answer": str(gt_answer),
            "question_type": q.get("type", "unknown"),
        })

    print(f"[RSVQA-HR] Loaded {len(dataset)} QA pairs  (skipped {skipped} missing images).")
    return dataset


# ──────────────────────────────────────────────
# EarthVQA
# ──────────────────────────────────────────────
def load_earthvqa_data(
    base_dir="/home/aiserver/Documents/opensource/VLM-execution-on-datasets/Raw dataset files/EarthVQA",
    max_samples=None,
    splits=None,
):
    """
    Loads the EarthVQA dataset (HuggingFace / Junjue-Wang).

    File layout on disk
    -------------------
    base_dir/
      Train/
        images_png/        <- PNG images
        masks_png/         <- semantic masks (not used for VQA inference)
      Val/
        images_png/
        masks_png/
      Test/
        images_png/
      Train_QA.json        <- list of {Type, Question, Answer, Image} dicts
      Val_QA.json
      Test_QA.json

    Each QA JSON entry looks like:
      {"Type": "Basic_Counting", "Question": "How many buildings are there?",
       "Answer": "5", "Image": "0001.png"}
    """
    if splits is None:
        splits = ["Test", "Val"]

    dataset = []
    skipped = 0

    for split in splits:
        qa_path = os.path.join(base_dir, f"{split}_QA.json")
        image_dir = os.path.join(base_dir, split, "images_png")

        if not os.path.exists(qa_path):
            print(f"[EarthVQA] Warning: {qa_path} not found, skipping split '{split}'")
            continue

        print(f"[EarthVQA] Loading QA from {qa_path} ...")
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_list = json.load(f)

        for entry in qa_list:
            if max_samples and len(dataset) >= max_samples:
                break

            img_name = entry.get("Image", entry.get("image", ""))
            img_path = os.path.join(image_dir, img_name)

            if not os.path.exists(img_path):
                skipped += 1
                continue

            dataset.append({
                "question_id": len(dataset),
                "image_id": img_name,
                "image_path": img_path,
                "question": entry.get("Question", entry.get("question", "")),
                "answer": str(entry.get("Answer", entry.get("answer", ""))),
                "question_type": entry.get("Type", entry.get("type", "unknown")),
            })

        if max_samples and len(dataset) >= max_samples:
            break

    print(f"[EarthVQA] Loaded {len(dataset)} QA pairs  (skipped {skipped} missing images).")
    return dataset


# Quick sanity test
if __name__ == "__main__":
    print("=== RSVLM-QA (first 3) ===")
    for item in load_rsvlmqa_data(max_samples=3):
        print(f"  Q: {item['question'][:80]}  |  A: {item['answer'][:60]}")

    print("\n=== DisasterM3 (first 3) ===")
    for item in load_disasterm3_data(max_samples=3):
        print(f"  Q: {item['question'][:80]}  |  A: {item['answer'][:60]}")

    print("\n=== RSVQA-HR (first 3) ===")
    for item in load_rsvqa_hr_data(max_samples=3):
        print(f"  Q: {item['question'][:80]}  |  A: {item['answer'][:60]}")
