"""
evaluation.py
=============
Task-type-aware evaluation metrics for the DisasterM3 benchmark.

Three evaluation tracks:

  Track A — Single-label MCQ
    accuracy: Extracted option letter matches GT letter.

  Track B — Multi-label MCQ
    subset_accuracy: Full set match.
    precision, recall, f1: Set-level metrics.
    hamming_score: Per-option average accuracy.

  Track C — Free-text generation
    bleu1, bleu4: Custom n-gram precision (Python 3.12-safe).
    rougeL: Longest common subsequence F-measure.

Also provides an option-letter parser that robustly extracts
letters like A, B, C… from varied VLM outputs such as:
  "B"  |  "B. Buildings"  |  "Answer: B, E, F"  |  "I think B and E"
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Set, Optional

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    print("⚠  Missing rouge_score. Run: pip install rouge-score")
    _ROUGE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  OPTION-LETTER PARSER
# ═══════════════════════════════════════════════════════════════

_VALID_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_option_letters(raw: str, max_options: int = 26) -> List[str]:
    """
    Extract option letters (A-Z) from a VLM's raw text output.

    Strategy (ordered by reliability):
      1. Look for explicit letter-dot patterns:  "B. Buildings"  → B
      2. Look for comma-separated letters:       "B, E, F"      → B, E, F
      3. Bare single-letter responses:           "B"             → B
      4. Letters after "answer:" prefix:         "Answer: B"     → B

    Parameters
    ----------
    raw : str
        The raw VLM text prediction.
    max_options : int
        Maximum valid option letter (e.g. 8 → only A-H are valid).

    Returns
    -------
    Sorted list of unique uppercase option letters found.
    """
    if not raw or not raw.strip():
        return []

    text = raw.strip()
    valid = set(chr(ord("A") + i) for i in range(min(max_options, 26)))

    # Strategy 1: "A. something" patterns
    dot_matches = re.findall(r"\b([A-Z])\.", text)
    if dot_matches:
        found = sorted(set(m for m in dot_matches if m in valid))
        if found:
            return found

    # Strategy 2: Comma-separated pattern like "B, E, F" or "B,E,F"
    comma_pattern = re.findall(r"\b([A-Z])\b", text.upper())
    # Filter to only valid option letters (not words like "I", "A" in prose)
    if len(comma_pattern) >= 1:
        # If the text is short (MCQ-style answer) trust all letters found
        if len(text) < 30:
            found = sorted(set(l for l in comma_pattern if l in valid))
            if found:
                return found

    # Strategy 3: Strip "Answer:" prefix then look for letters
    stripped = re.sub(r"(?i)^(answer|ans|option|choice)\s*[:\-]?\s*", "", text).strip()
    if stripped:
        letters = re.findall(r"\b([A-Z])\b", stripped.upper())
        if letters:
            found = sorted(set(l for l in letters if l in valid))
            if found:
                return found

    # Strategy 4: First capital letter in the response
    first_letter = re.search(r"([A-Z])", text.upper())
    if first_letter and first_letter.group(1) in valid:
        return [first_letter.group(1)]

    return []


# ═══════════════════════════════════════════════════════════════
#  TEXT NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


# ═══════════════════════════════════════════════════════════════
#  TRACK A — SINGLE-LABEL MCQ
# ═══════════════════════════════════════════════════════════════

def single_label_accuracy(pred_letters: List[str], gt_letters: List[str]) -> float:
    """1.0 if top predicted letter matches the single GT letter, else 0.0."""
    if not gt_letters:
        return 0.0
    gt = gt_letters[0]
    if not pred_letters:
        return 0.0
    return 1.0 if pred_letters[0] == gt else 0.0


# ═══════════════════════════════════════════════════════════════
#  TRACK B — MULTI-LABEL MCQ
# ═══════════════════════════════════════════════════════════════

def multi_label_metrics(pred_letters: List[str], gt_letters: List[str]) -> Dict[str, float]:
    """
    Compute set-level precision, recall, F1, subset accuracy, hamming score.
    """
    pred_set: Set[str] = set(pred_letters)
    gt_set: Set[str] = set(gt_letters)

    if not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "subset_accuracy": 0.0, "hamming_score": 0.0}

    intersection = pred_set & gt_set
    union = pred_set | gt_set

    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    subset_acc = 1.0 if pred_set == gt_set else 0.0
    hamming = len(intersection) / len(union) if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "subset_accuracy": subset_acc,
        "hamming_score": hamming,
    }


# ═══════════════════════════════════════════════════════════════
#  TRACK C — FREE-TEXT GENERATION
# ═══════════════════════════════════════════════════════════════

# ── Custom BLEU (Python 3.12-safe) ──────────────────────────

def _modified_precision(ref_tokens: List[str], hyp_tokens: List[str], n: int) -> float:
    if len(hyp_tokens) < n:
        return 0.0
    hyp_ngrams = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
    ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    clipped = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return clipped / total if total else 0.0


def _brevity_penalty(ref_len: int, hyp_len: int) -> float:
    if hyp_len == 0:
        return 0.0
    if hyp_len >= ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / hyp_len)


def compute_bleu(ref_tokens: List[str], hyp_tokens: List[str], weights: tuple) -> float:
    if not hyp_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for i in range(1, len(weights) + 1):
        p = _modified_precision(ref_tokens, hyp_tokens, i)
        if p == 0.0 and i > 1:
            p = 1.0 / (len(hyp_tokens) + 1)  # add-1 smoothing
        precisions.append(p)
    log_avg = 0.0
    for w, p in zip(weights, precisions):
        if p == 0:
            return 0.0
        log_avg += w * math.log(p)
    bp = _brevity_penalty(len(ref_tokens), len(hyp_tokens))
    return bp * math.exp(log_avg)


def free_text_metrics(pred: str, gt: str) -> Dict[str, float]:
    """Compute BLEU-1, BLEU-4, ROUGE-L for a free-text prediction."""
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gt).split()

    b1 = compute_bleu(g_tokens, p_tokens, weights=(1.0,)) if p_tokens and g_tokens else 0.0
    b4 = compute_bleu(g_tokens, p_tokens, weights=(0.25, 0.25, 0.25, 0.25)) if p_tokens and g_tokens else 0.0

    rouge = 0.0
    if _ROUGE_AVAILABLE and p_tokens and g_tokens:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge = scorer.score(normalize(gt), normalize(pred))["rougeL"].fmeasure

    return {"bleu1": b1, "bleu4": b4, "rougeL": rouge}


# ═══════════════════════════════════════════════════════════════
#  UNIFIED EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_sample(prediction_raw: str, ground_truth_options: List[str],
                    ground_truth_text: str, eval_track: str,
                    options_count: int = 26) -> Dict:
    """
    Evaluate a single sample, returning per-sample metrics
    based on its evaluation track.
    """
    result = {"eval_track": eval_track}

    if eval_track == "single_label_mcq":
        pred_letters = parse_option_letters(prediction_raw, max_options=options_count)
        result["prediction_parsed"] = pred_letters
        result["accuracy"] = single_label_accuracy(pred_letters, ground_truth_options)

    elif eval_track == "multi_label_mcq":
        pred_letters = parse_option_letters(prediction_raw, max_options=options_count)
        result["prediction_parsed"] = pred_letters
        ml = multi_label_metrics(pred_letters, ground_truth_options)
        result.update(ml)

    elif eval_track == "free_text":
        result["prediction_parsed"] = []
        ft = free_text_metrics(prediction_raw, ground_truth_text)
        result.update(ft)

    else:
        result["prediction_parsed"] = []

    return result


def evaluate_all(results: List[Dict]) -> Dict:
    """
    Aggregate metrics across all samples.

    Parameters
    ----------
    results : list of dict
        Each dict must contain: eval_track, per_sample_metrics.

    Returns
    -------
    dict with overall metrics and per-task-type breakdown.
    """
    by_track = defaultdict(list)
    by_task = defaultdict(list)

    for r in results:
        track = r.get("eval_track", "unknown")
        task = r.get("task_type", "unknown")
        metrics = r.get("per_sample_metrics", {})
        by_track[track].append(metrics)
        by_task[task].append((track, metrics))

    # ── Overall per-track ────────────────────────────────────
    overall = {}

    # Single-label MCQ
    sl = by_track.get("single_label_mcq", [])
    if sl:
        overall["mcq_single_label_accuracy"] = sum(m.get("accuracy", 0) for m in sl) / len(sl)

    # Multi-label MCQ
    ml = by_track.get("multi_label_mcq", [])
    if ml:
        for metric in ["precision", "recall", "f1", "subset_accuracy", "hamming_score"]:
            overall[f"mcq_multi_label_{metric}"] = sum(m.get(metric, 0) for m in ml) / len(ml)

    # Free-text
    ft = by_track.get("free_text", [])
    if ft:
        for metric in ["bleu1", "bleu4", "rougeL"]:
            overall[f"free_text_{metric}"] = sum(m.get(metric, 0) for m in ft) / len(ft)

    # ── Per-task breakdown ───────────────────────────────────
    per_task = {}
    for task, items in by_task.items():
        track = items[0][0]
        metrics_list = [m for _, m in items]
        n = len(metrics_list)
        task_metrics = {"track": track, "count": n}

        if track == "single_label_mcq":
            task_metrics["accuracy"] = sum(m.get("accuracy", 0) for m in metrics_list) / n
        elif track == "multi_label_mcq":
            for metric in ["precision", "recall", "f1", "subset_accuracy", "hamming_score"]:
                task_metrics[metric] = sum(m.get(metric, 0) for m in metrics_list) / n
        elif track == "free_text":
            for metric in ["bleu1", "bleu4", "rougeL"]:
                task_metrics[metric] = sum(m.get(metric, 0) for m in metrics_list) / n

        per_task[task] = task_metrics

    return {"overall": overall, "by_task": per_task}
