"""
evaluation.py
=============
VLM answer evaluation metrics for Remote Sensing VQA.

Computes:
  • Exact Match (EM)  – strict string equality after normalisation
  • BLEU-1 / BLEU-4   – n-gram precision (standard VQA metric)
  • ROUGE-L            – longest common subsequence (good for long answers)
"""

import re
from typing import List, Dict

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠  Missing libraries.  Run:  pip install nltk rouge-score")


# ── helpers ──────────────────────────────────
def normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


# ── per-sample metrics ──────────────────────
def exact_match(pred: str, gt: str) -> int:
    return int(normalize(pred) == normalize(gt))


def bleu_scores(pred: str, gt: str) -> dict:
    p = normalize(pred).split()
    g = [normalize(gt).split()]
    if not p or not g[0]:
        return {"bleu1": 0.0, "bleu4": 0.0}
    sm = SmoothingFunction().method4
    b1 = sentence_bleu(g, p, weights=(1, 0, 0, 0), smoothing_function=sm)
    try:
        b4 = sentence_bleu(g, p, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sm)
    except Exception:
        b4 = 0.0
    return {"bleu1": b1, "bleu4": b4}


def rouge_l(pred: str, gt: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(normalize(gt), normalize(pred))["rougeL"].fmeasure


# ── aggregate over full dataset ─────────────
def evaluate_predictions(results: List[Dict]) -> Dict:
    """
    Parameters
    ----------
    results : list of {"prediction": str, "ground_truth": str}

    Returns
    -------
    dict with averaged exact_match, bleu1, bleu4, rougeL
    """
    n = len(results)
    if n == 0:
        return {}
    sums = {"exact_match": 0.0, "bleu1": 0.0, "bleu4": 0.0, "rougeL": 0.0}
    for r in results:
        p, g = r["prediction"], r["ground_truth"]
        sums["exact_match"] += exact_match(p, g)
        b = bleu_scores(p, g)
        sums["bleu1"] += b["bleu1"]
        sums["bleu4"] += b["bleu4"]
        sums["rougeL"] += rouge_l(p, g)
    return {k: v / n for k, v in sums.items()}


def evaluate_by_type(results: List[Dict]) -> Dict[str, Dict]:
    """
    Same as evaluate_predictions but grouped by `question_type`.
    Each result dict should also contain a "question_type" key.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        buckets[r.get("question_type", "unknown")].append(r)
    return {qtype: evaluate_predictions(items) for qtype, items in buckets.items()}
