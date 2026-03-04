"""
evaluation.py
=============
VLM answer evaluation metrics for Remote Sensing VQA.

Computes (8 metrics total):
  • Exact Match (EM)  – strict string equality after normalisation
  • BLEU-1 / BLEU-4   – n-gram precision (standard VQA metric)
  • ROUGE-L            – longest common subsequence (good for long answers)
  • METEOR             – alignment-based, uses synonyms + stemming
  • BERTScore          – semantic embedding similarity  (F1)
  • Token F1           – token-level precision / recall / F1
  • CIDEr              – consensus-based image description evaluation
"""

import re
from typing import List, Dict
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠  Missing libraries.  Run:  pip install nltk rouge-score")

# Optional heavier metrics – gracefully degrade if not installed
_HAVE_METEOR = False
_HAVE_BERTSCORE = False
_HAVE_CIDER = False

try:
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor
    _HAVE_METEOR = True
except Exception:
    pass

try:
    from bert_score import score as _bert_score_fn
    _HAVE_BERTSCORE = True
except ImportError:
    pass

try:
    from pycocoevalcap.cider.cider import Cider as _CiderScorer
    _HAVE_CIDER = True
except ImportError:
    pass


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


def meteor(pred: str, gt: str) -> float:
    """METEOR score (requires nltk with wordnet)."""
    if not _HAVE_METEOR:
        return 0.0
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0
    try:
        return _nltk_meteor([g_tokens], p_tokens)
    except Exception:
        return 0.0


def token_f1(pred: str, gt: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


# ── batch metrics (computed once over all results) ──
def _compute_bertscore_batch(preds: List[str], gts: List[str]) -> List[float]:
    """BERTScore F1 for a batch of predictions."""
    if not _HAVE_BERTSCORE or not preds:
        return [0.0] * len(preds)
    try:
        _, _, f1 = _bert_score_fn(
            preds, gts, lang="en", verbose=False,
            device="cpu",  # compute on CPU to not interfere with GPU models
            batch_size=64,
        )
        return f1.tolist()
    except Exception:
        return [0.0] * len(preds)


def _compute_cider_batch(preds: List[str], gts: List[str]) -> List[float]:
    """CIDEr score for a batch of predictions."""
    if not _HAVE_CIDER or not preds:
        return [0.0] * len(preds)
    try:
        # CIDEr expects {id: [sentence]} format
        gts_dict = {i: [g] for i, g in enumerate(gts)}
        preds_dict = {i: [p] for i, p in enumerate(preds)}
        cider = _CiderScorer()
        avg_score, per_sample = cider.compute_score(gts_dict, preds_dict)
        return list(per_sample)
    except Exception:
        return [0.0] * len(preds)


# ── aggregate over full dataset ─────────────
def evaluate_predictions(results: List[Dict]) -> Dict:
    """
    Parameters
    ----------
    results : list of {"prediction": str, "ground_truth": str}

    Returns
    -------
    dict with averaged exact_match, bleu1, bleu4, rougeL, meteor, bertscore, token_f1, cider
    """
    n = len(results)
    if n == 0:
        return {}

    # Per-sample metrics
    sums = {
        "exact_match": 0.0, "bleu1": 0.0, "bleu4": 0.0,
        "rougeL": 0.0, "meteor": 0.0, "token_f1": 0.0,
    }
    for r in results:
        p, g = r["prediction"], r["ground_truth"]
        sums["exact_match"] += exact_match(p, g)
        b = bleu_scores(p, g)
        sums["bleu1"] += b["bleu1"]
        sums["bleu4"] += b["bleu4"]
        sums["rougeL"] += rouge_l(p, g)
        sums["meteor"] += meteor(p, g)
        sums["token_f1"] += token_f1(p, g)

    avg = {k: v / n for k, v in sums.items()}

    # Batch metrics (BERTScore, CIDEr)
    preds_norm = [normalize(r["prediction"]) for r in results]
    gts_norm = [normalize(r["ground_truth"]) for r in results]

    bert_scores = _compute_bertscore_batch(preds_norm, gts_norm)
    avg["bertscore"] = sum(bert_scores) / n if bert_scores else 0.0

    cider_scores = _compute_cider_batch(preds_norm, gts_norm)
    avg["cider"] = sum(cider_scores) / n if cider_scores else 0.0

    return avg


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
