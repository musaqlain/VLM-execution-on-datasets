import os
from dotenv import load_dotenv

# Load .env from the same directory as this file (not CWD)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))

# ── Root directories ─────────────────────────────────────────────────────────
# UPDATE these two paths to match where you placed the data on the Windows server.
# The EVAL_DIR is wherever this repo is cloned.
# The RAW_DATA_DIR is wherever the DisasterM3 benchmark JSON + images live.

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))          # auto-detected
RAW_DATA_DIR = os.environ.get(
    "DISASTERM3_DATA_DIR",
    r"C:\Datasets\DisasterM3_Bench"   # ← Change this if your data is elsewhere
)

# Dataset paths
BENCH_JSON = os.path.join(RAW_DATA_DIR, "benchmark_release.json")

# Results dir  (created automatically)
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Windows machine VLMs (5 models, all ≤7B, fit on RTX A4500 20 GB) ─────────
#
#  Model           | Params | ~VRAM (fp16) | Notes
# ─────────────────┼────────┼─────────────┼──────────────────────────────────
#  moondream2      |  1.6B  |  ~3 GB      | Tiny edge model; legacy baseline
#  phi-3.5-vision  |  4.2B  |  ~9 GB      | Dense, efficient MLLM
#  kimi-vl-a3b     |  2.8B  |  ~7 GB      | MoE; MoonViT native-res encoder
#  llava-1.5-7b    |  7B    | ~14 GB      | 2023 legacy baseline (CLIP+Vicuna)
#  qwen2.5-vl-7b   |  7B    | ~16 GB      | State-of-the-art instruction follower
# ─────────────────┴────────┴─────────────┴──────────────────────────────────

MODELS = {
    # "moondream2":     "vikhyatk/moondream2",  # DISABLED: incompatible with transformers>=5.x / PyTorch 2.5
    "phi-3.5-vision":   "microsoft/Phi-3.5-vision-instruct",
    "kimi-vl-a3b":      "moonshotai/Kimi-VL-A3B-Instruct",
    "llava-1.5-7b":     "llava-hf/llava-1.5-7b-hf",
    "qwen2.5-vl-7b":    "Qwen/Qwen2.5-VL-7B-Instruct",
}

# ── General inference settings ───────────────────────────────────────────────
MAX_NEW_TOKENS = 256     # raised from 128 — free-text tasks need more room
DEVICE         = "cuda"

# ── Task-type → evaluation track mapping ────────────────────────────────────
# Determines which metric suite is used for each task type.
TASK_TRACKS = {
    "Building Damage Counting":            "single_label_mcq",
    "Road Damage Counting":                "single_label_mcq",
    "Relational Reasoning":                "single_label_mcq",
    "Disaster Type Recognition":           "single_label_mcq",
    "Disaster Bearing Bodies Recognition": "multi_label_mcq",
    "Disaster Scene Recognition":          "multi_label_mcq",
    "Disaster Report":                     "free_text",
    "Disaster Restoration Advice":         "free_text",
    "Referring Expression Segmentation":   "segmentation",   # skipped by default
}

# Tracks that text-only VLMs can produce scored results for
EVALUABLE_TRACKS = {"single_label_mcq", "multi_label_mcq", "free_text"}
