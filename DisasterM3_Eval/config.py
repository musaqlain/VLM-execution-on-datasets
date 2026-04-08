import os
from dotenv import load_dotenv

# Load .env from the same directory as this file (not CWD)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))

# Root directories
ROOT_DIR = "/home/aipmu/Datasets for VLM/DisasterM3_Eval"
RAW_DATA_DIR = "/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Bench"

# Dataset paths
BENCH_JSON = os.path.join(RAW_DATA_DIR, "benchmark_release.json")

# Results dir
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of all 8 VLMs to evaluate
MODELS = {
    "moondream2": "vikhyatk/moondream2",
    "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b",
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "qwen-vl-chat": "Qwen/Qwen-VL-Chat",
    "instructblip-vicuna": "Salesforce/instructblip-vicuna-7b",
    "idefics2-8b": "HuggingFaceM4/idefics2-8b",
    "internvl2-4b": "OpenGVLab/InternVL2-4B",
    "llava-next-llama3": "llava-hf/llama3-llava-next-8b-hf"
}

# General inference settings
MAX_NEW_TOKENS = 128
DEVICE = "cuda"

# ── Task-type to evaluation track mapping ──────────────────
# Determines which metric suite to use for each task type.
TASK_TRACKS = {
    "Building Damage Counting":          "single_label_mcq",
    "Road Damage Counting":              "single_label_mcq",
    "Relational Reasoning":              "single_label_mcq",
    "Disaster Type Recognition":         "single_label_mcq",
    "Disaster Bearing Bodies Recognition": "multi_label_mcq",
    "Disaster Scene Recognition":        "multi_label_mcq",
    "Disaster Report":                   "free_text",
    "Disaster Restoration Advice":       "free_text",
    "Referring Expression Segmentation": "segmentation",
}

# Which tracks are evaluable by text-generating VLMs
EVALUABLE_TRACKS = {"single_label_mcq", "multi_label_mcq", "free_text"}
