#!/bin/bash
# ================================================================
# download_datasets.sh — Master Dataset Download Script
# ================================================================
# Downloads RSVQA-HR (Zenodo) and EarthVQA (HuggingFace) datasets
# into the "Raw dataset files" directory.
#
# PREREQUISITES:
#   - wget, tar, git, git-lfs installed
#   - ~20 GB free disk space
#
# HOW TO RUN:
#   cd "/home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace"
#   bash download_datasets.sh
#
# ESTIMATED TIME:
#   RSVQA-HR (~14 GB): 10-30 min depending on internet speed
#   EarthVQA (~2-3 GB): 5-10 min depending on internet speed
# ================================================================

# Note: we do NOT use 'set -e' because apt-get can return non-zero
# due to unrelated DKMS/kernel issues while still installing the
# package we need.

RAW_DIR="/home/aiserver/Documents/opensource/VLM-execution-on-datasets/Raw dataset files"
ZENODO_BASE="https://zenodo.org/api/records/6344367/files"

echo "========================================================"
echo "  DATASET DOWNLOAD SCRIPT  •  $(date)"
echo "========================================================"
echo ""

# ── Helper ──────────────────────────────────────────
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "❌ Required tool '$1' not found. Please install it first."
        exit 1
    fi
}

check_tool wget
check_tool tar
check_tool git

# ══════════════════════════════════════════════════════
# 1. RSVQA-HR  (Zenodo record 6344367, ~14 GB total)
# ══════════════════════════════════════════════════════
RSVQA_DIR="$RAW_DIR/RSVQA-HR"

echo "────────────────────────────────────────────────────────"
echo "  📦 1/2  RSVQA-HR (High Resolution)"
echo "────────────────────────────────────────────────────────"

if [ -d "$RSVQA_DIR/Data" ] && [ "$(ls -1 "$RSVQA_DIR/Data/" 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "  ✅ RSVQA-HR images already exist ($(ls -1 "$RSVQA_DIR/Data/" | wc -l) files). Skipping download."
else
    mkdir -p "$RSVQA_DIR"
    cd "$RSVQA_DIR"

    # Download Images.tar (~13.5 GB)
    echo "  📥 Downloading Images.tar (~13.5 GB) ..."
    if [ ! -f "Images.tar" ]; then
        wget -c --progress=bar:force:noscroll \
            "${ZENODO_BASE}/Images.tar/content" \
            -O Images.tar
    else
        echo "  ⚠  Images.tar already exists, verifying ..."
    fi

    # Verify checksum
    echo "  🔍 Verifying checksum ..."
    EXPECTED_MD5="e16c21a040ef4a17afe2d20dffe7758b"
    ACTUAL_MD5=$(md5sum Images.tar | awk '{print $1}')
    if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
        echo "  ⚠  Checksum mismatch! Expected: $EXPECTED_MD5  Got: $ACTUAL_MD5"
        echo "      The file may be corrupted. Consider deleting and re-downloading."
    else
        echo "  ✅ Checksum verified."
    fi

    # Extract images
    echo "  📂 Extracting Images.tar ..."
    tar -xf Images.tar
    echo "  ✅ Images extracted: $(ls -1 Data/ 2>/dev/null | wc -l) files"

    # Optionally remove tar to save space
    # rm -f Images.tar

    cd "$RSVQA_DIR"
fi

# Download JSON annotation files (these are small, always re-download to be safe)
echo "  📥 Downloading annotation JSON files ..."
cd "$RSVQA_DIR"

JSON_FILES=(
    "USGS_split_test_questions.json"
    "USGS_split_test_answers.json"
    "USGS_split_test_images.json"
    "USGS_split_val_questions.json"
    "USGS_split_val_answers.json"
    "USGS_split_val_images.json"
    "USGS_split_train_questions.json"
    "USGS_split_train_answers.json"
    "USGS_split_train_images.json"
    "USGS_split_test_phili_questions.json"
    "USGS_split_test_phili_answers.json"
    "USGS_split_test_phili_images.json"
    "USGSquestions.json"
    "USGSanswers.json"
    "USGSimages.json"
    "USGSpeople.json"
)

for jf in "${JSON_FILES[@]}"; do
    if [ ! -f "$jf" ]; then
        echo "    ↓ $jf"
        wget -q "${ZENODO_BASE}/${jf}/content" -O "$jf"
    else
        echo "    ✓ $jf (already exists)"
    fi
done

echo "  ✅ RSVQA-HR download complete!"
echo ""

# ══════════════════════════════════════════════════════
# 2. EarthVQA  (HuggingFace, ~2-3 GB)
# ══════════════════════════════════════════════════════
EARTHVQA_DIR="$RAW_DIR/EarthVQA"

echo "────────────────────────────────────────────────────────"
echo "  📦 2/2  EarthVQA"
echo "────────────────────────────────────────────────────────"

if [ -d "$EARTHVQA_DIR" ] && [ -f "$EARTHVQA_DIR/Test_QA.json" ]; then
    echo "  ✅ EarthVQA already exists. Skipping download."
else
    # Check for git-lfs
    if ! command -v git-lfs &>/dev/null; then
        echo "  ⚠  git-lfs not found. Installing ..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq git-lfs || true
    fi
    git lfs install --skip-smudge 2>/dev/null || true

    echo "  📥 Cloning EarthVQA from HuggingFace ..."
    echo "     (This requires HuggingFace access. If it fails, see manual instructions below.)"
    echo ""

    cd "$RAW_DIR"

    # Try cloning from HuggingFace
    if git clone https://huggingface.co/datasets/Kingdrone-Junjue/EarthVQA "$EARTHVQA_DIR" 2>/dev/null; then
        cd "$EARTHVQA_DIR"
        git lfs pull
        echo "  ✅ EarthVQA downloaded successfully!"
    else
        echo ""
        echo "  ⚠  HuggingFace clone failed. This dataset may require authentication."
        echo ""
        echo "  ╔══════════════════════════════════════════════════════════════╗"
        echo "  ║  MANUAL DOWNLOAD INSTRUCTIONS FOR EarthVQA:                ║"
        echo "  ║                                                            ║"
        echo "  ║  Option A: Use huggingface-cli                             ║"
        echo "  ║    pip install huggingface_hub                              ║"
        echo "  ║    huggingface-cli login                                   ║"
        echo "  ║    huggingface-cli download Kingdrone-Junjue/EarthVQA \\    ║"
        echo "  ║      --repo-type dataset \\                                 ║"
        echo "  ║      --local-dir \"$EARTHVQA_DIR\"                           ║"
        echo "  ║                                                            ║"
        echo "  ║  Option B: Download from Google Drive                      ║"
        echo "  ║    Visit: https://junjuewang.top/EarthVQA/                 ║"
        echo "  ║    Download and extract to:                                ║"
        echo "  ║    $EARTHVQA_DIR                                           ║"
        echo "  ║                                                            ║"
        echo "  ║  Expected structure after download:                        ║"
        echo "  ║    EarthVQA/                                               ║"
        echo "  ║      Train/images_png/   (image files)                     ║"
        echo "  ║      Val/images_png/     (image files)                     ║"
        echo "  ║      Test/images_png/    (image files)                     ║"
        echo "  ║      Train_QA.json                                         ║"
        echo "  ║      Val_QA.json                                           ║"
        echo "  ║      Test_QA.json                                          ║"
        echo "  ╚══════════════════════════════════════════════════════════════╝"
        echo ""
    fi
fi

echo ""
echo "========================================================"
echo "  DOWNLOAD SUMMARY"
echo "========================================================"
echo ""

# Summary
echo "  Dataset locations:"
echo "  ─────────────────"
echo "  DisasterM3:  $RAW_DIR/DisasterM3_Instruct/"
echo "  RSVLM-QA:    $RAW_DIR/RSVLM-QA/"
echo "  RSVQA-HR:    $RAW_DIR/RSVQA-HR/"
echo "  EarthVQA:    $RAW_DIR/EarthVQA/"
echo ""

# Verify each dataset
echo "  Status check:"
echo "  ─────────────"
for ds in "DisasterM3_Instruct" "RSVLM-QA" "RSVQA-HR" "EarthVQA"; do
    if [ -d "$RAW_DIR/$ds" ]; then
        count=$(find "$RAW_DIR/$ds" -type f | wc -l)
        echo "  ✅ $ds  ($count files)"
    else
        echo "  ❌ $ds  (NOT FOUND)"
    fi
done

echo ""
echo "========================================================"
echo "  NEXT STEPS"
echo "========================================================"
echo ""
echo "  1. If EarthVQA failed, follow the manual instructions above"
echo "  2. Set up the Python environment:"
echo "     cd /home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace"
echo "     bash setup_envs.sh"
echo "  3. Test that all datasets load correctly:"
echo "     source vlm_env_main/bin/activate"
echo "     python datasets_loader.py"
echo "  4. Run the full VLM benchmark:"
echo "     nohup bash run_all.sh > master_log.txt 2>&1 &"
echo ""
echo "  ✅  Done!  $(date)"
echo "========================================================"
