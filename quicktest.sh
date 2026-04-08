#!/bin/bash
# Quick end-to-end test: 1 model, 5 samples, stratified
cd "/home/aipmu/Datasets for VLM/DisasterM3_Eval"
source vlm_env_disasterm3/bin/activate

if [ -f ".env" ]; then
    export $(grep -v '^#' ".env" | xargs)
fi

python3 run_test.py --model moondream2 --max_samples 5 --stratified --output_prefix "quicktest" 2>&1 | tee "/home/aipmu/Datasets for VLM/DisasterM3_Eval/quicktest_output.txt"
