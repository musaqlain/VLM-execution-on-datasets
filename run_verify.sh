#!/bin/bash
cd "/home/aipmu/Datasets for VLM/DisasterM3_Eval"
source vlm_env_disasterm3/bin/activate
python3 verify_metrics.py 2>&1 | tee "/home/aipmu/Datasets for VLM/DisasterM3_Eval/verification_report.txt"
