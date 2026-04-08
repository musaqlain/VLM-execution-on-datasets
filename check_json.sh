#!/bin/bash
cd "/home/aipmu/Datasets for VLM/DisasterM3_Eval"
source vlm_env_disasterm3/bin/activate
python3 -c "
import json
with open('results/quicktest_moondream2.json') as f:
    d = json.load(f)
print('=== METADATA ===')
for k,v in d['metadata'].items():
    print(f'  {k}: {v}')
print()
print('=== OVERALL METRICS ===')
for k,v in d['metrics_overall'].items():
    print(f'  {k}: {v:.4f}')
print()
print('=== BY TASK ===')
for task, m in d['metrics_by_task'].items():
    print(f'  {task}: {m}')
print()
print('=== SAMPLE PREDICTION #1 ===')
p = d['predictions'][0]
for k,v in p.items():
    if k != 'prediction_raw':
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: {repr(v)[:80]}')
print()
print('JSON is valid and machine-readable ✅')
" 2>&1 | tee "/home/aipmu/Datasets for VLM/DisasterM3_Eval/json_check.txt"
