#!/bin/bash
cd "/home/aipmu/Datasets for VLM/DisasterM3_Eval"
source vlm_env_disasterm3/bin/activate
python3 -c "
from dataset_loader import load_disasterm3_bench
items = load_disasterm3_bench(max_samples=10, stratified=True)
print(f'Loaded {len(items)} items')
for i, item in enumerate(items):
    print(f'  [{i}] track={item[\"eval_track\"]:20s} task={item[\"task_type\"][:40]}')

from evaluation import parse_option_letters
tests = [
    ('B', ['B']),
    ('B. Buildings', ['B']),
    ('Answer: B', ['B']),
    ('B, E, F', ['B', 'E', 'F']),
    ('Answer: F. Buildings', ['F']),
    ('H', ['H']),
]
print()
all_pass = True
for raw, expected in tests:
    result = parse_option_letters(raw, max_options=8)
    ok = result == expected
    if not ok: all_pass = False
    print(f'  {\"PASS\" if ok else \"FAIL\"} parse(\"{raw}\") -> {result}')
print(f'Parser tests: {\"ALL PASS\" if all_pass else \"SOME FAILED\"} ')
" 2>&1 | tee "/home/aipmu/Datasets for VLM/DisasterM3_Eval/test_output.txt"
