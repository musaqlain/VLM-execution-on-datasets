#!/bin/bash
cd "/home/aipmu/Datasets for VLM/DisasterM3_Eval"
source vlm_env_disasterm3/bin/activate
echo "=== Testing dataset_loader ==="
python3 -c "
from dataset_loader import load_disasterm3_bench
items = load_disasterm3_bench(max_samples=10, stratified=True)
print(f'\nLoaded {len(items)} items')
for i, item in enumerate(items):
    print(f'  [{i}] track={item[\"eval_track\"]:20s} task={item[\"task_type\"][:35]:35s} gt_opts={item[\"ground_truth_options\"]} ans={item[\"answer\"][:60]}')
"
echo ""
echo "=== Testing evaluation parser ==="
python3 -c "
from evaluation import parse_option_letters, single_label_accuracy, multi_label_metrics, free_text_metrics
# Test option parser
tests = [
    ('B', ['B']),
    ('B. Buildings', ['B']),
    ('Answer: B', ['B']),
    ('B, E, F', ['B', 'E', 'F']),
    ('I think the answer is B and E', ['B', 'E']),
    ('Answer: F. Buildings', ['F']),
    ('H', ['H']),
]
print('Option parser tests:')
for raw, expected in tests:
    result = parse_option_letters(raw, max_options=8)
    status = '✅' if result == expected else '❌'
    print(f'  {status} parse(\"{raw}\") -> {result} (expected {expected})')

# Test single-label accuracy
print(f'\\nSingle-label accuracy:')
print(f'  match: {single_label_accuracy([\"B\"], [\"B\"])}')
print(f'  miss:  {single_label_accuracy([\"A\"], [\"B\"])}')

# Test multi-label metrics
print(f'\\nMulti-label metrics:')
m = multi_label_metrics([\"B\"], [\"B\", \"E\", \"F\", \"G\"])
print(f'  pred=[B] gt=[B,E,F,G] -> P={m[\"precision\"]:.2f} R={m[\"recall\"]:.2f} F1={m[\"f1\"]:.2f}')
m2 = multi_label_metrics([\"B\", \"E\", \"F\", \"G\"], [\"B\", \"E\", \"F\", \"G\"])
print(f'  pred=[B,E,F,G] gt=[B,E,F,G] -> P={m2[\"precision\"]:.2f} R={m2[\"recall\"]:.2f} F1={m2[\"f1\"]:.2f}')

# Test free text
print(f'\\nFree-text metrics:')
ft = free_text_metrics('Buildings are damaged', 'Buildings show significant damage')
print(f'  BLEU1={ft[\"bleu1\"]:.3f} ROUGE-L={ft[\"rougeL\"]:.3f}')
"
echo ""
echo "=== All tests done ==="
