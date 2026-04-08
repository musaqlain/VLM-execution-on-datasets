#!/bin/bash
# Deep inspection of DisasterM3 benchmark for evaluation redesign
cd "/home/aipmu/Datasets for VLM/DisasterM3_Eval"
source vlm_env_disasterm3/bin/activate

python3 -c "
import json, os
from collections import Counter, defaultdict

BENCH_JSON = '/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Bench/benchmark_release.json'
with open(BENCH_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total entries: {len(data)}')

# 1. ALL FIELD NAMES
print(f'\\n=== ALL FIELD NAMES (from first entry) ===')
for k, v in data[0].items():
    print(f'  {k}: {type(v).__name__} -> {repr(v)[:120]}')

# 2. Task type distribution
tasks = Counter(e.get('task','MISSING') for e in data)
print(f'\\n=== TASK TYPES ({len(tasks)}) ===')
for task, count in tasks.most_common():
    print(f'  {count:6d}  {task}')

# 3. Field coverage
print(f'\\n=== FIELD COVERAGE ===')
all_keys = set()
for e in data:
    all_keys.update(e.keys())
for key in sorted(all_keys):
    has = sum(1 for e in data if e.get(key))
    print(f'  {key:30s}: {has:6d}/{len(data)}')

# 4. GT answer format per task type
print(f'\\n=== GT ANSWER FORMAT PER TASK ===')
by_task = defaultdict(list)
for e in data:
    by_task[e.get('task','MISSING')].append(e)

for task, entries in sorted(by_task.items()):
    gt_opts = [e.get('ground_truth_option','') for e in entries]
    has_opt = sum(1 for g in gt_opts if g and g != 'N/A')
    multi = sum(1 for g in gt_opts if g and ',' in g)
    single = has_opt - multi
    # Check option count distribution
    opt_counts = Counter()
    for g in gt_opts:
        if g and g != 'N/A':
            letters = [x.strip().rstrip('.') for x in g.split(',')]
            letters = [x for x in letters if len(x)==1 and x.isalpha()]
            opt_counts[len(letters)] += 1
    print(f'\\n  {task} ({len(entries)} entries):')
    print(f'    has_option: {has_opt}, single_label: {single}, multi_label: {multi}')
    print(f'    option_count_dist: {dict(opt_counts)}')
    # Show 2 sample GTs
    for e in entries[:2]:
        print(f'    GT_option: {repr(e.get(\"ground_truth_option\",\"\"))[:60]}')
        print(f'    GT_text:   {repr(e.get(\"ground_truth\",\"\"))[:80]}')
        print(f'    options_str: {repr(e.get(\"options_str\",\"\"))[:80]}')
        print(f'    prompt:    {repr(e.get(\"prompts\",\"\"))[:80]}')

# 5. Image coverage
print(f'\\n=== IMAGE PATHS ===')
has_pre = sum(1 for e in data if e.get('pre_image_path'))
has_post = sum(1 for e in data if e.get('post_image_path'))
# Check how many images actually exist on disk
RAW = '/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Bench'
exist_pre = 0
exist_post = 0
for e in data[:200]:  # sample check
    pre = e.get('pre_image_path','')
    post = e.get('post_image_path','')
    if pre:
        p = os.path.join(RAW, pre.replace('\\\\','/'))
        if os.path.exists(p): exist_pre += 1
    if post:
        p = os.path.join(RAW, post.replace('\\\\','/'))
        if os.path.exists(p): exist_post += 1
print(f'  pre_image_path present:  {has_pre}/{len(data)}')
print(f'  post_image_path present: {has_post}/{len(data)}')
print(f'  pre exists (of 200):     {exist_pre}/200')
print(f'  post exists (of 200):    {exist_post}/200')

# 6. Directory listing of images
print(f'\\n=== IMAGE DIRECTORIES ===')
for d in ['test_images', 'masks']:
    full = os.path.join(RAW, d)
    if os.path.isdir(full):
        subdirs = os.listdir(full)
        count = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full,f))])
        print(f'  {d}/ -> {len(subdirs)} items ({count} files)')
        for sd in subdirs[:5]:
            sdp = os.path.join(full, sd)
            if os.path.isdir(sdp):
                fc = len(os.listdir(sdp))
                print(f'    {sd}/ -> {fc} files')
    else:
        print(f'  {d}/ -> NOT FOUND')

# 7. Unique option counts per options_str
print(f'\\n=== OPTION COUNTS ===')
opt_cnt = Counter()
for e in data:
    os_str = e.get('options_str','')
    if os_str:
        opts = [x.strip() for x in os_str.split(',') if '.' in x]
        opt_cnt[len(opts)] += 1
print(f'  Distribution of # of options in options_str: {dict(opt_cnt.most_common(10))}')

# 8. Prompt format analysis
print(f'\\n=== PROMPT FORMAT ===')
prompt_types = Counter()
for e in data:
    p = e.get('prompts','')
    if isinstance(p, list):
        prompt_types['list'] += 1
    elif isinstance(p, str):
        prompt_types['string'] += 1
    else:
        prompt_types[type(p).__name__] += 1
print(f'  Prompt data types: {dict(prompt_types)}')
# Show a list-type prompt
for e in data:
    if isinstance(e.get('prompts'), list):
        print(f'  Sample list prompt: {repr(e[\"prompts\"])[:200]}')
        break
" 2>&1 | tee "/home/aipmu/Datasets for VLM/DisasterM3_Eval/benchmark_analysis.txt"
