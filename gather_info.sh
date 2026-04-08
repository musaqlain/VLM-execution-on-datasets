#!/bin/bash
echo "===== GPU INFO ====="
nvidia-smi 2>/dev/null || echo "nvidia-smi not found"

echo ""
echo "===== PYTHON VERSION ====="
python3 --version

echo ""
echo "===== FREE MEMORY ====="
free -h

echo ""
echo "===== DISK SPACE ====="
df -h /home/aipmu

echo ""
echo "===== BENCHMARK JSON SCHEMA ====="
python3 -c "
import json
with open('/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Bench/benchmark_release.json') as f:
    data = json.load(f)
print('Type:', type(data).__name__)
if isinstance(data, list):
    print('Total entries:', len(data))
    if data:
        print('First entry keys:', list(data[0].keys()))
        import json as j
        print('First entry sample:')
        print(j.dumps(data[0], indent=2, ensure_ascii=False)[:2000])
elif isinstance(data, dict):
    print('Top-level keys:', list(data.keys()))
    for k,v in data.items():
        if isinstance(v, list):
            print(f'  {k}: list of {len(v)} items')
            if v and isinstance(v[0], dict):
                print(f'    First item keys: {list(v[0].keys())}')
                import json as j
                print(f'    First item sample:')
                print(j.dumps(v[0], indent=2, ensure_ascii=False)[:1000])
        else:
            print(f'  {k}: {type(v).__name__} = {str(v)[:200]}')
"

echo ""
echo "===== TEST IMAGES COUNT ====="
ls "/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Bench/test_images/" | wc -l

echo ""
echo "===== TRAIN IMAGES COUNT ====="
ls "/home/aipmu/Datasets for VLM/Raw dataset files/DisasterM3_Instruct/train_images/" | wc -l

echo ""
echo "===== EXISTING ENVS ====="
which conda 2>/dev/null || echo "no conda"
ls "/home/aipmu/Datasets for VLM/DisasterM3_Eval/" 2>/dev/null || echo "DisasterM3_Eval does not exist yet"

echo ""
echo "===== DONE ====="
