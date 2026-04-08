import os
import glob
root_dir = r"\\wsl.localhost\Ubuntu\home\aipmu\Datasets for VLM\Raw dataset files\RSVLM-QA"
for root, dirs, files in os.walk(root_dir):
    for f in files:
        if f.endswith(".json"):
            print(os.path.join(root, f).replace(root_dir, ""))
