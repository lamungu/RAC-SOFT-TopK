#!/bin/bash
# Prepare data for the DEFAULT "Our Approach" pipeline.
#
# The official task release (gitlab.com/checkthat_lab/clef2024-checkthat-lab,
# task5/data/) ships JSONL files (one JSON object per line) named *.json.
# This repo needs BOTH forms:
#   - data/main/English_{train,dev}.json  : JSON ARRAY  (json.load in
#     make_data/preprocessing.py and scorer/verification_scorer.py)
#   - data/main/*.jsonl                   : JSONL       (jsonlines in
#     inference_driver.py)
# Place the official files anywhere under data/main/ (as .json or .jsonl);
# this script normalizes both forms and runs the repo's preprocessing.
set -e
cd "$(dirname "$0")/.."

mkdir -p data/main data/preprocessed results/en

python - <<'EOF'
import json, os
try:
    import jsonlines
except ImportError:
    raise SystemExit("pip install jsonlines")

splits = ["English_train", "English_dev", "English_test"]
for s in splits:
    p_json, p_jsonl = f"data/main/{s}.json", f"data/main/{s}.jsonl"
    data = None
    for p in (p_json, p_jsonl):
        if not os.path.exists(p):
            continue
        try:
            data = json.load(open(p))          # JSON array
            break
        except json.JSONDecodeError:
            data = list(jsonlines.open(p))     # JSONL (official release)
            break
    if data is None:
        print(f"!! {s}: neither {p_json} nor {p_jsonl} found - "
              f"copy the official task5/data/{s}.json into data/main/")
        continue
    with jsonlines.open(p_jsonl, "w") as w:    # JSONL for inference_driver
        for rec in data:
            w.write(rec)
    if s != "English_test":                    # ARRAY for preprocessing + scorer
        json.dump(data, open(p_json, "w"), ensure_ascii=False)
    print(f"{s}: {len(data)} samples (array + jsonl written)")
EOF

# Preprocessing: remove links, emoji -> text (paper 5.2.2)
python make_data/preprocessing.py \
    --file_path ./data/main/English_train.json \
    --folder_path ./data/preprocessed/ --language en
python make_data/preprocessing.py \
    --file_path ./data/main/English_dev.json \
    --folder_path ./data/preprocessed/ --language en

# Sanity checks: label mapping + augmented dataset sizes (paper Table 1)
python - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from dataset import ContrastiveDataset, ClassifierDataset, JointDataset2

tr, dv = "./data/preprocessed/English_train.json", "./data/preprocessed/English_dev.json"
print(f"ContrastiveDataset train/val: "
      f"{len(ContrastiveDataset(tr, 3, 'en'))}/{len(ContrastiveDataset(dv, 3, 'en'))}  (paper: 233/48)")
print(f"ClassifierDataset  train/val: "
      f"{len(ClassifierDataset(tr, 'en'))}/{len(ClassifierDataset(dv, 'en'))}  (paper: 297/51)")
print(f"JointDataset2      train/val: "
      f"{len(JointDataset2(tr, 'en', 64))}/{len(JointDataset2(dv, 'en', 64))}  (paper: 305/41)")

pre = json.load(open("./data/preprocessed/meta.json"))
root = json.load(open("./meta.json"))
if pre != root:
    raise SystemExit("LABEL MAPPING MISMATCH: data/preprocessed/meta.json != meta.json "
                     "- JointDataset2 hard-codes NEI=2 and inference uses meta.json; "
                     "training/inference labels would disagree.")
print("label mapping OK:", root["labels_2_ids"])
EOF
