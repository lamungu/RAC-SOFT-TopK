#!/bin/bash
# DEFAULT "Our Approach" (paper 5.2.3) - full pipeline.
# Target dev scores (paper Table 2, ColBERT-B + BERT-B + MaxSim row):
#   MAP 0.733 | R@5 0.778 | M-F1 0.472 | Strict-M-F1 0.472
#
# Phase 1: independent classifier pre-training   (5 epochs, bs 2, lr 1e-5)
# Phase 2: joint end-to-end training             (5 epochs, bs 1, lr 1e-5, k=5, eps=0.01)
# Phase 3: classifier fine-tune, retriever frozen (5 epochs, bs 1, lr 1e-5)
# Then: inference on dev + retrieval/verification scoring.
# Setup on the GPU box: uv sync && source .venv/bin/activate  (pyproject.toml)
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --- data ---
bash scripts/prepare_data.sh

# --- training (sequential; each phase consumes the previous checkpoint) ---
echo "=== Phase 1: independent classifier (bert-base-uncased) ==="
python classifier_train.py --config ./configs/approach/phase1_classifier.yaml

echo "=== Phase 2: joint end-to-end (colbert retriever + bert classifier) ==="
python joint_train.py --config ./configs/approach/phase2_joint.yaml

echo "=== Phase 3: frozen retriever, classifier fine-tune ==="
python joint_train.py --config ./configs/approach/phase3_frozen_retriever.yaml

# --- inference on dev (JSONL input; outputs TREC run + JSONL predictions) ---
echo "=== Inference on dev ==="
python inference_driver.py --config ./configs/approach/infer_dev.yaml

# verification_scorer reads predictions with json.load -> convert JSONL to array
python - <<'EOF'
import json, jsonlines
pred = list(jsonlines.open("./results/en/approach_dev.json"))
json.dump(pred, open("./results/en/approach_dev_arr.json", "w"), ensure_ascii=False)
print(f"converted {len(pred)} predictions to ./results/en/approach_dev_arr.json")
EOF

# --- scoring (dev) ---
# stdlib scorer: MAP/R@5 with pytrec-eval semantics, no pyterrier/JDK needed
# (the original scorer/retrieval_scorer.py needs pyterrier + a JDK)
echo "=== Scoring (dev): expect MAP 0.733, R@5 0.778, M-F1 0.472, Strict 0.472 ==="
python scorer/verification_scorer.py \
    --actual_file ./data/main/English_dev.json \
    --pred_file ./results/en/approach_dev_arr.json \
    --out_file ./verification.csv

python scorer/retrieval_scorer_stdlib.py \
    --pred_file ./results/en/approach_dev.txt \
    --actual_file ./data/main/dev_qrels.txt \
    --output_file ./retrieval.csv

# optional: test set (silver qrels - partial gold; numbers differ from paper Table 3)
# python inference_driver.py --config ./configs/approach/infer_test.yaml
# python -c "import json,jsonlines; json.dump(list(jsonlines.open('./results/en/approach_test.json')), open('./results/en/approach_test_arr.json','w'), ensure_ascii=False)"
# python scorer/verification_scorer.py --actual_file ./data/main/English_test.json --pred_file ./results/en/approach_test_arr.json --out_file ./verification.csv
# python scorer/retrieval_scorer_stdlib.py --pred_file ./results/en/approach_test.txt --actual_file ./data/main/test_qrels_silver.txt --output_file ./retrieval.csv
