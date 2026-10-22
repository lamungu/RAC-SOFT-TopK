# Reproducing the paper's dev results with this repo

Target: **Table 2, "Our Approach" row** (ColBERT-B retriever + BERT-B classifier + MaxSim),
validation set: **MAP 0.733, R@5 0.778, M-F1 0.472, Strict-M-F1 0.472**.
(Test set, Table 3: MAP 0.559, R@5 0.634, M-F1 0.482, Strict-M-F1 0.454.)

## Pipeline (paper §5.2.3 / Algorithm 1 → repo scripts)

| Phase | Paper | Repo command |
|---|---|---|
| Data | links removed, emoji→text (§5.2.2) | `bash scripts/prepare_data.sh` |
| 1. Independent classifier | D_C pairs, bs 2, lr 1e-5, 5 epochs, BERT-B (§5.2.1/§5.2.3) | `python classifier_train.py --config configs/approach/phase1_classifier.yaml` |
| 2. Joint end-to-end | D_J doc-sets (n=64), bs 1, lr 1e-5, k=5, ε=0.01, 5 epochs; R=colbert-ir/colbertv2.0, H=phase-1 weights | `python joint_train.py --config configs/approach/phase2_joint.yaml` |
| 3. Frozen-retriever polish | D_J, bs 1, lr 1e-5, 5 epochs, retriever frozen | `python joint_train.py --config configs/approach/phase3_frozen_retriever.yaml` |
| Inference | hard top-5, score-weighted mean of per-evidence logits, argmax | `python inference_driver.py --config configs/approach/infer_dev.yaml` |
| Scoring | MAP/R@5 (qrels), M-F1/Strict-M-F1 | `scorer/retrieval_scorer_stdlib.py` (no deps) or `scorer/retrieval_scorer.py` (pyterrier+JDK), `scorer/verification_scorer.py` |

One-shot: `bash scripts/approach.sh`. Code-level optimizer details not stated in the paper
but present in `trainers.py`: Adam with weight_decay=0.01, `CosineAnnealingLR(T_max=epochs)`,
seed 42, and **final-epoch weights are saved** (no best-model checkpointing; with patience=5
and epochs=5 early stopping can never trigger, so "5 epochs (early stopping)" ≡ exactly 5
full epochs). Note: `max_grad_norm` is inert in the released code (see discrepancy 8).

Losses (`models/rac_model.py`): `loss = density + CE`, equal weight, matching paper Eq. 7.
`epsilon: 8e-8` is a numerical stabilizer inside `DensityLoss`; the paper's soft-top-k
entropy `ε=0.01` is `soft_k_epsilon` (OT/Sinkhorn, 200 iterations).

## Data format (important)

The official release is **JSONL** (`timeline`/`evidence` items are `[profile_url, tweet_id, text]`),
but the repo is inconsistent about formats:

- `make_data/preprocessing.py` and `scorer/verification_scorer.py` use `json.load` → need a
  **JSON array** at `data/main/English_{train,dev}.json`
- `inference_driver.py` uses `jsonlines` → needs **JSONL** (`data/main/*.jsonl`)
- `inference.py` writes predictions as **JSONL**, but `verification_scorer.py` reads them with
  `json.load` → `scripts/approach.sh` converts the prediction file to an array before scoring

`scripts/prepare_data.sh` normalizes all of this (works from the official files placed as
`.json` or `.jsonl`) and verifies that `data/preprocessed/meta.json` equals root `meta.json`
(refutes=0, supports=1, not enough info=2 — required because `JointDataset2` hard-codes NEI
label 2 and inference decodes labels via `meta.json`).

## Paper-vs-released-code discrepancies (flagged)

1. **Augmented dataset counts differ from Table 1.** The released dataset classes on the
   current official data give train/val: contrastive 276/57, classifier pairs 344/81, joint
   doc-sets 362/102 — vs the paper's 233/48, 297/51, 305/41 (an older data snapshot). The
   regimen is identical either way, but exact dev scores may shift. `prepare_data.sh`
   prints the counts so you can see which snapshot your data is.
2. **`score_fn: maxsim` is not ColBERT MaxSim.** `MaxSimScores` computes, per document, the
   *mean over doc tokens* of the *max over query tokens* cosine (ColBERT: sum over query
   tokens of max over doc tokens). The paper calls it MaxSim; the reported numbers were made
   with this code, so the code variant is used. **Kept as-is.**
3. **Aggregation differs from Eq. 4.** Code: `softmax(Σ sᵢ·logitsᵢ / mean(s))` (weighted mean
   of *logits*, softmaxed); paper writes a convex combination of *probabilities*. Inference
   matches training, so this is internally consistent. **Kept as-is.**
4. **The shipped `configs/joint_scratch/joint_config.yaml` ("main") is Phase 3 only**, points
   at `./logs/en/joint_first1/checkpoints` (a Phase-2 run not shipped in the repo), sets the
   classifier fallback to colbertv2.0, and the shipped `infer.yaml`/`evaluate_*.sh` point at
   yet other log/result dirs — `scripts/main.sh` is not runnable as shipped. The configs in
   `configs/approach/` are the consistent, self-contained version of the full 3-phase approach.
5. **Shipped independent configs differ from the paper's independent rows** (retriever:
   `loss_fn: cosine` vs the paper's MaxSim; classifier: colbert backbone, batch 1 vs the
   paper's bert batch 2 / colbert batch 2). Not needed for "Our Approach" (no independent
   retriever training), noted for completeness.
6. **`JointDataset2`** caps evidence per doc-set at 5 (chunks of 5 for larger evidence
   lists); `DensityLoss` compares the first `min(k, #evidence)` rank buckets against the
   one-hot evidence rows. **Kept as-is.**
7. **Inference bug (fixed in this repo):** `infer_retriever` did
   `if data_sample.pop("label", None): ... data_sample["label"]` — the key was already
   popped, so inference **crashed with KeyError on any sample with a truthy label** (all
   SUPPORTS / NOT-ENOUGH-INFO samples). It only ever "worked" on the unlabeled test set.
   Fixed by capturing the popped value; test-set behavior is unchanged.
8. **Gradient clipping is inert in the released code (kept as-is):** `trainers.py` passes
   `self.model.parameters()` (a generator) to `optim.Adam`, which consumes it; the later
   `clip_grad_norm_(parameters, ...)` iterates an exhausted generator (the "empty generator"
   warning) — so `max_grad_norm` never clipped. The paper's numbers were produced in this
   state (Adam, wd=0.01, no effective clipping); do not "fix" it for an exact reproduction.

## Environment & validation

- **GPU box (uv):** `uv sync && source .venv/bin/activate` — `pyproject.toml` (+ `uv.lock`)
  pins the paper's stack (Python 3.10, torch 2.0.1 = PyPI/CUDA 11.8 build, transformers
  4.39.3; `models/bert_seq_cls.py` is a copy of the 4.39.3 BERT implementation). Single GPU,
  32 GB-class; paper reports ~1 h for the whole 3-phase training. (The pipeline has also
  been smoke-tested here with torch 2.12 / transformers 4.57.)
- **Scoring:** `scorer/retrieval_scorer_stdlib.py` computes MAP/R@5 with pytrec-eval
  semantics and no dependencies (no pyterrier/JDK). The original `scorer/retrieval_scorer.py`
  (pyterrier) is kept unchanged and is an optional `uv sync --extra terrier`.
- **qrels semantics (matters!):** only 19 of the 32 dev rumors have qrel judgments; the 13
  NEI rumors earn no credit and are *excluded from the denominator* (mean over qrel queries)
  — this is how a dev MAP of 0.733 is possible.
- **The released-weights Drive link in the README is broken** (checked 2026-10), so the
  paper's numbers can only be validated by re-training: run `bash scripts/approach.sh` and
  compare against Table 2.
- **Smoke test (done, CPU):** a 6-sample subset through all 3 phases + inference + both
  scorers passes, including the checkpoint hand-off between phases (HF-saved classifier
  checkpoint loaded by the local model class, and vice versa) and the frozen-retriever phase.
  Joint steps run at ~4-14 s/step on CPU — a full run (362+102 joint samples × 10 epochs)
  is why you need the GPU box.
- **Silver test qrels** (`test_qrels_silver.txt`) are optional: `approach.sh` includes
  commented test-set commands. Silver = partial gold, so test numbers will not match
  Table 3 (which used the hidden official qrels).

## Other Table 2 rows (if needed)

- **Joint Training from Scratch** (MAP 0.606, R@5 0.662, M-F1 0.362): skip Phase 1 and the
  Phase-3 polish — one joint run initialized from raw HF weights
  (`retriever_path: None`, `classifier_path: None`, `freeze_*: False`, 5 epochs).
- **Independent Training**: `configs/independent/{retriever,classifier}/config.yaml`
  (contrastive / pair classification), then inference with `configs/infer.yaml`-style paths.
