# Dependency-free retrieval scorer (no pyterrier/JDK needed).
# Computes MAP (AP) and R@5 with pytrec-eval semantics:
#   - mean is over the queries present in the QRELS file (19 for English dev;
#     the 13 NEI rumors have no qrel judgments and are excluded, i.e. they
#     earn no credit - matching the official evaluation)
#   - AP_q  = (sum over hit ranks r of  precision@r) / |gold_q|
#   - R@5_q = |gold_q ∩ top-5| / |gold_q|
#   - a qrel query missing from the run scores 0 on both metrics
# Usage is identical to scorer/retrieval_scorer.py.
import argparse
from collections import defaultdict
from csv import writer


def evaluate_run(pred_path, golden_path):
    qrels = defaultdict(set)
    with open(golden_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if rel > 0:
                qrels[qid].add(docid)

    runs = defaultdict(list)
    with open(pred_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            qid, _, rank, docid = parts[0], parts[1], int(parts[2]), parts[3]
            runs[qid].append((rank, docid))

    aps, r5s = [], []
    for qid in qrels:
        gold = qrels[qid]
        ranked = sorted(runs.get(qid, []))
        hits, prec_sum = 0, 0.0
        for r, docid in ranked:
            if docid in gold:
                hits += 1
                prec_sum += hits / r
        ap = prec_sum / len(gold)
        hits_at5 = sum(1 for _, d in ranked[:5] if d in gold)
        r5 = hits_at5 / len(gold)
        aps.append(ap)
        r5s.append(r5)

    n = len(qrels)
    return {"AP": sum(aps) / n, "R@5": sum(r5s) / n, "num_q": n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual_file", "-g", required=True, type=str,
                        help="Path to file with gold annotations (qrels).")
    parser.add_argument("--pred_file", "-p", required=True, type=str,
                        help="Path to TREC-format run file.")
    parser.add_argument("--output_file", required=False, type=str,
                        help="Path to csv results file.")
    args = parser.parse_args()

    eval_ = evaluate_run(args.pred_file, args.actual_file)
    print(f"MAP={eval_['AP']:.4f}  R@5={eval_['R@5']:.4f}  (over {eval_['num_q']} qrel queries)")
    if args.output_file:
        with open(args.output_file, "a") as f_object:
            writer_object = writer(f_object, delimiter="\t")
            writer_object.writerow(
                [args.pred_file.split("/")[-1], eval_["AP"], eval_["R@5"]]
            )
            f_object.close()
