"""
Score VLA NuX predictions with pycocoevalcap (BLEU-4, METEOR, ROUGE_L,
CIDEr) on three views of each pair:
  - full:      raw pred string vs raw GT string ("Narration:...\nReasoning:...")
  - narration: parsed narration text only
  - reasoning: parsed reasoning text only

Parsing uses a lenient cross-line regex (re.DOTALL):
  ^...Narration:\s*(.*?)\s*\nReasoning:\s*(.*?)\.?\s*$
Mirrors Hint-AD's design but fixes the cross-line issue in their original
pattern (`.*?\.` without re.DOTALL never matches our format, where the
narration line has no trailing period). When pred regex fails the entry
is dropped from the narration/reasoning buckets but still counted in
"full".

Inputs:
  --pred-jsonl       JSONL from inference_drivevla.py (one {id, question,
                     answer} per line; answer is a list[str]).
  --val-nux-json     Output of val_nux_convertor.py (gt_narration,
                     gt_reasoning, gt_answer, sample_token, question).

Outputs:
  --out          (optional) JSON dump of the metrics summary.
  --result-file  (optional) per-row dump (token<sp><sp><sp><sp>pred<sp>...<sp>gt).
"""

import argparse
import json
import re

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# pycocoevalcap.meteor uses a Java subprocess that deadlocks on large
# (>~10k) batches via stdio buffering. Switch to nltk's pure-Python
# METEOR (per-sample, then averaged); numbers match Java METEOR within
# ~1% in our spot checks.
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
from nltk.translate.meteor_score import single_meteor_score


SPLIT_RE = re.compile(
    r"Narration:\s*(.*?)\s*\nReasoning:\s*(.*?)\.?\s*$",
    re.DOTALL,
)


def split_caption(text: str):
    """Return (narration, reasoning) or (None, None) if parse fails."""
    if not text:
        return None, None
    m = SPLIT_RE.search(text)
    if not m:
        return None, None
    n = m.group(1).strip()
    r = m.group(2).strip().rstrip('.').strip()
    if not n or not r:
        return None, None
    return n, r


def score_one(pairs):
    """pairs: list[(pred, gt)] -> dict of {BLEU_4, METEOR, ROUGE_L, CIDEr}."""
    if not pairs:
        return {"BLEU_4": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0,
                "n": 0}
    refs  = {i: [g if g else " "] for i, (_, g) in enumerate(pairs)}
    hypos = {i: [p if p else " "] for i, (p, _) in enumerate(pairs)}

    out = {"n": len(pairs)}
    bleu_score, _   = Bleu(4).compute_score(refs, hypos)
    out["BLEU_4"]   = float(bleu_score[3])
    rouge_score, _  = Rouge().compute_score(refs, hypos)
    out["ROUGE_L"]  = float(rouge_score)
    cider_score, _  = Cider().compute_score(refs, hypos)
    out["CIDEr"]    = float(cider_score)

    # METEOR via nltk (per-sample, averaged). Tokenize with .split() to
    # match the upstream pycocoevalcap convention.
    meteor_scores = []
    for p, g in pairs:
        ref_toks = (g if g else " ").split()
        hyp_toks = (p if p else " ").split()
        try:
            meteor_scores.append(single_meteor_score(ref_toks, hyp_toks))
        except Exception:
            meteor_scores.append(0.0)
    out["METEOR"] = float(np.mean(meteor_scores)) if meteor_scores else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-jsonl', required=True)
    ap.add_argument('--val-nux-json', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--result-file', default=None)
    args = ap.parse_args()

    preds = {}
    with open(args.pred_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            a = rec['answer']
            if isinstance(a, list):
                a = a[0] if a else ''
            preds[rec['id']] = a

    with open(args.val_nux_json) as f:
        val_entries = json.load(f)

    full_pairs, narr_pairs, reas_pairs, rows = [], [], [], []
    missing = 0
    pred_split_failed = 0
    for entry in val_entries:
        qid = entry['id']
        if qid not in preds:
            missing += 1
            continue
        pred = preds[qid]
        gt_full = str(entry['gt_answer'])
        gt_n = entry.get('gt_narration')
        gt_r = entry.get('gt_reasoning')

        full_pairs.append((pred, gt_full))

        pred_n, pred_r = split_caption(pred)
        if pred_n is None or pred_r is None:
            pred_split_failed += 1
        elif gt_n and gt_r:
            narr_pairs.append((pred_n, gt_n))
            reas_pairs.append((pred_r, gt_r))

        rows.append((entry['sample_token'], pred, gt_full))

    if missing:
        print(f"WARNING: {missing}/{len(val_entries)} predictions missing.")
    if pred_split_failed:
        print(f"INFO: {pred_split_failed}/{len(full_pairs)} pred outputs "
              f"could not be split into narration/reasoning; those are "
              f"excluded from the narration/reasoning buckets but still "
              f"counted in 'full'.")

    print("\n=========== Scoring full ===========")
    full_scores = score_one(full_pairs)
    print(json.dumps({k: full_scores[k] for k in ("BLEU_4","METEOR","ROUGE_L","CIDEr")}, indent=2))

    print("\n=========== Scoring narration ===========")
    narr_scores = score_one(narr_pairs)
    print(json.dumps({k: narr_scores[k] for k in ("BLEU_4","METEOR","ROUGE_L","CIDEr")}, indent=2))

    print("\n=========== Scoring reasoning ===========")
    reas_scores = score_one(reas_pairs)
    print(json.dumps({k: reas_scores[k] for k in ("BLEU_4","METEOR","ROUGE_L","CIDEr")}, indent=2))

    summary = {
        "missing":           missing,
        "total_gt":          len(val_entries),
        "scored_full":       full_scores["n"],
        "scored_split_ok":   narr_scores["n"],
        "pred_split_failed": pred_split_failed,
        "full":      {k: full_scores[k] for k in ("BLEU_4","METEOR","ROUGE_L","CIDEr")},
        "narration": {k: narr_scores[k] for k in ("BLEU_4","METEOR","ROUGE_L","CIDEr")},
        "reasoning": {k: reas_scores[k] for k in ("BLEU_4","METEOR","ROUGE_L","CIDEr")},
    }

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote metrics json -> {args.out}")

    if args.result_file:
        with open(args.result_file, 'w') as f:
            for token, pred, gt in rows:
                pred_one_line = pred.replace('\n', ' \\n ')
                gt_one_line   = gt.replace('\n', ' \\n ')
                f.write(f"{token}    {pred_one_line}    {gt_one_line}\n")
        print(f"Wrote result dump -> {args.result_file}")


if __name__ == '__main__':
    main()
