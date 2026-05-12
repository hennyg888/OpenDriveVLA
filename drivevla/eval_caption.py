"""
Score VLA nuCaption predictions using the lidargpt3d-style scorer:
  - Corpus BLEU-1/2/3/4 (lidargpt3d's custom corpus impl, lowercased
    word-only tokens, ~equivalent to pycocoevalcap + PTBTokenizer)
  - ROUGE-L (per-sample LCS F1, averaged)
  - CIDEr (TF-IDF cosine over n-grams 1-4)
  - BERTScore P/R/F1 (default model: bert-base-uncased, matches the
    paper convention)

All metrics are reported on a 0-100 scale. The previous nltk
sentence-BLEU + roberta-large pipeline is dropped — sentence-BLEU is
non-standard for image captioning and is incomparable to published
nuCaption numbers; roberta-large vs bert-base-uncased is also a
moving target. Switching to this scorer aligns with both
`lidargpt3d/evaluation/metrics/text_metrics.py` and the OpenDriveVLA
paper's tables.

Inputs (unchanged):
  --pred-jsonl       JSONL from inference_drivevla.py.
  --val-caption-json Output of val_caption_convertor.py.

Outputs:
  --out          (optional) JSON dump of the metrics summary.
  --result-file  (optional) per-row dump (token<sp><sp><sp><sp>q<sp>...<sp>gt).
"""

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_LIDARGPT3D = os.path.join(_REPO_ROOT, 'lidargpt3d')
if _LIDARGPT3D not in sys.path:
    sys.path.insert(0, _LIDARGPT3D)

from evaluation.metrics.text_metrics import (
    corpus_bleu, rouge_l_score, CiderScorer, BERTScoreScorer,
)
from evaluation.metrics.common import postprocess_generation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-jsonl', required=True,
                    help='JSONL from inference_drivevla.py.')
    ap.add_argument('--val-caption-json', required=True,
                    help='Output of val_caption_convertor.py.')
    ap.add_argument('--out', default=None,
                    help='Optional JSON dump of the metrics summary.')
    ap.add_argument('--result-file', default=None,
                    help='Optional per-row dump (token    q    pred    gt).')
    ap.add_argument('--bert-model', default='bert-base-uncased',
                    help='HF model id or local path; default matches '
                         'lidargpt3d / OpenDriveVLA paper convention.')
    ap.add_argument('--bert-device', default='cuda')
    ap.add_argument('--bertscore-batch-size', type=int, default=64)
    ap.add_argument('--skip-bertscore', action='store_true',
                    help='Skip BERTScore (e.g. offline runs).')
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
            preds[rec['id']] = postprocess_generation(a)

    with open(args.val_caption_json) as f:
        val_entries = json.load(f)

    pred_list, ref_list, rows = [], [], []
    missing = 0
    for entry in val_entries:
        qid = entry['id']
        if qid not in preds:
            missing += 1
            continue
        pred = preds[qid]
        gt   = str(entry['gt_answer'])
        pred_list.append(pred)
        ref_list.append([gt])
        rows.append((entry['sample_token'], entry['question'], pred, gt))

    if missing:
        print(f"WARNING: {missing}/{len(val_entries)} predictions missing; "
              f"metrics computed over the {len(pred_list)} present pairs.")

    print(f"Scoring {len(pred_list)} pairs ...")

    print("=== corpus BLEU ===")
    bleu = corpus_bleu(pred_list, ref_list, max_order=4)
    for k in ('bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'):
        print(f"  {k} = {bleu[k]:.2f}")

    print("=== ROUGE-L ===")
    rouge_scores = [
        max((rouge_l_score(p, r) for r in refs), default=0.0)
        for p, refs in zip(pred_list, ref_list)
    ]
    rouge_l_mean = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    print(f"  ROUGE-L = {rouge_l_mean:.2f}")

    print("=== CIDEr (building doc-frequency over all refs) ===")
    cider = CiderScorer(ref_list)
    cider_scores = [cider.score(p, r) for p, r in zip(pred_list, ref_list)]
    cider_mean = float(np.mean(cider_scores)) if cider_scores else 0.0
    print(f"  CIDEr = {cider_mean:.4f}")

    bs_p = bs_r = bs_f1 = None
    if not args.skip_bertscore:
        print(f"=== BERTScore (model={args.bert_model}, "
              f"device={args.bert_device}, bs={args.bertscore_batch_size}) ===")
        scorer = BERTScoreScorer(
            model_name_or_path=args.bert_model,
            device=args.bert_device,
            batch_size=args.bertscore_batch_size,
        )
        bs = scorer.score(pred_list, ref_list)
        bs_p  = bs['bertscore_precision']
        bs_r  = bs['bertscore_recall']
        bs_f1 = bs['bertscore_f1']
        print(f"  BERTScore P  = {bs_p:.2f}")
        print(f"  BERTScore R  = {bs_r:.2f}")
        print(f"  BERTScore F1 = {bs_f1:.2f}")

    summary = {
        'missing':  missing,
        'total_gt': len(val_entries),
        'scored':   len(pred_list),
        'BLEU_1':   bleu['bleu-1'],
        'BLEU_2':   bleu['bleu-2'],
        'BLEU_3':   bleu['bleu-3'],
        'BLEU_4':   bleu['bleu-4'],
        'ROUGE_L':  rouge_l_mean,
        'CIDEr':    cider_mean,
        'bertscore_p':  bs_p,
        'bertscore_r':  bs_r,
        'bertscore_f1': bs_f1,
        'bert_model':   args.bert_model if not args.skip_bertscore else None,
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote metrics json -> {args.out}")

    if args.result_file:
        with open(args.result_file, 'w') as f:
            for token, q, pred, gt in rows:
                f.write(f"{token}    {q}    {pred}    {gt}\n")
        print(f"Wrote result dump -> {args.result_file}")


if __name__ == '__main__':
    main()
