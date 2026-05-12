"""
Score VLA NuScenes-QA predictions against the val ground truth using the
exact semantics of the upstream scorer at
NuScenes-QA/src/execution/result_eval.py::Eval:

  - raw string equality (no .strip() / .lower() / normalization)
  - three buckets: 'Overall', template_type, template_type_num_hop
  - print as "%d / %d = %.2f" (two-decimal percentage)
  - optional token<sp><sp><sp><sp>q<sp><sp><sp><sp>pred<sp><sp><sp><sp>gt
    row dump (the upstream result_eval_file)

We cannot call upstream Eval directly: it expects a classifier
`ix2ans`/`ans_ix_list`. Our VLA produces free text. Accounting is
byte-identical; only the input format differs.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-jsonl', required=True,
                    help='JSONL from inference_drivevla.py (one {id,question,answer} per line)')
    ap.add_argument('--val-qa-json', required=True,
                    help='Output of val_qa_convertor.py (ground truth + sidecar metadata)')
    ap.add_argument('--result-file', default=None,
                    help='Optional upstream-style row dump (token    q    pred    gt)')
    ap.add_argument('--out', default=None,
                    help='Optional JSON dump of the per-bucket table')
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

    with open(args.val_qa_json) as f:
        val_entries = json.load(f)

    correct_by_q_type = defaultdict(list)
    rows = []
    missing = 0
    for entry in val_entries:
        qid = entry['id']
        if qid not in preds:
            missing += 1
            continue
        pred = preds[qid]
        gt = str(entry['gt_answer'])

        correct = 1 if pred == gt else 0
        correct_by_q_type['Overall'].append(correct)
        q_type = entry['template_type']
        sub = q_type + '_' + str(entry['num_hop'])
        correct_by_q_type[q_type].append(correct)
        correct_by_q_type[sub].append(correct)

        rows.append((entry['sample_token'], entry['question'], pred, gt))

    if missing:
        print(f"WARNING: {missing}/{len(val_entries)} predictions missing "
              f"- upstream scorer would assert; metrics below cover the "
              f"{len(rows)} present predictions.")

    q_dict = {}
    for q_type, vals in sorted(correct_by_q_type.items()):
        vals = np.asarray(vals)
        q_dict[q_type] = [int(vals.sum()), int(vals.shape[0])]
    for q_type, (val, tol) in q_dict.items():
        print(q_type, '%d / %d = %.2f' % (val, tol, 100.0 * val / tol))

    if args.result_file:
        with open(args.result_file, 'w') as f:
            for token, q, pred, gt in rows:
                f.write(f"{token}    {q}    {pred}    {gt}\n")
        print(f"Wrote result dump -> {args.result_file}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({
                'missing': missing,
                'total_gt': len(val_entries),
                'scored': len(rows),
                'buckets': {k: {'correct': v[0], 'total': v[1],
                                'pct': 100.0 * v[0] / v[1] if v[1] else 0.0}
                            for k, v in q_dict.items()},
            }, f, indent=2)
        print(f"Wrote metrics json -> {args.out}")


if __name__ == '__main__':
    main()
