import argparse
import json
import os
import pickle
import sys

import numpy as np

from utils.trajectory_utils import retrieve_traj, check_traj
from eval_share.evaluation import planning_evaluation

CACHED_NUSCENES_PKL = "data/nuscenes/cached_nuscenes_info.pkl"


def load_valid_tokens(pkl_path=CACHED_NUSCENES_PKL):
    """Return the set of sample tokens whose 6-step ego future is fully valid."""
    with open(pkl_path, "rb") as f:
        cached = pickle.load(f)
    return {t for t, d in cached.items() if np.all(d["gt_ego_fut_masks"] == 1)}

def evaluate_planning_oriented_vlm(output_path, include_end_of_scene=False):
    '''
    Convert the planning results from conversations to pred_trajs_dict and pred_trajs_multi_modal_dict
    '''
    out_dir = os.path.dirname(output_path)
    
    pred_trajs_dict = {}
    pred_trajs_multi_modal_dict = {}

    skipped = 0
    with open(output_path, 'r') as f:
        for line in f:
            conv_result = json.loads(line.strip())

            # retrieve multi-modal planning trajectories
            traj_multi_modal = []
            valid = True
            for answer in conv_result['answer']:
                traj = retrieve_traj(answer)
                if traj is None:
                    valid = False
                    break
                try:
                    check_traj(traj)
                except AssertionError:
                    valid = False
                    break
                traj_multi_modal.append(traj)

            if not valid or len(traj_multi_modal) == 0:
                skipped += 1
                continue

            # TODO: pick the best planning trajectory
            pred_trajs_dict[conv_result['id']] = [traj_multi_modal[0]]
            pred_trajs_multi_modal_dict[conv_result['id']] = traj_multi_modal

    if skipped > 0:
        print(f"[WARNING] Skipped {skipped} samples with no valid trajectory in model output.")

    # Save pred_trajs_dict to a json file
    with open(os.path.join(out_dir, "pred_trajs_dict.json"), 'w') as f:
        json.dump(pred_trajs_dict, f, indent=2)
    print(f"Saving pred_trajs_dict to {os.path.join(out_dir, 'pred_trajs_dict.json')}")

    # Save planning_trajectory_multi_modal to a json file
    with open(os.path.join(out_dir, "pred_trajs_multi_modal_dict.json"), 'w') as f:
        json.dump(pred_trajs_multi_modal_dict, f, indent=2)
    print(f"Saving pred_trajs_multi_modal to {os.path.join(out_dir, 'pred_trajs_multi_modal_dict.json')}")

    # Setup logging to save the evaluation results to a file
    log_path = os.path.join(out_dir, "eval_results.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"Saving evaluation results to {log_path}")
    sys.stdout = open(log_path, 'w')

    """
    # TODO: check only_vehicle=False or True
    This evaluation function will report both the STP-3 metric (avg over avg) and the UniAD metric. 
    Since UniAD only considers the vehicle category when generating ground truth occupancy, while ST-P3 considers both the vehicle and pedestrian categories.
    If you want to report the STP-3 metric, please set only_vehicle=False.
    if you want to report the UniAD metric, please set only_vehicle=True.
    """
    if include_end_of_scene:
        print("[eval] --include-end-of-scene: evaluating on ALL val tokens "
              "(no fut_mask filter)")
        planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=True)
    else:
        valid_tokens = load_valid_tokens()
        print(f"[eval] restricting metrics to {len(valid_tokens)} tokens with fully valid 6-step futures")
        planning_evaluation(pred_trajs_dict, subset=valid_tokens, only_vehicle=True)
    
    # Restore stdout and print log contents
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    # Read and print log contents
    with open(log_path, 'r') as f:
        log_content = f.read()
        print(log_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--include-end-of-scene",
        action="store_true",
        help="Include val samples whose 6-step ego future walks past "
             "end of scene (default: filter them out).",
    )
    args = parser.parse_args()

    evaluate_planning_oriented_vlm(
        args.output, include_end_of_scene=args.include_end_of_scene
    )
    print(f">>> Evaluation results saved to {os.path.dirname(args.output)}")

if __name__ == "__main__":
    main()