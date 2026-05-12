"""Token-position attention heatmap + embedding distribution analysis.

CLI entry point. Loads an OpenDriveVLA checkpoint with eager attention,
runs both teacher-forced and autoregressive forwards on a small inspected
sample set, plus a teacher-forced aggregate over many samples, and saves
heatmaps + scatter plots to ``--output-dir``.

Vision tokens are produced ONLINE by the FusionAD vision tower
(``use_uniad_pth=False``); cached .pth features are NOT used so the
analysis reflects the current checkpoint end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from mmengine import Config
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.train.train import DataArguments

from drivevla.data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from drivevla.data_utils.nuscenes_llava_datacollector import (
    DataCollatorForLLaVANuScenesDataset,
)
from drivevla.utils.tensor_utils import move_data_to_device

from drivevla.analysis import plot_utils
from drivevla.analysis.extract_attention import (
    extract_autoregressive,
    extract_teacher_forced,
)
from drivevla.analysis.extract_embeddings import (
    FeatureCapture,
    sample_vocab_embeds,
    slice_embeds_by_groups,
)
from drivevla.analysis.group_spans import GROUP_ORDER, compute_spans, spans_to_dict


# ---------------------------------------------------------------------------
# Sample picking
# ---------------------------------------------------------------------------


def _cast_floats(obj, dtype):
    """Recursively cast float32 tensors in a nested dict/list to ``dtype``."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.dtype != dtype:
            return obj.to(dtype)
        return obj
    if isinstance(obj, dict):
        return {k: _cast_floats(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_floats(v, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_floats(v, dtype) for v in obj)
    return obj


def _command_of(cached_entry: dict) -> str:
    cmd = cached_entry["gt_ego_fut_cmd"]
    right, left, forward = cmd
    if right > 0:
        return "turn right"
    if left > 0:
        return "turn left"
    return "keep forward"


def auto_pick_inspect_tokens(
    dataset: LLaVANuScenesDataset, n_per_command: int = 3
) -> List[str]:
    """Pick a stratified set: n_per_command for each of the three commands.

    Picks only from samples present in ``dataset.data_infos`` (i.e., the
    actual split this dataset was instantiated on) AND with fully valid
    6-step ego futures.
    """
    in_split = {info["token"] for info in dataset.data_infos}
    by_cmd: Dict[str, List[str]] = {"turn left": [], "turn right": [], "keep forward": []}
    for token in in_split:
        entry = dataset.cached_nuscenes_data.get(token)
        if entry is None:
            continue
        if not np.all(entry["gt_ego_fut_masks"] == 1):
            continue
        cmd = _command_of(entry)
        by_cmd[cmd].append(token)
    rng = random.Random(0)
    picked: List[str] = []
    for cmd, tokens in by_cmd.items():
        rng.shuffle(tokens)
        picked.extend(tokens[:n_per_command])
    return picked


def find_dataset_idx(dataset: LLaVANuScenesDataset, sample_token: str) -> Optional[int]:
    for i, info in enumerate(dataset.data_infos):
        if info["token"] == sample_token:
            return i
    return None


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------


def analyze_one_sample(
    model,
    tokenizer,
    dataset: LLaVANuScenesDataset,
    collator: DataCollatorForLLaVANuScenesDataset,
    idx: int,
    device: torch.device,
    modes: List[str],
    out_dir: Path,
    save_embeds: bool,
) -> Dict[str, dict]:
    """Run TF and/or AR modes on one sample. Returns a dict with reduced
    attention matrices, generated text, and (optionally) sliced embeddings.
    """
    raw = dataset[idx]
    # `id` in test mode is "<token>_trajectory" (process_traj_data sets qa_id).
    # The dataset reduces back to the bare token via split('_')[0].
    raw_id = raw["id"]
    sample_token = raw_id.split("_")[0] if isinstance(raw_id, str) else raw_id
    # The collator runs mmcv.parallel.collate + remove_datacontainer on
    # uniad_data so the model's vision tower can consume it.
    sample = collator([raw])
    sample = move_data_to_device(sample, device)
    # Cast all fp32 tensors in the sample to the model's compute dtype so
    # that conv/BN weights and inputs match.
    target_dtype = next(model.parameters()).dtype
    sample = _cast_floats(sample, target_dtype)
    cached_entry = dataset.cached_nuscenes_data[sample_token]

    result: Dict[str, dict] = {"token": sample_token, "command": _command_of(cached_entry)}

    # No autocast: the model has been uniformly cast and inputs match.
    no_grad_ctx = torch.inference_mode()
    tf_vision_tower_result = None

    # ----- Teacher-forced (with feature capture) -----
    if "tf" in modes:
        # The GT response string the dataset would have produced.
        from llava.constants import DEFAULT_TRAJ_START_TOKEN, DEFAULT_TRAJ_END_TOKEN
        from drivevla.data_utils.build_llava_conversation import generate_user_message
        _, _, _, traj_message = generate_user_message(cached_entry)
        gt_response = f"{DEFAULT_TRAJ_START_TOKEN}{traj_message}{DEFAULT_TRAJ_END_TOKEN}"

        with FeatureCapture(model) as cap, no_grad_ctx:
            from drivevla.analysis.extract_attention import _reduce_layer_attention
            from drivevla.analysis.group_spans import GROUP_ORDER as _GO

            # Build full input ids (prompt + GT).
            prompt_ids = sample["input_ids"]
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            response_ids = tokenizer(
                gt_response, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(prompt_ids.device)
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)

            outputs = model(
                input_ids=full_ids,
                uniad_data=sample.get("uniad_data"),
                uniad_pth=sample.get("uniad_pth"),
                qa_instance_ind=sample.get("qa_instance_ind"),
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            attentions = outputs.attentions
            actual_seq_len = attentions[0].shape[-1]

            # Now we have scene/track/map lengths from the capture.
            spans, predicted_total = compute_spans(
                tokenizer,
                cached_entry,
                scene_len=cap.features.scene_len,
                track_len=cap.features.track_len,
                map_len=cap.features.map_len,
                output_text=gt_response,
                include_ego_history=dataset.include_ego_history,
            )
            if predicted_total != actual_seq_len:
                raise AssertionError(
                    f"[{sample_token}] span sum {predicted_total} != actual seq_len "
                    f"{actual_seq_len}. Spans: {spans_to_dict(spans)}"
                )

            output_span = next(s for s in spans if s.name == "output")
            num_layers = len(attentions)
            tf_reduced = torch.zeros(
                (num_layers, output_span.length, len(_GO)), dtype=torch.float32
            )
            for li, layer_attn in enumerate(attentions):
                tf_reduced[li] = _reduce_layer_attention(
                    layer_attn, output_span.start, output_span.end, spans
                )
            del outputs, attentions

            result["tf"] = {
                "attention": tf_reduced.detach().cpu().numpy(),
                "spans": spans_to_dict(spans),
                "seq_len": actual_seq_len,
                "gt_response": gt_response,
                "scene_len": cap.features.scene_len,
                "track_len": cap.features.track_len,
                "map_len": cap.features.map_len,
            }
            # Save the raw vision_tower output so AR can reuse it.
            tf_vision_tower_result = cap.features.vision_tower_result

            # Slice inputs_embeds for the embedding distribution analysis.
            if save_embeds and cap.features.inputs_embeds is not None:
                emb_groups = slice_embeds_by_groups(cap.features.inputs_embeds, spans)
                result["embeds"] = emb_groups

    # ----- Autoregressive -----
    if "ar" in modes:
        # AR uses prompt-only spans. The "output" column in AR attention
        # is repurposed to mean "gen_so_far".
        #
        # Reuse the vision_tower_result captured during TF (same input,
        # same features) to skip FusionAD's stateful prev_frame_info path
        # for the second forward. Falls back to a fresh capture if TF
        # mode wasn't run.
        if "tf" in modes and tf_vision_tower_result is not None:
            scene_len = result["tf"]["scene_len"]
            track_len = result["tf"]["track_len"]
            map_len = result["tf"]["map_len"]
            ar_uniad_pth = tf_vision_tower_result
            ar_uniad_data = None
        else:
            with FeatureCapture(model) as cap, no_grad_ctx:
                _ = model.prepare_inputs_labels_for_multimodal_uniad_vlm(
                    sample["input_ids"].unsqueeze(0)
                    if sample["input_ids"].dim() == 1
                    else sample["input_ids"],
                    None, None, None, None, None,
                    uniad_data=sample.get("uniad_data"),
                    uniad_pth=sample.get("uniad_pth"),
                    qa_instance_ind=sample.get("qa_instance_ind"),
                )
                scene_len = cap.features.scene_len
                track_len = cap.features.track_len
                map_len = cap.features.map_len
                ar_uniad_pth = cap.features.vision_tower_result
                ar_uniad_data = None

        spans_prompt, total_prompt = compute_spans(
            tokenizer,
            cached_entry,
            scene_len=scene_len,
            track_len=track_len,
            map_len=map_len,
            output_text=None,
            include_ego_history=dataset.include_ego_history,
        )
        # Build a sample dict that bypasses uniad_data and supplies the
        # cached vision_tower_result via uniad_pth.
        ar_sample = dict(sample)
        ar_sample["uniad_data"] = ar_uniad_data
        ar_sample["uniad_pth"] = ar_uniad_pth
        ar_result = extract_autoregressive(
            model, tokenizer, ar_sample, spans_prompt, total_prompt, max_new_tokens=128
        )
        result["ar"] = {
            "attention": ar_result.attention.detach().cpu().numpy(),
            "spans": spans_to_dict(spans_prompt),
            "prompt_len": ar_result.prompt_len,
            "generated_text": ar_result.generated_text,
        }

    return result


# ---------------------------------------------------------------------------
# Plotting orchestration
# ---------------------------------------------------------------------------


def plot_inspect_sample(sample_result: dict, out_dir: Path):
    token = sample_result["token"]
    cmd = sample_result["command"]
    sample_dir = out_dir / "inspect" / token
    sample_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = f"{token[:8]} | {cmd}"

    if "tf" in sample_result:
        tf_attn = sample_result["tf"]["attention"]  # (L, O, G)
        plot_utils.plot_attention_heatmap_per_layer(
            tf_attn, list(GROUP_ORDER), str(sample_dir / "tf_per_layer.png"),
            title=f"{title_prefix} | teacher-forced",
        )
        plot_utils.plot_attention_aggregate_heatmap(
            tf_attn.mean(axis=1), list(GROUP_ORDER),
            str(sample_dir / "tf_layer_x_group.png"),
            title=f"{title_prefix} | TF | mean over output positions",
        )

    if "ar" in sample_result:
        ar_attn = sample_result["ar"]["attention"]  # (L, K, G)
        plot_utils.plot_attention_heatmap_per_layer(
            ar_attn, list(GROUP_ORDER), str(sample_dir / "ar_per_layer.png"),
            title=f"{title_prefix} | autoregressive (col 'output' = gen_so_far)",
        )
        plot_utils.plot_attention_aggregate_heatmap(
            ar_attn.mean(axis=1), list(GROUP_ORDER),
            str(sample_dir / "ar_layer_x_group.png"),
            title=f"{title_prefix} | AR | mean over generation steps",
        )

    if "tf" in sample_result and "ar" in sample_result:
        plot_utils.plot_tf_vs_ar_bar(
            sample_result["tf"]["attention"],
            sample_result["ar"]["attention"],
            list(GROUP_ORDER[:-1]),  # drop "output" column
            str(sample_dir / "tf_vs_ar_bar.png"),
            title=f"{title_prefix} | TF vs AR (mean over layers, output positions)",
        )

    # Save raw npz for replotting / re-analysis.
    np_payload = {}
    if "tf" in sample_result:
        np_payload["tf_attn"] = sample_result["tf"]["attention"]
    if "ar" in sample_result:
        np_payload["ar_attn"] = sample_result["ar"]["attention"]
    if np_payload:
        np.savez_compressed(sample_dir / "attention.npz", **np_payload)

    # Save metadata.
    meta = {
        "token": token,
        "command": cmd,
        "tf": {k: v for k, v in sample_result.get("tf", {}).items() if k != "attention" and k != "spans"} | {
            "spans": sample_result.get("tf", {}).get("spans"),
        } if "tf" in sample_result else None,
        "ar": {
            "prompt_len": sample_result.get("ar", {}).get("prompt_len"),
            "generated_text": sample_result.get("ar", {}).get("generated_text"),
            "spans": sample_result.get("ar", {}).get("spans"),
        } if "ar" in sample_result else None,
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def plot_embedding_for_sample(
    sample_result: dict,
    vocab_emb: np.ndarray,
    out_dir: Path,
):
    token = sample_result["token"]
    cmd = sample_result["command"]
    if "embeds" not in sample_result:
        return
    sample_dir = out_dir / "inspect" / token
    sample_dir.mkdir(parents=True, exist_ok=True)
    groups = sample_result["embeds"]
    # Add the random vocab as another group for reference.
    groups = dict(groups)
    groups["text_vocab_random"] = vocab_emb

    # Stack and project with PCA.
    names = []
    arrays = []
    for name, arr in groups.items():
        names.append(name)
        arrays.append(arr)
    sizes = [a.shape[0] for a in arrays]
    X = np.concatenate(arrays, axis=0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0)
    Y = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    # Slice back per group.
    out_pca = {}
    cur = 0
    for name, n in zip(names, sizes):
        out_pca[name] = Y[cur : cur + n]
        cur += n
    plot_utils.plot_embedding_scatter(
        out_pca,
        str(sample_dir / "embed_pca.png"),
        title=f"{token[:8]} | {cmd} | PCA",
        method_name="PC",
        explained_var=(float(var[0]), float(var[1])),
    )

    # t-SNE for second view.
    from sklearn.manifold import TSNE
    perplexity = max(5, min(30, X.shape[0] // 5))
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity, init="pca")
    Y2 = tsne.fit_transform(X)
    out_tsne = {}
    cur = 0
    for name, n in zip(names, sizes):
        out_tsne[name] = Y2[cur : cur + n]
        cur += n
    plot_utils.plot_embedding_scatter(
        out_tsne,
        str(sample_dir / "embed_tsne.png"),
        title=f"{token[:8]} | {cmd} | t-SNE (perplexity={perplexity})",
        method_name="t-SNE",
    )


def plot_aggregate(
    aggregate_lg: np.ndarray,           # (N, L, G)
    commands: List[str],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_lg = aggregate_lg.mean(axis=0)
    plot_utils.plot_attention_aggregate_heatmap(
        mean_lg, list(GROUP_ORDER),
        str(out_dir / "aggregate_layer_x_group.png"),
        title=f"Aggregate over {len(aggregate_lg)} samples | mean attention mass",
    )
    plot_utils.plot_layer_curves(
        mean_lg, list(GROUP_ORDER),
        str(out_dir / "aggregate_layer_curves.png"),
        title=f"Aggregate over {len(aggregate_lg)} samples",
    )

    # Per-command split.
    for cmd in ("turn left", "turn right", "keep forward"):
        idx = [i for i, c in enumerate(commands) if c == cmd]
        if not idx:
            continue
        sub = aggregate_lg[idx].mean(axis=0)
        plot_utils.plot_attention_aggregate_heatmap(
            sub, list(GROUP_ORDER),
            str(out_dir / f"aggregate_{cmd.replace(' ', '_')}.png"),
            title=f"{cmd} | n={len(idx)}",
        )

    np.savez_compressed(
        out_dir / "aggregate.npz",
        attn=aggregate_lg, commands=np.array(commands, dtype=object),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument(
        "--data", default=None, type=str,
        help="Path to a conversation JSON. If omitted, online generation "
             "from cached_nuscenes_info.pkl is used (val split).",
    )
    parser.add_argument(
        "--nuscenes-cfg",
        default="projects/configs/fusionad/fusion_base_track_map.py", type=str,
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--inspect-tokens", default="auto", type=str,
        help="'auto' (stratified by command) or comma-separated sample tokens.",
    )
    parser.add_argument(
        "--n-per-command", default=3, type=int,
        help="When --inspect-tokens=auto, number of samples per command.",
    )
    parser.add_argument("--aggregate-n", default=200, type=int)
    parser.add_argument(
        "--modes", default="tf,ar", type=str,
        help="Comma-separated subset of {tf, ar}.",
    )
    parser.add_argument(
        "--layers", default="all", type=str,
        help="'all' or comma-separated layer indices to render in per-layer plots.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--vocab-sample-n", default=1000, type=int,
        help="Number of random LLM-vocab embeddings to plot as reference cluster.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> output_dir: {out_dir}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    assert all(m in {"tf", "ar"} for m in modes)

    # ------- Load model with eager attention -------
    disable_torch_init()
    overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
    print(f">>> loading model: {args.model_path}")
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name="llava_qwen",
        device_map=args.device,
        attn_implementation="eager",
        multimodal=True,
        overwrite_config=overwrite_config,
    )
    model.eval()
    # Cast the entire model uniformly so that BN/conv weights match.
    # Mirrors DeepSpeed's inference behavior in inference_drivevla.py;
    # we then cast the float32 entries inside the dataset sample to the
    # same dtype before forward (no torch.cuda.amp.autocast — that would
    # leave BN weights in their original dtype and break the conv path).
    target_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = model.to(target_dtype)

    # ------- Build dataset (online vision, no uniad_pth) -------
    uniad_cfg: Config = Config.fromfile(args.nuscenes_cfg)
    data_args = DataArguments(
        data_path=args.data,
        lazy_preprocess=True,
        frames_upbound=32,
    )
    dataset = LLaVANuScenesDataset(
        tokenizer, data_args, uniad_cfg.data.test,
        llava_test_mode=True,
        use_uniad_pth=False,
        skip_build_conversation=False,
    )
    collator = DataCollatorForLLaVANuScenesDataset(
        tokenizer=tokenizer, llava_test_mode=True,
    )
    print(f">>> dataset size: {len(dataset)}")

    device = torch.device(args.device)

    # ------- Resolve inspect tokens -------
    if args.inspect_tokens == "auto":
        inspect_tokens = auto_pick_inspect_tokens(dataset, n_per_command=args.n_per_command)
    else:
        inspect_tokens = [t.strip() for t in args.inspect_tokens.split(",") if t.strip()]
    print(f">>> inspect tokens ({len(inspect_tokens)}): {inspect_tokens}")

    # ------- Vocab embedding sample (shared across inspect samples) -------
    vocab_emb = sample_vocab_embeds(model, n=args.vocab_sample_n)

    # ------- Per-sample inspect runs -------
    inspect_results = []
    for token in tqdm(inspect_tokens, desc="inspect"):
        idx = find_dataset_idx(dataset, token)
        if idx is None:
            print(f"[WARN] token {token} not in dataset, skipping.")
            continue
        try:
            res = analyze_one_sample(
                model, tokenizer, dataset, collator, idx, device, modes,
                out_dir, save_embeds=("tf" in modes),
            )
        except AssertionError as e:
            print(f"[ERROR] sample {token} failed span check: {e}")
            continue
        plot_inspect_sample(res, out_dir)
        if "embeds" in res:
            plot_embedding_for_sample(res, vocab_emb, out_dir)
        inspect_results.append(res)

    # ------- Aggregate run (TF only, reduced to (L, G) per sample) -------
    if args.aggregate_n > 0 and "tf" in modes:
        # Pick aggregate sample indices: random sample of dataset entries
        # whose 6-step ego future is fully valid. Avoid overlap with inspect.
        rng = random.Random(1)
        valid_indices = []
        for i, info in enumerate(dataset.data_infos):
            entry = dataset.cached_nuscenes_data.get(info["token"])
            if entry is None:
                continue
            if not np.all(entry["gt_ego_fut_masks"] == 1):
                continue
            if info["token"] in inspect_tokens:
                continue
            valid_indices.append(i)
        rng.shuffle(valid_indices)
        valid_indices = valid_indices[: args.aggregate_n]

        agg_lg = []
        agg_cmd = []
        for idx in tqdm(valid_indices, desc="aggregate"):
            try:
                res = analyze_one_sample(
                    model, tokenizer, dataset, collator, idx, device, ["tf"],
                    out_dir, save_embeds=False,
                    )
            except (AssertionError, RuntimeError) as e:
                print(f"[WARN] aggregate idx {idx} failed: {e}")
                continue
            attn_lg = res["tf"]["attention"].mean(axis=1)  # (L, G)
            agg_lg.append(attn_lg)
            agg_cmd.append(res["command"])

        if agg_lg:
            agg_lg = np.stack(agg_lg, axis=0)
            plot_aggregate(agg_lg, agg_cmd, out_dir / "aggregate")

    # ------- Save run summary -------
    summary = {
        "model_path": args.model_path,
        "data": args.data,
        "inspect_tokens": inspect_tokens,
        "aggregate_n": args.aggregate_n,
        "modes": modes,
        "group_order": list(GROUP_ORDER),
        "n_inspect_completed": len(inspect_results),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f">>> done. summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
