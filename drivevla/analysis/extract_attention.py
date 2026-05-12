"""Attention extraction for OpenDriveVLA.

Two modes:

  - Teacher-forced (``extract_teacher_forced``):
      Run a single forward pass with prompt + GT response concatenated.
      Returns reduced attention tensor of shape (num_layers, num_output_pos,
      num_groups), where attention is averaged over heads and summed
      within each group span.

  - Autoregressive (``extract_autoregressive``):
      Call ``model.generate(..., output_attentions=True,
      return_dict_in_generate=True)``. Returns reduced attention of shape
      (num_layers, num_generated_steps, num_groups + 1), where the extra
      group is "gen_so_far" — attention paid to previously-generated
      tokens. Also returns the generated input_ids and decoded string.

In both modes we reduce per-layer attention to group-level immediately
and free the full attention tensor, so peak memory stays at one layer's
worth even at seq_len ≈ 2500.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from llava.constants import IGNORE_INDEX

from drivevla.analysis.group_spans import GROUP_ORDER, Span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spans_index(spans: List[Span]) -> Dict[str, Span]:
    return {s.name: s for s in spans}


def _reduce_layer_attention(
    layer_attn: torch.Tensor,
    output_start: int,
    output_end: int,
    spans: List[Span],
) -> torch.Tensor:
    """Reduce a single layer's full attention matrix to (num_output_pos, num_groups).

    layer_attn shape: (B=1, H, T, T). We average over heads, slice the rows
    corresponding to the output region, then sum within each group span.
    """
    # (H, T, T) -> (T, T) by mean over heads.
    attn = layer_attn[0].mean(dim=0).float()
    rows = attn[output_start:output_end]  # (O, T)
    out = torch.zeros((rows.shape[0], len(GROUP_ORDER)), dtype=torch.float32)
    span_by_name = _spans_index(spans)
    for gi, gname in enumerate(GROUP_ORDER):
        if gname not in span_by_name:
            continue
        s = span_by_name[gname]
        out[:, gi] = rows[:, s.start : s.end].sum(dim=1)
    return out


# ---------------------------------------------------------------------------
# Teacher-forced extraction
# ---------------------------------------------------------------------------


@dataclass
class TFAttentionResult:
    # (num_layers, num_output_pos, num_groups)
    attention: torch.Tensor
    # Output spans (start, end) within the post-expansion sequence.
    output_start: int
    output_end: int
    # Total post-expansion sequence length.
    seq_len: int


@torch.inference_mode()
def extract_teacher_forced(
    model,
    tokenizer,
    sample_data: dict,
    spans: List[Span],
    seq_len_predicted: int,
    output_text: str,
) -> TFAttentionResult:
    """Run teacher-forced forward, return per-layer per-output-pos group masses.

    sample_data: a dict from the dataset's __getitem__, must contain
        ``input_ids`` (pre-expansion, includes -201/-202/-203), ``uniad_data``,
        and optionally ``qa_instance_ind``. The caller is responsible for
        moving everything to the right device.
    spans: result of ``compute_spans(..., output_text=output_text)``. The
        last span must be the "output" span.
    seq_len_predicted: predicted total post-expansion length from
        ``compute_spans``. Used to assert the actual length matches.
    """
    # 1. Build full input_ids by concatenating GT response tokens.
    prompt_ids = sample_data["input_ids"]  # (1, T_pre)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    response_ids = tokenizer(
        output_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(prompt_ids.device)
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)

    # 2. Forward with output_attentions=True.
    outputs = model(
        input_ids=full_ids,
        uniad_data=sample_data.get("uniad_data"),
        uniad_pth=sample_data.get("uniad_pth"),
        qa_instance_ind=sample_data.get("qa_instance_ind"),
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )

    # 3. Validate sequence length.
    attentions = outputs.attentions  # tuple of (1, H, T, T)
    if not attentions:
        raise RuntimeError(
            "outputs.attentions is empty — eager attention path failed. "
            "Check attn_implementation='eager' on model load."
        )
    actual_seq_len = attentions[0].shape[-1]
    if actual_seq_len != seq_len_predicted:
        raise AssertionError(
            f"Span sum ({seq_len_predicted}) != actual seq_len ({actual_seq_len}). "
            f"Spans: {[(s.name, s.start, s.end) for s in spans]}"
        )

    # 4. Locate output rows.
    output_span = next(s for s in spans if s.name == "output")

    # 5. Per-layer reduce.
    num_layers = len(attentions)
    reduced = torch.zeros(
        (num_layers, output_span.length, len(GROUP_ORDER)),
        dtype=torch.float32,
    )
    for li, layer_attn in enumerate(attentions):
        reduced[li] = _reduce_layer_attention(
            layer_attn, output_span.start, output_span.end, spans
        )
        # Free immediately. attentions is a tuple; we can't replace items,
        # but Python should release once we're past this layer.
    # Hint to GC: drop reference to outputs.
    del outputs, attentions

    return TFAttentionResult(
        attention=reduced,
        output_start=output_span.start,
        output_end=output_span.end,
        seq_len=actual_seq_len,
    )


# ---------------------------------------------------------------------------
# Autoregressive extraction
# ---------------------------------------------------------------------------


@dataclass
class ARAttentionResult:
    # (num_layers, num_gen_steps, num_groups + 1)
    # Last column = "gen_so_far" (attention to previously generated tokens).
    attention: torch.Tensor
    # The decoded answer string.
    generated_text: str
    # Generated input_ids (1, num_gen_steps).
    generated_ids: torch.Tensor
    # Prompt length (post-expansion). Used to slice prompt vs gen.
    prompt_len: int


@torch.inference_mode()
def extract_autoregressive(
    model,
    tokenizer,
    sample_data: dict,
    spans: List[Span],
    seq_len_predicted: int,
    max_new_tokens: int = 256,
) -> ARAttentionResult:
    """Run greedy generate with output_attentions, reduce per-step per-layer.

    spans must NOT include an "output" span (this is the prompt-only span list).
    """
    prompt_ids = sample_data["input_ids"]
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    gen_out = model.generate(
        prompt_ids,
        uniad_data=sample_data.get("uniad_data"),
        uniad_pth=sample_data.get("uniad_pth"),
        qa_instance_ind=sample_data.get("qa_instance_ind"),
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        output_attentions=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    # gen_out.attentions: tuple of length K = num_new_tokens. Step 0 is the
    # prefill (query_len = prompt_len) where the row that generated the FIRST
    # new token is the LAST row. Steps 1..K-1 each have query_len = 1 with
    # key_len = prompt_len + step.
    step_attns = gen_out.attentions
    if not step_attns:
        raise RuntimeError("generate did not return attentions.")
    num_layers = len(step_attns[0])
    K = len(step_attns)

    # Sanity check: prefill key_len should equal seq_len_predicted.
    prefill_key_len = step_attns[0][0].shape[-1]
    if prefill_key_len != seq_len_predicted:
        raise AssertionError(
            f"Prefill key_len ({prefill_key_len}) != predicted seq_len "
            f"({seq_len_predicted}). Spans: {[(s.name, s.start, s.end) for s in spans]}"
        )

    span_by_name = _spans_index(spans)
    out = torch.zeros((num_layers, K, len(GROUP_ORDER)), dtype=torch.float32)

    for k, layers in enumerate(step_attns):
        # query_row: the position whose output sampled this step's token.
        query_row = -1 if k == 0 else 0
        for li in range(num_layers):
            attn = layers[li][0].mean(dim=0).float()  # (query_len, key_len)
            row = attn[query_row]  # (key_len,)
            for gi, gname in enumerate(GROUP_ORDER):
                if gname == "output":
                    # Repurpose the "output" column for "gen_so_far":
                    # attention to all positions strictly to the left of
                    # this step's query position, but only those past
                    # prompt_len (i.e., previously-generated tokens).
                    out[li, k, gi] = row[prefill_key_len:].sum()
                elif gname in span_by_name:
                    s = span_by_name[gname]
                    out[li, k, gi] = row[s.start : s.end].sum()

    # When inputs_embeds is passed (the llava generate path), HF returns
    # only the newly generated token ids in `sequences`.
    generated_ids = gen_out.sequences
    if generated_ids.shape[1] != K:
        # Fallback: slice off the prompt portion if HF returned prompt+gen.
        generated_ids = generated_ids[:, -K:]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return ARAttentionResult(
        attention=out,
        generated_text=generated_text,
        generated_ids=generated_ids,
        prompt_len=prefill_key_len,
    )
