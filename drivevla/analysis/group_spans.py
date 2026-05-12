"""Compute post-multimodal-expansion token group spans.

For a single sample, given:
  - the tokenizer
  - the user's `cached_nuscenes_data` entry (carries cmd, ego/his strings)
  - the actual (scene_len, track_len, map_len) produced by the model's vision tower
  - optional GT output text (for teacher-forced mode)
returns a span dict {group_name: (start, end)} aligned to the LLM's
post-expansion `inputs_embeds` sequence.

Strategy:
  - Reconstruct the prompt as an ordered list of fragments where each
    fragment is either pure text or one of the {<SCENE>, <TRACK>, <MAP>}
    placeholder slots.
  - Tokenize each text fragment in isolation with add_special_tokens=False.
    The bracket tokens (<scene_start>, <scene_end>, <traj_start>, ...) and
    chat-template tokens (<|im_start|>, <|im_end|>) are all single added
    tokens in the tokenizer vocab, so per-fragment tokenization is exact:
    splitting at these single-token boundaries does not change BPE
    merging.
  - Each placeholder slot contributes scene_len / track_len / map_len
    tokens after expansion.
  - Concatenate the per-fragment lengths to derive spans.

The caller is expected to compare the total predicted length to the actual
post-expansion sequence length and fail loudly on mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from drivevla.data_utils.build_llava_conversation import generate_user_message
from llava.conversation import conv_templates


# ---------------------------------------------------------------------------
# Prompt reconstruction
# ---------------------------------------------------------------------------

# Mirror of build_llava_conversation: every line that ends in "\n" must keep
# its newline so that fragment-by-fragment tokenization aggregates exactly.
_USER_OPEN = "<|im_start|>user\n"
_IM_END = "<|im_end|>\n"
_ASST_OPEN = "<|im_start|>assistant\n"


def _system_block() -> str:
    """Return the system block (`<|im_start|>system\\n...<|im_end|>\\n`).

    Pulled from the qwen_planning_oriented_vlm conv template so it stays
    in lock-step with what the dataset produces.
    """
    template = conv_templates["qwen_planning_oriented_vlm"]
    # template.system already starts with "<|im_start|>system\n..."
    return template.system + template.sep + "\n"


def _build_user_segments(
    cached_entry: dict,
    include_ego_history: bool = True,
) -> List[Tuple[str, str]]:
    """Return an ordered list of (name, text) for the user message body.

    Names "__SCENE__", "__TRACK__", "__MAP__" are placeholder slots that
    do not contribute text — they will be replaced with the corresponding
    feature blocks during multimodal expansion.
    """
    ego_msg, his_msg, cmd_msg, _ = generate_user_message(cached_entry)

    ego_line = f"Ego states: {ego_msg}\n" if include_ego_history else ""
    his_line = (
        f"Historical trajectory (last 2 seconds): {his_msg}\n"
        if include_ego_history
        else ""
    )

    return [
        ("user_pre_scene", "Scene information: <scene_start>"),
        ("__SCENE__", ""),
        ("between_scene_track", "<scene_end>\nObject-wise tracking information: <track_start>"),
        ("__TRACK__", ""),
        ("between_track_map", "<track_end>\nMap information: <map_start>"),
        ("__MAP__", ""),
        ("after_map", "<map_end>\n"),
        ("ego", ego_line),
        ("his", his_line),
        ("cmd", f"Mission goal: {cmd_msg}\n"),
        ("instruct", "Planning trajectory: <trajectory>"),
    ]


# ---------------------------------------------------------------------------
# Span computation
# ---------------------------------------------------------------------------


@dataclass
class Span:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


GROUP_ORDER: Tuple[str, ...] = (
    "system",
    "user_pre_scene",
    "scene",
    "between_scene_track",
    "track",
    "between_track_map",
    "map",
    "after_map",
    "ego",
    "his",
    "cmd",
    "instruct",
    "asst_pre",
    "output",
)


def compute_spans(
    tokenizer,
    cached_entry: dict,
    scene_len: int,
    track_len: int,
    map_len: int,
    output_text: Optional[str] = None,
    include_ego_history: bool = True,
) -> Tuple[List[Span], int]:
    """Return ordered list of Spans + the predicted total length.

    Caller should verify ``predicted_total == actual_post_expansion_seq_len``
    and treat any mismatch as a fatal error.
    """

    # 1. Build the ordered fragment list.
    user_segments = _build_user_segments(
        cached_entry, include_ego_history=include_ego_history
    )

    # System block.
    fragments: List[Tuple[str, Optional[str], Optional[str]]] = []
    # Each entry: (group_name, kind, text). kind in {"text","scene","track","map"}.
    fragments.append(("system", "text", _system_block() + _USER_OPEN))

    for name, text in user_segments:
        if name == "__SCENE__":
            fragments.append(("scene", "scene", None))
        elif name == "__TRACK__":
            fragments.append(("track", "track", None))
        elif name == "__MAP__":
            fragments.append(("map", "map", None))
        else:
            # Drop empty text fragments to keep spans interpretable.
            if text:
                fragments.append((name, "text", text))

    fragments.append(("asst_pre", "text", _IM_END + _ASST_OPEN))

    if output_text is not None:
        # Teacher-forced mode: append the GT response.
        fragments.append(("output", "text", output_text))

    # 2. Compute per-fragment token lengths.
    spans: List[Span] = []
    cursor = 0
    for name, kind, text in fragments:
        if kind == "text":
            ids = tokenizer(text, add_special_tokens=False).input_ids
            length = len(ids)
        elif kind == "scene":
            length = scene_len
        elif kind == "track":
            length = track_len
        elif kind == "map":
            length = map_len
        else:
            raise ValueError(f"Unknown kind: {kind}")
        if length == 0:
            # Skip zero-length groups (e.g., when ego/his lines are disabled
            # via include_ego_history=False).
            continue
        spans.append(Span(name=name, start=cursor, end=cursor + length))
        cursor += length

    return spans, cursor


def spans_to_dict(spans: List[Span]) -> Dict[str, Tuple[int, int]]:
    return {s.name: (s.start, s.end) for s in spans}
