"""Capture post-projection (pre-LLM) inputs_embeds and the per-modality
feature lengths produced by FusionAD.

Used in two ways:

  - As a context manager on the model — wrap the TF forward call so that
    the captures land in the context object without a second forward.

  - As a standalone helper that calls
    ``prepare_inputs_labels_for_multimodal_uniad_vlm`` directly when only
    the embedding map is needed (cheaper, no LLM forward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class CapturedFeatures:
    scene: Optional[torch.Tensor] = None  # (S, hidden) -- post-projection
    track: Optional[torch.Tensor] = None  # (Tk, hidden)
    map: Optional[torch.Tensor] = None    # (M, hidden)
    inputs_embeds: Optional[torch.Tensor] = None  # (1, T, hidden)
    # Raw vision_tower output dict (pre-projection). Reusable as
    # `uniad_pth=` argument to a subsequent forward to skip re-running
    # FusionAD on the same sample (avoids its stateful prev_frame_info
    # cache, which has its own dtype issues across calls).
    vision_tower_result: Optional[dict] = None

    @property
    def scene_len(self) -> int:
        return 0 if self.scene is None else self.scene.shape[0]

    @property
    def track_len(self) -> int:
        return 0 if self.track is None else self.track.shape[0]

    @property
    def map_len(self) -> int:
        return 0 if self.map is None else self.map.shape[0]


class FeatureCapture:
    """Context manager that monkey-patches the model to capture features.

    Usage:
        with FeatureCapture(model) as cap:
            outputs = model(...)
        scene_len = cap.features.scene_len
        embeds = cap.features.inputs_embeds  # (1, T, hidden)
    """

    def __init__(self, model):
        self._model = model
        self.features = CapturedFeatures()
        self._orig_encode = None
        self._orig_prepare = None
        self._orig_vt_forward = None

    def __enter__(self):
        cap = self.features
        self._orig_encode = self._model.encode_vision_tower_result

        def wrapped_encode(vision_tower_result):
            cap.vision_tower_result = vision_tower_result
            scene, track, map_ = self._orig_encode(vision_tower_result)
            cap.scene = scene.detach()
            cap.track = None if track is None else track.detach()
            cap.map = map_.detach()
            return scene, track, map_

        # Assign to the BOUND method on the instance.
        self._model.encode_vision_tower_result = wrapped_encode

        self._orig_prepare = self._model.prepare_inputs_labels_for_multimodal_uniad_vlm

        def wrapped_prepare(*args, **kwargs):
            ret = self._orig_prepare(*args, **kwargs)
            # ret = (input_ids, position_ids, attention_mask, past_key_values,
            #        inputs_embeds, labels)
            inputs_embeds = ret[4]
            if inputs_embeds is not None:
                cap.inputs_embeds = inputs_embeds.detach()
            return ret

        self._model.prepare_inputs_labels_for_multimodal_uniad_vlm = wrapped_prepare
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original methods.
        if self._orig_encode is not None:
            self._model.encode_vision_tower_result = self._orig_encode
        if self._orig_prepare is not None:
            self._model.prepare_inputs_labels_for_multimodal_uniad_vlm = self._orig_prepare


# ---------------------------------------------------------------------------
# Slicing utilities for the distribution analysis
# ---------------------------------------------------------------------------


def slice_embeds_by_groups(
    inputs_embeds: torch.Tensor,
    spans,
) -> dict:
    """Return {group_name: numpy array of shape (n, hidden)} for each span."""
    if inputs_embeds.dim() == 3:
        inputs_embeds = inputs_embeds[0]
    arr = inputs_embeds.float().cpu().numpy()
    out = {}
    for s in spans:
        if s.length == 0:
            continue
        out[s.name] = arr[s.start : s.end]
    return out


def sample_vocab_embeds(model, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Random sample n embeddings from the LLM's input embedding table.

    Returns float32 numpy array of shape (n, hidden).
    """
    rng = np.random.default_rng(seed)
    emb = model.get_input_embeddings().weight.detach()
    V = emb.shape[0]
    n = min(n, V)
    idx = rng.choice(V, size=n, replace=False)
    out = emb[torch.as_tensor(idx, dtype=torch.long, device=emb.device)]
    return out.float().cpu().numpy()
