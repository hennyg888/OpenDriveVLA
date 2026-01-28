import json
import re
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

STAGE1_PATH = "/home/s56cai/OpenDriveVLA/data/nuCaption/stage1_scene_data.json"
FINAL_PATH = "/home/s56cai/OpenDriveVLA/data/TOD3DCap/final_caption_bbox_token.json"
OUT_PATH = "/home/s56cai/OpenDriveVLA/data/TOD3DCap/stage1_agent_data.jsonl"

USER_PROMPT = "Please provide a caption and the BEV coordinate for the following object:<track start><OBJECT><track end>"


def read_json_or_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] in ("[", "{"):
        return json.loads(text)

    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def iter_final_items(final_data) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(final_data, dict):
        for k, v in final_data.items():
            yield str(k), v
        return

    if isinstance(final_data, list):
        for i, v in enumerate(final_data):
            if not isinstance(v, dict):
                continue
            sid = v.get("sample_id")
            if sid is None:
                sid = str(i)
            yield str(sid), v
        return

    raise TypeError(f"Unsupported final_data type: {type(final_data)}")


def get_bev_coordinate(sample: Dict[str, Any]) -> Tuple[float, float]:
    loc = sample.get("localization_caption", {})
    return float(loc["x_offset"]), float(loc["y_offset"])


def create_final_caption_with_bev(sample: Dict[str, Any]) -> str:
    attr = sample.get("attribute_caption", {}) or {}
    mot = sample.get("motion_caption", {}) or {}
    loc = sample.get("localization_caption", {}) or {}
    dep = sample.get("depth_caption", {}) or {}
    mp = sample.get("map_caption", {}) or {}

    appearance = attr.get("attribute_caption", "").strip()
    motion = mot.get("motion_caption", "").strip()
    localization = loc.get("localization_caption", "").strip()
    depth = dep.get("depth_caption", "").strip()
    map_location = mp.get("map_caption", "").strip()
    relation = (sample.get("relation_caption") or "").strip()

    if not appearance or not motion or not localization:
        raise KeyError("Missing required caption fields (appearance/motion/localization).")

    first = f"A {appearance} {map_location} is {motion}." if map_location else f"A {appearance} is {motion}."
    second = (
        f"It is {localization} and {relation}."
        if relation and relation != "none"
        else f"It is {localization} and {depth}."
    )

    bev_x, bev_y = get_bev_coordinate(sample)
    third = f"The BEV coordinate is ({bev_x:.2f},{bev_y:.2f})."

    return f"{first} {second} {third}".strip()


def collect_stage1_sample_tokens(stage1_data) -> set:
    tokens = set()

    def add_from_obj(obj):
        if isinstance(obj, dict):
            if "sample_token" in obj and isinstance(obj["sample_token"], str):
                tokens.add(obj["sample_token"])
            for v in obj.values():
                add_from_obj(v)
        elif isinstance(obj, list):
            for v in obj:
                add_from_obj(v)

    add_from_obj(stage1_data)
    return tokens


def main():
    print(f"Loading stage1: {STAGE1_PATH}")
    stage1 = read_json_or_jsonl(STAGE1_PATH)
    stage1_tokens = collect_stage1_sample_tokens(stage1)
    print(f"Collected stage1 sample_tokens: {len(stage1_tokens)}")

    print(f"Loading final: {FINAL_PATH}")
    final_data = read_json_or_jsonl(FINAL_PATH)

    out_p = Path(OUT_PATH)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_no_token = 0
    skipped_not_in_stage1 = 0
    skipped_bad_fields = 0

    with out_p.open("w", encoding="utf-8") as f:
        for sample_id, sample in iter_final_items(final_data):
            if not isinstance(sample, dict):
                continue

            st = sample.get("sample_token")
            if not isinstance(st, str):
                skipped_no_token += 1
                continue

            if st not in stage1_tokens:
                skipped_not_in_stage1 += 1
                continue

            try:
                caption = create_final_caption_with_bev(sample)
            except Exception:
                skipped_bad_fields += 1
                continue

            out_item = {
                "sample_id": sample_id,
                "conversation": [
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": caption},
                ],
            }
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            kept += 1

    print("Done.")
    print(f"Output: {OUT_PATH}")
    print(f"kept={kept}")
    print(f"skipped_no_token={skipped_no_token}")
    print(f"skipped_not_in_stage1={skipped_not_in_stage1}")
    print(f"skipped_bad_fields={skipped_bad_fields}")


if __name__ == "__main__":
    main()