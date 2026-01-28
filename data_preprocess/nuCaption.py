import json
import re
import argparse
from collections import defaultdict

VIEWS_REQUIRED = {
    "front": "front",
    "front left": "front left",
    "front right": "front right",
    "back": "back",
    "back left": "back left",
    "back right": "back right",
}

VIEW_ORDER = ["front left", "front", "front right", "back right", "back", "back left"]

ALLOWED_SUFFIXES = {
    "Please describe the current scene.": 0,
    "Provide an overview of the surrounding landscape or environment.": 1,
}

QUESTION_RE = re.compile(
    r"^\s*This is the car's\s+(front|front left|front right|back|back left|back right)\s+view\.\s*(.+?)\s*$",
    re.IGNORECASE,
)

def load_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def extract_scene_shows(answer: str):
    if not answer:
        return None
    idx = answer.find("The scene shows")
    if idx == -1:
        return None
    return answer[idx:].strip()

def normalize_view(v: str) -> str:
    return re.sub(r"\s+", " ", v.strip().lower())

def build_group_caption(view_to_caption: dict) -> str:
    lines = []
    for view in VIEW_ORDER:
        cap = view_to_caption.get(view)
        if not cap:
            continue
        lines.append(f"This is the scene caption of the {view} view: {cap}")
    return "\n".join(lines) + ("\n" if lines else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/s56cai/OpenDriveVLA/data/nuCaption/train.json")
    ap.add_argument("--output", default="/home/s56cai/OpenDriveVLA/data/nuCaption/stage1_scene_data.json")
    args = ap.parse_args()

    data = load_json_or_jsonl(args.input)

    # sample_token -> suffix -> view -> caption
    buckets = defaultdict(lambda: defaultdict(dict))
    split_map = {}

    skipped_bad_q = 0
    skipped_no_scene_shows = 0

    for item in data:
        sample_token = item.get("sample_token")
        if not sample_token:
            continue

        split_map.setdefault(sample_token, item.get("split", "train"))

        q = item.get("question", "")
        m = QUESTION_RE.match(q)
        if not m:
            skipped_bad_q += 1
            continue

        view = normalize_view(m.group(1))
        suffix = m.group(2).strip()
        if view not in VIEWS_REQUIRED:
            skipped_bad_q += 1
            continue

        cap = extract_scene_shows(item.get("answer", ""))
        if not cap:
            skipped_no_scene_shows += 1
            continue

        buckets[sample_token][suffix].setdefault(view, cap)

    outputs = []
    kept_tokens = 0
    dropped_tokens = 0

    for sample_token, suffix_map in buckets.items():
        complete_groups = []
        for suffix, view_map in suffix_map.items():
            if suffix not in ALLOWED_SUFFIXES:
                continue
            if set(view_map.keys()) >= set(VIEWS_REQUIRED.keys()):
                complete_groups.append((ALLOWED_SUFFIXES[suffix], suffix, view_map))

        if not complete_groups:
            dropped_tokens += 1
            continue

        complete_groups.sort(key=lambda x: x[0])
        _, chosen_suffix, chosen_view_map = complete_groups[0]

        assistant_text = build_group_caption(chosen_view_map)

        outputs.append({
            "split": split_map.get(sample_token, "train"),
            "sample_token": sample_token,
            "conversation": [
                {
                    "role": "user",
                    "content": "Please provide a caption for the following scene: <scene_start><SCENE><scene_end>",
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ],
        })
        kept_tokens += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"[Done] input items: {len(data)}")
    print(f"[Done] output groups: {len(outputs)}")
    print(f"[Info] kept_tokens={kept_tokens}, dropped_tokens={dropped_tokens}")
    print(f"[Info] skipped_bad_question_format={skipped_bad_q}")
    print(f"[Info] skipped_no_The_scene_shows_in_answer={skipped_no_scene_shows}")
    print(f"[Saved] {args.output}")

if __name__ == "__main__":
    main()