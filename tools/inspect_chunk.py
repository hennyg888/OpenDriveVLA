"""Inspect chunk_*.pkl files produced by `pretrain_eval.py`.

Usage examples:
  python tools/inspect_chunk.py /path/to/chunk_00000.pkl           # print summary + first item
  python tools/inspect_chunk.py /path/to/rank00 --list           # list chunk files in dir
  python tools/inspect_chunk.py /path/to/rank00 -i 2 --limit 3   # print 3 items from chunk index 2
  python tools/inspect_chunk.py /path/to/chunk_00000.pkl --json  # pretty JSON output

The script safely converts torch tensors / numpy arrays to Python lists for printing.
"""
from __future__ import annotations
import argparse
import os
import pickle
import pprint
import json
from typing import Any, List

try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None


def list_chunk_files(directory: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.startswith("chunk_") and f.endswith(".pkl")]
    files.sort()
    return [os.path.join(directory, f) for f in files]


def _normalize(obj: Any) -> Any:
    """Convert tensors/ndarrays and common ML structs to Python-native types for safe printing/JSON.

    - Handles torch.Tensor and numpy.ndarray
    - Converts objects exposing `.tensor`, `.numpy()` or `__array__` (e.g. LiDARInstance3DBoxes)
    - Recurses into dicts / lists / tuples
    - Falls back to a small serializable dict with the object's type and repr
    """
    # torch tensor
    try:
        if torch is not None and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # numpy array
    try:
        if np is not None and isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # objects that expose a `.tensor` attribute (e.g. LiDARInstance3DBoxes)
    if hasattr(obj, "tensor"):
        try:
            return _normalize(getattr(obj, "tensor"))
        except Exception:
            pass

    # objects that provide a numpy() method
    if hasattr(obj, "numpy") and callable(getattr(obj, "numpy")):
        try:
            return _normalize(obj.numpy())
        except Exception:
            pass

    # objects that implement __array__
    try:
        if hasattr(obj, "__array__") and callable(getattr(obj, "__array__")):
            return _normalize(obj.__array__())
    except Exception:
        pass

    # dict / list / tuple -> recurse
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]

    # final fallback: provide a serializable description
    return {"__type__": obj.__class__.__name__, "repr": str(obj)}


def load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def print_summary(items: List[Any], show_keys: bool = True) -> None:
    print(f"file contains {len(items)} item(s)")
    if len(items) == 0:
        return
    sample = items[0]
    print("--- sample[0] type/keys summary ---")
    print(f"type: {type(sample)}")
    if isinstance(sample, dict) and show_keys:
        print("keys:")
        for k, v in sample.items():
            print(f"  {k}: {type(v)}")

def describe_obj(obj: Any) -> str:
    """Return a compact one-line description for `obj` (type, shape/len/dtype when available)."""
    # torch tensor
    try:
        if torch is not None and isinstance(obj, torch.Tensor):
            return f"Tensor{tuple(obj.shape)} dtype={getattr(obj, 'dtype', None)}"
    except Exception:
        pass

    # numpy array
    try:
        if np is not None and isinstance(obj, np.ndarray):
            return f"ndarray{obj.shape} dtype={getattr(obj, 'dtype', None)}"
    except Exception:
        pass

    # objects with a `.tensor` attribute (e.g. LiDARInstance3DBoxes)
    if hasattr(obj, "tensor"):
        try:
            return describe_obj(getattr(obj, "tensor"))
        except Exception:
            pass

    # objects exposing numpy()
    if hasattr(obj, "numpy") and callable(getattr(obj, "numpy")):
        try:
            return describe_obj(obj.numpy())
        except Exception:
            pass

    if isinstance(obj, dict):
        return f"dict(len={len(obj)})"
    if isinstance(obj, (list, tuple)):
        return f"list(len={len(obj)})"

    if hasattr(obj, "shape"):
        try:
            return f"{obj.__class__.__name__}{tuple(getattr(obj, 'shape'))}"
        except Exception:
            pass

    return obj.__class__.__name__


def list_objects_in_items(items: List[Any], limit: int = 1) -> None:
    """Print top-level object/key types for up to `limit` items."""
    n = min(len(items), max(0, limit))
    for i in range(n):
        it = items[i]
        print(f"item[{i}] type={type(it)}")
        if isinstance(it, dict):
            for k, v in it.items():
                print(f"  {k}: {describe_obj(v)}")
        else:
            print(f"  value: {describe_obj(it)}")


def summarize_keys(items: List[Any]) -> None:
    """Aggregate presence and value-type descriptions for top-level keys across all items."""
    total = len(items)
    stats = {}
    for it in items:
        if isinstance(it, dict):
            for k, v in it.items():
                rec = stats.setdefault(k, {"count": 0, "types": {}})
                rec["count"] += 1
                key_desc = describe_obj(v)
                rec["types"][key_desc] = rec["types"].get(key_desc, 0) + 1
        else:
            rec = stats.setdefault("__item__", {"count": 0, "types": {}})
            rec["count"] += 1
            key_desc = describe_obj(it)
            rec["types"][key_desc] = rec["types"].get(key_desc, 0) + 1

    print(f"summary across {total} item(s):")
    for k, rec in sorted(stats.items(), key=lambda kv: -kv[1]["count"]):
        types_summary = ", ".join(f"{t}({c})" for t, c in sorted(rec["types"].items(), key=lambda x: -x[1]))
        print(f"  {k}: present in {rec['count']}/{total} items; types: {types_summary}")

def main():
    p = argparse.ArgumentParser(description="Inspect chunk_*.pkl created by pretrain_eval.py")
    p.add_argument("path", help="path to a chunk .pkl file or a rank directory containing chunk_*.pkl")
    p.add_argument("-i", "--chunk-index", type=int, default=0, help="index of chunk file when a directory is given")
    p.add_argument("-n", "--limit", type=int, default=1, help="how many items from the chunk to print (default=1)")
    p.add_argument("--list", action="store_true", help="list chunk files when a directory is provided")
    p.add_argument("--json", action="store_true", help="output JSON-serializable representation")
    p.add_argument("--show-keys", action="store_true", help="show top-level keys of a sample (for dict items)")
    p.add_argument("--objects", action="store_true", help="list top-level objects/keys and types for each item (respect -n limit)")
    p.add_argument("--objects-summary", action="store_true", help="show aggregated key/type summary across all items in the chunk")
    p.add_argument("--names", action="store_true", help="print only top-level object/key names for the first item in the chunk")
    p.add_argument("--key", "-k", dest="key", type=str, help="print the value of this top-level key from item[0]")
    args = p.parse_args()

    if os.path.isdir(args.path):
        files = list_chunk_files(args.path)
        if len(files) == 0:
            print(f"no chunk_*.pkl files found in directory: {args.path}")
            return
        if args.list:
            for i, f in enumerate(files):
                print(f"[{i}] {f}")
            return
        if args.chunk_index < 0 or args.chunk_index >= len(files):
            print(f"chunk index out of range: {args.chunk_index} (0..{len(files)-1})")
            return
        target = files[args.chunk_index]
    else:
        if not os.path.exists(args.path):
            raise SystemExit(f"file not found: {args.path}")
        target = args.path

    items = load_pkl(target)
    if not isinstance(items, list):
        print("WARNING: expected a list in the pickle, got:", type(items))
        pprint.pprint(items)
        return

    print(f"Loaded: {target}")

    # quick: print only top-level key names for the first item
    if args.names:
        if len(items) == 0:
            print("no items in the chunk")
            return
        sample = items[0]
        if isinstance(sample, dict):
            print("item[0] keys:")
            for k in sample.keys():
                print(f"  {k}")
        else:
            print(f"item[0] is not a dict (type={type(sample)})")
        return

    # print a specific top-level key from item[0]
    if args.key:
        if len(items) == 0:
            print("no items in the chunk")
            return
        sample = items[0]
        if not isinstance(sample, dict):
            print(f"item[0] is not a dict (type={type(sample)})")
            return
        if args.key not in sample:
            keys = ", ".join(str(k) for k in sample.keys())
            print(f"key '{args.key}' not found in item[0]. Available keys: {keys}")
            return
        val = sample[args.key]
        print(f"item[0]['{args.key}']:")
        if args.json:
            ser = _normalize(val)
            print(json.dumps(ser, indent=2, ensure_ascii=False))
        else:
            pp = pprint.PrettyPrinter(depth=6, compact=False)
            pp.pprint(_normalize(val))
        return

    # objects summary/listing requested?
    if args.objects_summary:
        summarize_keys(items)
        return

    if args.objects:
        list_objects_in_items(items, limit=args.limit)
        return

    # default behaviour: show sample keys (if requested) and print items (json/pretty)
    print_summary(items, show_keys=args.show_keys)

    to_print = min(len(items), max(0, args.limit))
    for idx in range(to_print):
        print("\n--- item[{}] ---".format(idx))
        item = items[idx]
        if args.json:
            ser = _normalize(item)
            print(json.dumps(ser, indent=2, ensure_ascii=False))
        else:
            ser = _normalize(item)
            pp = pprint.PrettyPrinter(depth=6, compact=False)
            pp.pprint(ser)


if __name__ == "__main__":
    main()
