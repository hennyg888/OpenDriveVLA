import os
import sys
import time
import importlib
import pickle
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import mmcv
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import build_dataset
from llava.utils import pad_bevfeature
from tqdm import tqdm


def _add_repo_to_syspath():
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())
    return repo_root


_add_repo_to_syspath()

UNIAD_BEVFUSION_CONFIG = "/home/hhguo/OpenDriveVLA/projects/configs/uniad_bevfusion/pretrain.py"
UNIAD_BEVFUSION_CKPT = "/home/s56cai/OpenDriveVLA/projects/work_dirs/uniad_bevfusion/pretrain/epoch_5.pth"

EVAL_MOD = ["track", "map"]

JSON_PREFIX = os.path.join(
    "test",
    os.path.basename(UNIAD_BEVFUSION_CONFIG).replace(".py", ""),
    "latest",
)

# ===== Chunk writing config =====
CHUNK_SIZE = 50
# ===============================


def unwrap_dc(x):
    return x.data if isinstance(x, DC) else x


def unwrap_metas(img_metas):
    if isinstance(img_metas, DC):
        img_metas = img_metas.data
    while isinstance(img_metas, list):
        img_metas = img_metas[0]
    return img_metas, [img_metas]


def to_torch_list(x, device):
    out = []
    for a in x:
        if torch.is_tensor(a):
            out.append(a.to(device))
        else:
            out.append(torch.from_numpy(a).to(device))
    return out


def to_device(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(xx, device) for xx in x]
    return x


def strip_unwanted(d):
    if not isinstance(d, dict):
        return d
    d.pop("occ", None)
    d.pop("motion", None)
    d.pop("planning", None)
    d.pop("result_occ", None)
    d.pop("result_motion", None)
    d.pop("result_planning", None)
    for k, v in list(d.items()):
        if isinstance(v, dict):
            strip_unwanted(v)
    return d


def to_cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_cpu_tree(v) for v in x)
    return x


def pack_for_nuscenes_e2e(out: Dict[str, Any]) -> Dict[str, Any]:
    out = strip_unwanted(out)
    out = to_cpu_tree(out)

    rt = out.get("result_track", {}) or {}
    if isinstance(rt, list):
        rt = rt[0] if len(rt) > 0 and isinstance(rt[0], dict) else {}
    elif not isinstance(rt, dict):
        rt = {}
    rs = out.get("result_seg", {}) or {}

    packed = {}
    packed.update(rt)

    if "ret_iou" in rs:
        packed["ret_iou"] = rs["ret_iou"]
    
    # if "traj" in rt:
    #     packed["traj"] = rt["traj"]
    # if "traj_scores" in rt:
    #     packed["traj_scores"] = rt["traj_scores"]

    return packed


def init_dist():
    is_distributed = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)

    if is_distributed and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size, is_distributed


# ---------- Robust chunk I/O ----------

def atomic_dump_pkl(obj: Any, path: str):
    """Atomic pickle write (avoid truncated on crash)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_load_pkl(path: str):
    """Safe load pickle; raise if corrupted."""
    with open(path, "rb") as f:
        return pickle.load(f)


def list_chunks(rank_dir: str) -> List[str]:
    """List chunk_XXXXX.pkl in order."""
    if not os.path.exists(rank_dir):
        return []
    files = [f for f in os.listdir(rank_dir) if f.startswith("chunk_") and f.endswith(".pkl")]
    files.sort()
    return [os.path.join(rank_dir, f) for f in files]


def resume_state_from_chunks(rank_dir: str) -> Dict[str, Any]:
    """
    Determine:
      - done_n: how many samples already persisted for this rank
      - next_chunk_id: next chunk index to write
    Also handle last chunk corruption:
      - if last chunk load fails -> move to .corrupt and ignore it
    """
    chunks = list_chunks(rank_dir)
    done_n = 0
    kept_chunks: List[str] = []

    for p in chunks:
        try:
            arr = safe_load_pkl(p)
            if not isinstance(arr, list):
                raise RuntimeError(f"chunk is not list: {p} type={type(arr)}")
            done_n += len(arr)
            kept_chunks.append(p)
        except Exception as e:
            ts = time.strftime("%Y%m%d_%H%M%S")
            corrupt = p + f".corrupt.{ts}"
            try:
                os.replace(p, corrupt)
            except Exception:
                pass
            print(f"[WARN] drop corrupt chunk {p} -> {corrupt}: {repr(e)}", flush=True)
            # stop at first corrupt chunk because later ones are unlikely consistent
            break

    next_chunk_id = len(kept_chunks)
    return {"done_n": done_n, "next_chunk_id": next_chunk_id}




def main():
    local_rank, rank, world_size, is_distributed = init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(
        f"[rank{rank}] init_dist ok | local_rank={local_rank} world_size={world_size} "
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    cfg_bf = Config.fromfile(UNIAD_BEVFUSION_CONFIG)
    if cfg_bf.get("plugin", False):
        sys.path.insert(0, os.path.join(os.getcwd(), cfg_bf.plugin_dir))
        importlib.import_module("projects.mmdet3d_plugin")

    cfg_bf.data.test.eval_mod = EVAL_MOD

    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet3d.models import build_model as build_mmdet3d_model

    print(f"[rank{rank}] build Uniad_Bevfusion...", flush=True)
    cfg_tm = Config.fromfile(UNIAD_BEVFUSION_CONFIG)
    cfg_tm.model.pop("train_cfg", None)
    cfg_tm.model.pop("test_cfg", None)
    model_tm = build_mmdet3d_model(cfg_tm.model).to(device).eval()
    ckpt_tm = torch.load(UNIAD_BEVFUSION_CKPT, map_location="cpu")
    model_tm.load_state_dict(ckpt_tm.get("state_dict", ckpt_tm), strict=False)
    print(f"[rank{rank}] Uniad_Bevfusion ready", flush=True)

    dataset = build_dataset(cfg_bf.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=getattr(cfg_bf.data, "workers_per_gpu", 0),
        dist=is_distributed,
        shuffle=False,
        nonshuffler_sampler=getattr(cfg_bf.data, "nonshuffler_sampler", None),
    )

    # output dir (use absolute path to avoid cwd mismatch)
    global JSON_PREFIX
    JSON_PREFIX = os.path.abspath(JSON_PREFIX)

    if rank == 0:
        mmcv.mkdir_or_exist(JSON_PREFIX)
        print(f"[rank0] output dir: {JSON_PREFIX}", flush=True)

    print(f"[rank{rank}] before barrier A", flush=True)
    if is_distributed:
        dist.barrier()
    print(f"[rank{rank}] after barrier A", flush=True)

    # -------------------------
    # Chunk dir + resume
    # -------------------------
    mmcv.mkdir_or_exist(JSON_PREFIX)
    rank_dir = os.path.join(JSON_PREFIX, f"rank{rank:02d}")
    mmcv.mkdir_or_exist(rank_dir)

    st = resume_state_from_chunks(rank_dir)
    done_n = st["done_n"]
    chunk_id = st["next_chunk_id"]

    loader_len = len(data_loader)
    print(
        f"[rank{rank}] resume: done_n={done_n}, next_chunk_id={chunk_id}, "
        f"rank_dir={rank_dir}, loader_len={loader_len}",
        flush=True,
    )

    if done_n >= loader_len:
        print(
            f"[rank{rank}] resume complete -> skip dataloader loop "
            f"(done_n={done_n} >= loader_len={loader_len})",
            flush=True,
        )
        wrote, skipped = 0, 0
    else:
        # buffer for current chunk
        buffer: List[Any] = []

        t0 = time.time()
        wrote, skipped = 0, 0

        it = data_loader
        if rank == 0:
            it = tqdm(data_loader, total=loader_len, dynamic_ncols=True)

        # Iterate and SKIP first done_n samples for this rank
        for iter_idx, data in enumerate(it):
            if iter_idx < done_n:
                skipped += 1
                continue

            img = unwrap_dc(data["img"])
            points = unwrap_dc(data["points"])
            meta0, metas = unwrap_metas(data["img_metas"])

            timestamp = to_device(unwrap_dc(data.get("timestamp", None)), device)
            l2g_r_mat = to_device(unwrap_dc(data.get("l2g_r_mat", None)), device)
            l2g_t = to_device(unwrap_dc(data.get("l2g_t", None)), device)

            gt_lane_labels = to_device(unwrap_dc(data.get("gt_lane_labels", None)), device)
            gt_lane_bboxes = to_device(unwrap_dc(data.get("gt_lane_bboxes", None)), device)
            gt_lane_masks = to_device(unwrap_dc(data.get("gt_lane_masks", None)), device)

            if torch.is_tensor(points):
                points = [points[0].to(device)]
            else:
                points = [p.to(device) for p in points]

            if isinstance(img, list):
                img = torch.stack(img, dim=0).unsqueeze(0).to(device)
            else:
                img = img.to(device)

            camera2ego = to_torch_list(meta0["camera2ego"], device)
            lidar2ego = torch.from_numpy(meta0["lidar2ego"]).to(device)
            lidar2camera = to_torch_list(meta0["lidar2camera"], device)
            camera2lidar = to_torch_list(meta0["camera2lidar"], device)
            lidar2image = to_torch_list(meta0["lidar2image"], device)
            camera_intrinsics = to_torch_list(meta0["camera_intrinsics"], device)
            img_aug_matrix = to_torch_list(meta0["img_aug_matrix"], device)
            lidar_aug_matrix = torch.from_numpy(meta0["lidar_aug_matrix"]).to(device)

            with torch.no_grad():
                out = model_tm(
                    return_loss=False,
                    img=img,
                    points=points,
                    timestamp=timestamp,
                    l2g_r_mat=l2g_r_mat,
                    l2g_t=l2g_t,
                    gt_lane_labels=gt_lane_labels,
                    gt_lane_bboxes=gt_lane_bboxes,
                    gt_lane_masks=gt_lane_masks,
                    camera2ego=camera2ego,
                    lidar2ego=lidar2ego,
                    lidar2camera=lidar2camera,
                    camera2lidar=camera2lidar,
                    lidar2image=lidar2image,
                    camera_intrinsics=camera_intrinsics,
                    img_aug_matrix=img_aug_matrix,
                    lidar_aug_matrix=lidar_aug_matrix,
                    img_metas=metas,
                )

            packed = pack_for_nuscenes_e2e(out)
            buffer.append(packed)
            wrote += 1

            # flush a full chunk
            if len(buffer) >= CHUNK_SIZE:
                out_path = os.path.join(rank_dir, f"chunk_{chunk_id:05d}.pkl")
                atomic_dump_pkl(buffer, out_path)
                chunk_id += 1
                buffer = []

        # flush remaining
        if len(buffer) > 0:
            out_path = os.path.join(rank_dir, f"chunk_{chunk_id:05d}.pkl")
            atomic_dump_pkl(buffer, out_path)
            chunk_id += 1
            buffer = []

        dt = time.time() - t0
        print(
            f"[rank{rank}] done inference: wrote={wrote} skipped={skipped} total_seen={wrote+skipped}, {dt:.1f}s",
            flush=True,
        )

    print(f"[rank{rank}] rank_dir: {rank_dir} chunks={len(list_chunks(rank_dir))}", flush=True)

    # -------------------------
    # Sync + merge (rank0) - FIXED: interleave like DistributedSampler (stride)
    # -------------------------
    if is_distributed:
        dist.barrier()
        if rank != 0:
            return

    assert rank == 0
    size = len(dataset)

    # 1) load each rank's list in its own order
    rank_lists: List[List[Any]] = []
    for r in range(world_size):
        rdir = os.path.join(JSON_PREFIX, f"rank{r:02d}")
        chunks = list_chunks(rdir)
        if len(chunks) == 0:
            raise RuntimeError(f"missing chunks for rank {r}: {rdir}")

        r_items: List[Any] = []
        for p in chunks:
            arr = safe_load_pkl(p)
            if not isinstance(arr, list):
                raise RuntimeError(f"bad chunk type: {p} type={type(arr)}")
            r_items.extend(arr)
        rank_lists.append(r_items)

    # 2) interleave to reconstruct global dataset order: [r0[0], r1[0], ..., rK[0], r0[1], ...]
    max_len = max(len(x) for x in rank_lists)
    ordered_results: List[Any] = []
    for i in range(max_len):
        for r in range(world_size):
            if i < len(rank_lists[r]):
                ordered_results.append(rank_lists[r][i])

    # dataloader may pad some samples
    ordered_results = ordered_results[:size]
    if len(ordered_results) != size:
        raise RuntimeError(f"missing_results={size - len(ordered_results)} (got {len(ordered_results)} / {size})")

    res = dataset.evaluate(ordered_results, jsonfile_prefix=JSON_PREFIX)
    print(res, flush=True)
    print(f"[rank0] evaluate done. results in: {JSON_PREFIX}", flush=True)



if __name__ == "__main__":
    main()
