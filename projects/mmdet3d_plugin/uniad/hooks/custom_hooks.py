from mmcv.runner.hooks.hook import HOOKS, Hook
import os.path as osp
import torch


def _get_encoder_param_sample(model):
    """Get one scalar from encoder for debugging (whether step() updates it)."""
    m = getattr(model, 'module', model)
    if not hasattr(m, 'track_map_former'):
        return None
    tmf = m.track_map_former
    if not hasattr(tmf, 'pts_bbox_head'):
        return None
    head = tmf.pts_bbox_head
    if not hasattr(head, 'transformer'):
        return None
    enc = getattr(head.transformer, 'encoder', None)
    if enc is None or not hasattr(enc, 'layers') or len(enc.layers) == 0:
        return None
    layer0 = enc.layers[0]
    if not hasattr(layer0, 'attentions') or len(layer0.attentions) == 0:
        return None
    att = layer0.attentions[0]
    if not hasattr(att, 'sampling_offsets') or not hasattr(att.sampling_offsets, 'weight'):
        return None
    w = att.sampling_offsets.weight
    return w.data[0, 0].item()


# state_dict key for the same param (unwrapped model, no 'module.' prefix)
_ENCODER_PARAM_KEY = (
    'track_map_former.pts_bbox_head.transformer.encoder.layers.0.'
    'attentions.0.sampling_offsets.weight'
)


def _get_encoder_param_from_state_dict(state_dict):
    """Read encoder param sample from checkpoint state_dict."""
    if state_dict is None or _ENCODER_PARAM_KEY not in state_dict:
        return None
    t = state_dict[_ENCODER_PARAM_KEY]
    if not isinstance(t, torch.Tensor) or t.numel() < 1:
        return None
    return t.flatten()[0].item()


def _is_rank0() -> bool:
    try:
        from mmcv.runner import get_dist_info
        rank, _ = get_dist_info()
        return rank == 0
    except Exception:
        return False


def _encoder_keys(state_dict):
    if not isinstance(state_dict, dict):
        return []
    prefix = 'track_map_former.pts_bbox_head.transformer.encoder.'
    return [k for k in state_dict.keys() if k.startswith(prefix)]


def _tensor_stats(a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    return d.max().item(), d.sum().item()


@HOOKS.register_module()
class DebugEncoderParamHook(Hook):
    """Log one encoder param value after optimizer.step() to verify encoder is updated."""

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            try:
                from mmcv.runner import get_dist_info
                rank, _ = get_dist_info()
                if rank != 0:
                    return
            except Exception:
                pass
            val = _get_encoder_param_sample(runner.model)
            if val is not None:
                runner.logger.info(
                    f'[DebugEncoderParamHook] iter={runner.iter} '
                    f'encoder_param_sample={val:.6f}'
                )


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class SaveInitialCheckpointHook(Hook):
    """Save initial (pre-train) weights as epoch_0.pth before training starts."""

    def __init__(self, filename: str = 'epoch_0.pth'):
        self.filename = filename
        self._saved = False

    def before_run(self, runner):
        if self._saved:
            return
        if not _is_rank0():
            return
        ckpt_path = osp.join(runner.work_dir, self.filename)
        if osp.isfile(ckpt_path):
            runner.logger.info(
                f'[SaveInitialCheckpointHook] initial checkpoint already exists: {ckpt_path}'
            )
            self._saved = True
            return
        try:
            from mmcv.runner import save_checkpoint
        except Exception as e:
            runner.logger.warning(
                f'[SaveInitialCheckpointHook] failed to import save_checkpoint: {e}'
            )
            return
        # runner.model 可能是 DDP / DataParallel，save_checkpoint 内部会自动 unwrap
        save_checkpoint(runner.model, ckpt_path, optimizer=None, meta=dict(epoch=0, iter=0))
        runner.logger.info(
            f'[SaveInitialCheckpointHook] saved initial checkpoint to {ckpt_path}'
        )
        self._saved = True


@HOOKS.register_module()
class VerifyEncoderCheckpointHook(Hook):
    """After each epoch save, verify ckpt matches memory; optionally compare encoder in epoch_{n-1} vs epoch_{n}."""

    def __init__(self, compare_prev_epoch: bool = True, topk: int = 5):
        self.compare_prev_epoch = compare_prev_epoch
        self.topk = topk

    def after_train_epoch(self, runner):
        if not _is_rank0():
            return

        # CheckpointHook saves to work_dir by default, filename epoch_{epoch+1}.pth
        cur_epoch = runner.epoch + 1
        ckpt_path = osp.join(runner.work_dir, f'epoch_{cur_epoch}.pth')
        if not osp.isfile(ckpt_path):
            return
        in_mem = _get_encoder_param_sample(runner.model)
        if in_mem is None:
            return
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict') if isinstance(ckpt, dict) else None
            on_disk = _get_encoder_param_from_state_dict(state)
        except Exception as e:
            runner.logger.warning(f'[VerifyEncoderCheckpointHook] load {ckpt_path} failed: {e}')
            return
        if on_disk is None:
            return
        diff = abs(in_mem - on_disk)
        if diff > 1e-6:
            runner.logger.error(
                f'[VerifyEncoderCheckpointHook] encoder param mismatch: '
                f'in_memory={in_mem:.6f} vs checkpoint={on_disk:.6f} (diff={diff:.6f}) '
                f'ckpt={ckpt_path}'
            )
        else:
            runner.logger.info(
                f'[VerifyEncoderCheckpointHook] encoder param OK in_memory={in_mem:.6f} '
                f'checkpoint={on_disk:.6f}'
            )

        # Compare encoder tensors between consecutive epoch checkpoints (epoch_{n-1} vs epoch_{n})
        if not self.compare_prev_epoch or cur_epoch < 2:
            return
        prev_path = osp.join(runner.work_dir, f'epoch_{cur_epoch - 1}.pth')
        if not osp.isfile(prev_path):
            return
        try:
            prev_ckpt = torch.load(prev_path, map_location='cpu')
            prev_state = prev_ckpt.get('state_dict') if isinstance(prev_ckpt, dict) else None
            cur_state = state
        except Exception as e:
            runner.logger.warning(f'[VerifyEncoderCheckpointHook] load prev {prev_path} failed: {e}')
            return

        prev_keys = set(_encoder_keys(prev_state))
        cur_keys = set(_encoder_keys(cur_state))
        common = sorted(prev_keys & cur_keys)
        if not common:
            runner.logger.warning(
                f'[VerifyEncoderCheckpointHook] no encoder keys found for compare: prev={prev_path} cur={ckpt_path}'
            )
            return

        max_abs = 0.0
        sum_abs = 0.0
        same = 0
        diff_cnt = 0
        top = []  # (max_abs_diff, sum_abs_diff, key)

        for k in common:
            a = prev_state[k]
            b = cur_state[k]
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                continue
            if a.shape != b.shape:
                continue
            a = a.float()
            b = b.float()
            m, s = _tensor_stats(a, b)
            max_abs = max(max_abs, m)
            sum_abs += s
            if s == 0.0:
                same += 1
            else:
                diff_cnt += 1
            top.append((m, s, k))

        top.sort(key=lambda x: x[0], reverse=True)
        top = top[: max(1, int(self.topk))]

        runner.logger.info(
            f'[VerifyEncoderCheckpointHook] encoder ckpt diff epoch_{cur_epoch-1} -> epoch_{cur_epoch}: '
            f'common_keys={len(common)} same={same} diff={diff_cnt} '
            f'max_abs_diff={max_abs:.6e} sum_abs_diff={sum_abs:.6e}'
        )
        for m, s, k in top:
            runner.logger.info(
                f'[VerifyEncoderCheckpointHook] top_diff max_abs={m:.6e} sum_abs={s:.6e} key={k}'
            )


