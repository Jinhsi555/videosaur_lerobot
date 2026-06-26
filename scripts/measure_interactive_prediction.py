#!/usr/bin/env python3
"""Measure feature-target and decoder-prediction agreement for interactive viewers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from videosaur import configuration, models, utils
from videosaur.inference import _move_to_device, _resolve_device, load_inference_config, prepare_video
from videosaur.interactive_export import (
    DEFAULT_FEATURE_KEY,
    DEFAULT_PREDICTION_KEY,
    resolve_prediction_dims,
    resolve_prediction_remove_last_n_frames,
    resolve_similarity_config,
)


DEFAULT_VIEWER_ROOT = "inference_output/visualization_ssv2"
DEFAULT_CHECKPOINT = (
    "logs/videosaur/2026-06-11-15-51-53_lerobot_something_something_v2_2/"
    "checkpoints/step=100000-v1.ckpt"
)
DEFAULT_MODEL_CONFIG = "configs/videosaur/lerobot_something_something_v2.yml"
DEFAULT_INFERENCE_CONFIG = "configs/inference/movi_c.yml"


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    *,
    disable_pretrained: bool = True,
):
    overrides = ["model.encoder.backbone.pretrained=false"] if disable_pretrained else None
    config = configuration.load_config(config_path, overrides)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def finite_softmax(logits: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(logits)
    has_finite = finite.any(dim=-1, keepdim=True)
    safe = torch.where(finite, logits, torch.full_like(logits, -torch.inf))
    safe = torch.where(has_finite, safe, torch.zeros_like(safe))
    probs = torch.softmax(safe, dim=-1)
    if not bool(has_finite.all()):
        uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
        probs = torch.where(has_finite, probs, uniform)
    return probs


class Accumulator:
    def __init__(self, n_patches: int):
        self.n_patches = n_patches
        self.elem = 0
        self.rows = 0
        self.sums: dict[str, float] = {}
        self.maxs: dict[str, float] = {}
        self.top_counts = {
            "top1": 0,
            "top5": 0,
            "top10": 0,
            "aff_target_argmax": 0,
            "pred_aff_argmax": 0,
        }
        self.pred_hist = torch.zeros(n_patches, dtype=torch.long)
        self.target_hist = torch.zeros(n_patches, dtype=torch.long)

    def add_sum(self, name: str, value: float):
        self.sums[name] = self.sums.get(name, 0.0) + float(value)

    def add_max(self, name: str, value: float):
        self.maxs[name] = max(self.maxs.get(name, -float("inf")), float(value))

    def update(self, affinity: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor):
        _, n_patches, _ = target.shape
        elem = target.numel()
        rows = target.shape[0] * n_patches
        self.elem += elem
        self.rows += rows

        for prefix, lhs, rhs in (
            ("affinity_target", affinity, target),
            ("target_prediction", target, prediction),
            ("affinity_prediction", affinity, prediction),
        ):
            diff = rhs - lhs
            abs_diff = diff.abs()
            self.add_sum(f"{prefix}_mae", abs_diff.sum().item())
            self.add_sum(f"{prefix}_mse", (diff * diff).sum().item())
            self.add_max(f"{prefix}_max_abs", abs_diff.max().item())

        diff = prediction - target
        self.add_sum("target_prediction_mean_signed", diff.sum().item())
        self.add_sum("target_prediction_tv", (0.5 * diff.abs().sum(dim=-1)).sum().item())

        eps = 1e-12
        log_prediction = torch.log(prediction.clamp_min(eps))
        log_target = torch.log(target.clamp_min(eps))
        ce = -(target * log_prediction).sum(dim=-1)
        entropy = -(target * log_target).sum(dim=-1)
        kl = ce - entropy
        self.add_sum("ce_target_prediction", ce.sum().item())
        self.add_sum("entropy_target", entropy.sum().item())
        self.add_sum("kl_target_prediction", kl.sum().item())

        dot = (target * prediction).sum(dim=-1)
        norm = target.norm(dim=-1) * prediction.norm(dim=-1)
        self.add_sum("prob_cos_target_prediction", (dot / norm.clamp_min(eps)).sum().item())

        target_centered = target - target.mean(dim=-1, keepdim=True)
        pred_centered = prediction - prediction.mean(dim=-1, keepdim=True)
        corr = (target_centered * pred_centered).sum(dim=-1) / (
            target_centered.norm(dim=-1) * pred_centered.norm(dim=-1)
        ).clamp_min(eps)
        self.add_sum("prob_corr_target_prediction", corr.sum().item())

        target_peak, target_idx = target.max(dim=-1)
        pred_peak, pred_idx = prediction.max(dim=-1)
        affinity_idx = affinity.max(dim=-1).indices
        self.add_sum("target_peak", target_peak.sum().item())
        self.add_sum("prediction_peak", pred_peak.sum().item())
        self.add_sum(
            "prediction_prob_at_target_top1",
            prediction.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1).sum().item(),
        )
        self.add_sum(
            "target_prob_at_prediction_top1",
            target.gather(-1, pred_idx.unsqueeze(-1)).squeeze(-1).sum().item(),
        )

        top10 = torch.topk(prediction, k=min(10, n_patches), dim=-1).indices
        target_expanded = target_idx.unsqueeze(-1)
        self.top_counts["top1"] += int((pred_idx == target_idx).sum().item())
        self.top_counts["top5"] += int(
            (top10[..., : min(5, n_patches)] == target_expanded).any(dim=-1).sum().item()
        )
        self.top_counts["top10"] += int((top10 == target_expanded).any(dim=-1).sum().item())
        self.top_counts["aff_target_argmax"] += int((affinity_idx == target_idx).sum().item())
        self.top_counts["pred_aff_argmax"] += int((pred_idx == affinity_idx).sum().item())

        self.pred_hist += torch.bincount(pred_idx.reshape(-1).detach().cpu(), minlength=n_patches)
        self.target_hist += torch.bincount(
            target_idx.reshape(-1).detach().cpu(), minlength=n_patches
        )

    def merge(self, other: "Accumulator"):
        self.elem += other.elem
        self.rows += other.rows
        for key, value in other.sums.items():
            self.sums[key] = self.sums.get(key, 0.0) + value
        for key, value in other.maxs.items():
            self.maxs[key] = max(self.maxs.get(key, -float("inf")), value)
        for key, value in other.top_counts.items():
            self.top_counts[key] = self.top_counts.get(key, 0) + value
        self.pred_hist += other.pred_hist
        self.target_hist += other.target_hist

    def summary(self):
        rows = max(1, self.rows)
        elem = max(1, self.elem)
        return {
            "rows": self.rows,
            "elem": self.elem,
            "aff_target_mae": self.sums.get("affinity_target_mae", 0.0) / elem,
            "aff_target_rmse": math.sqrt(self.sums.get("affinity_target_mse", 0.0) / elem),
            "aff_pred_mae": self.sums.get("affinity_prediction_mae", 0.0) / elem,
            "aff_pred_rmse": math.sqrt(self.sums.get("affinity_prediction_mse", 0.0) / elem),
            "target_pred_mae": self.sums.get("target_prediction_mae", 0.0) / elem,
            "target_pred_rmse": math.sqrt(self.sums.get("target_prediction_mse", 0.0) / elem),
            "target_pred_max_abs": self.maxs.get("target_prediction_max_abs", 0.0),
            "target_pred_mean_signed": self.sums.get("target_prediction_mean_signed", 0.0)
            / elem,
            "tv": self.sums.get("target_prediction_tv", 0.0) / rows,
            "ce": self.sums.get("ce_target_prediction", 0.0) / rows,
            "entropy": self.sums.get("entropy_target", 0.0) / rows,
            "kl": self.sums.get("kl_target_prediction", 0.0) / rows,
            "prob_cos": self.sums.get("prob_cos_target_prediction", 0.0) / rows,
            "prob_corr": self.sums.get("prob_corr_target_prediction", 0.0) / rows,
            "target_peak": self.sums.get("target_peak", 0.0) / rows,
            "pred_peak": self.sums.get("prediction_peak", 0.0) / rows,
            "pred_at_target_top1": self.sums.get("prediction_prob_at_target_top1", 0.0)
            / rows,
            "target_at_pred_top1": self.sums.get("target_prob_at_prediction_top1", 0.0)
            / rows,
            "top1": self.top_counts["top1"] / rows,
            "top5": self.top_counts["top5"] / rows,
            "top10": self.top_counts["top10"] / rows,
            "aff_target_argmax": self.top_counts["aff_target_argmax"] / rows,
            "pred_aff_argmax": self.top_counts["pred_aff_argmax"] / rows,
            "pred_unique": int((self.pred_hist > 0).sum().item()),
            "target_unique": int((self.target_hist > 0).sum().item()),
            "pred_top_share": float(self.pred_hist.max().item() / rows),
            "target_top_share": float(self.target_hist.max().item() / rows),
        }


def read_manifest_paths(viewer_root: Path):
    if viewer_root.name.endswith("-viewer") and (viewer_root / "manifest.json").exists():
        return [viewer_root / "manifest.json"]
    return sorted(viewer_root.glob("*-viewer/manifest.json"))


def process_video(model, model_config, config, manifest_path: Path, device, chunk_size: int):
    manifest = json.loads(manifest_path.read_text())
    video_path = manifest["input_path"]
    inputs = prepare_video(video_path, config.input.transforms)
    inputs = _move_to_device(inputs, device)
    with torch.inference_mode():
        outputs = model(inputs)

    n_frames = int(manifest["n_frames"])
    feature_key = manifest.get("feature_key", DEFAULT_FEATURE_KEY)
    prediction_key = manifest.get("prediction_key", DEFAULT_PREDICTION_KEY)
    features = utils.read_path(outputs, path=feature_key, error=False)
    if features is None:
        features = utils.read_path(outputs, path="encoder.backbone_features")
    prediction = utils.read_path(outputs, path=prediction_key)

    if features.ndim != 4 or features.shape[0] != 1 or features.shape[1] != n_frames:
        raise RuntimeError(f"Bad features shape {tuple(features.shape)} for {video_path}")
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] != n_frames:
        raise RuntimeError(f"Bad prediction shape {tuple(prediction.shape)} for {video_path}")

    n_patches = features.shape[2]
    if prediction.shape[2] != n_patches:
        raise RuntimeError(
            f"Prediction patches {prediction.shape[2]} != feature patches {n_patches}"
        )

    similarity_config = resolve_similarity_config(model_config)
    time_shift = int(similarity_config.get("time_shift", 1))
    remove_last = resolve_prediction_remove_last_n_frames(model_config, time_shift)
    pred_dims = resolve_prediction_dims({}, model_config, prediction.shape[-1], n_patches)
    prediction_frames = min(max(0, n_frames - remove_last), max(0, n_frames - time_shift))

    if manifest.get("prediction_dims") and list(pred_dims) != list(manifest["prediction_dims"]):
        print(
            f"WARN pred_dims mismatch manifest={manifest.get('prediction_dims')} "
            f"resolved={pred_dims}",
            flush=True,
        )
    if manifest.get("prediction_frames") and int(manifest["prediction_frames"]) != prediction_frames:
        print(
            f"WARN prediction_frames mismatch manifest={manifest.get('prediction_frames')} "
            f"resolved={prediction_frames}",
            flush=True,
        )

    src = features[0, :prediction_frames].float()
    dst = features[0, time_shift : time_shift + prediction_frames].float()
    if bool(similarity_config.get("normalize", True)):
        src = F.normalize(src, p=2.0, dim=-1)
        dst = F.normalize(dst, p=2.0, dim=-1)
    logits = prediction[0, :prediction_frames, :, pred_dims[0] : pred_dims[1]].float()

    acc = Accumulator(n_patches)
    temperature = float(similarity_config.get("temperature", 1.0) or 1.0)
    threshold = similarity_config.get("threshold")
    mask_diagonal = bool(similarity_config.get("mask_diagonal", False))
    softmax = bool(similarity_config.get("softmax", True))

    for start in range(0, prediction_frames, chunk_size):
        end = min(prediction_frames, start + chunk_size)
        affinity = torch.bmm(src[start:end], dst[start:end].transpose(1, 2))
        target_logits = affinity.clone()
        padding = -torch.inf if softmax else -1.0 / temperature
        if threshold is not None:
            target_logits = target_logits.masked_fill(target_logits < float(threshold), padding)
        target_logits = target_logits / temperature
        if mask_diagonal:
            diag = torch.eye(n_patches, dtype=torch.bool, device=device).unsqueeze(0)
            target_logits = target_logits.masked_fill(diag, padding)
        target = finite_softmax(target_logits) if softmax else target_logits
        prediction_prob = torch.softmax(logits[start:end], dim=-1)
        acc.update(affinity, target, prediction_prob)
        del affinity, target_logits, target, prediction_prob

    del inputs, outputs, features, prediction, src, dst, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return video_path, prediction_frames, acc


def print_csv(rows):
    print("\nPER_VIDEO_CSV")
    print(
        "video,pred_frames,rows,aff_target_mae,aff_pred_mae,target_pred_mae,"
        "target_pred_rmse,target_pred_max_abs,tv,kl,ce,entropy,prob_cos,prob_corr,"
        "top1,top5,top10,pred_at_target_top1,target_peak,pred_peak,pred_unique,pred_top_share"
    )
    for video, pred_frames, summary in rows:
        print(
            ",".join(
                [
                    video,
                    str(pred_frames),
                    str(summary["rows"]),
                    f"{summary['aff_target_mae']:.8f}",
                    f"{summary['aff_pred_mae']:.8f}",
                    f"{summary['target_pred_mae']:.8f}",
                    f"{summary['target_pred_rmse']:.8f}",
                    f"{summary['target_pred_max_abs']:.8f}",
                    f"{summary['tv']:.8f}",
                    f"{summary['kl']:.8f}",
                    f"{summary['ce']:.8f}",
                    f"{summary['entropy']:.8f}",
                    f"{summary['prob_cos']:.8f}",
                    f"{summary['prob_corr']:.8f}",
                    f"{summary['top1']:.8f}",
                    f"{summary['top5']:.8f}",
                    f"{summary['top10']:.8f}",
                    f"{summary['pred_at_target_top1']:.8f}",
                    f"{summary['target_peak']:.8f}",
                    f"{summary['pred_peak']:.8f}",
                    str(summary["pred_unique"]),
                    f"{summary['pred_top_share']:.8f}",
                ]
            )
        )


def print_overall(summary):
    print("\nOVERALL")
    for key in [
        "rows",
        "elem",
        "aff_target_mae",
        "aff_target_rmse",
        "aff_pred_mae",
        "aff_pred_rmse",
        "target_pred_mae",
        "target_pred_rmse",
        "target_pred_max_abs",
        "target_pred_mean_signed",
        "tv",
        "kl",
        "ce",
        "entropy",
        "prob_cos",
        "prob_corr",
        "top1",
        "top5",
        "top10",
        "aff_target_argmax",
        "pred_aff_argmax",
        "pred_at_target_top1",
        "target_at_pred_top1",
        "target_peak",
        "pred_peak",
        "pred_unique",
        "target_unique",
        "pred_top_share",
        "target_top_share",
    ]:
        print(f"{key}={summary[key]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run inference for interactive viewer manifests and measure raw affinity, "
            "feature target, and decoder prediction differences."
        )
    )
    parser.add_argument(
        "viewer_root",
        nargs="?",
        default=DEFAULT_VIEWER_ROOT,
        help="Viewer directory or a parent directory containing *-viewer/manifest.json.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--inference-config", default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="Override config device, for example cuda/cpu.")
    parser.add_argument(
        "--keep-pretrained",
        action="store_true",
        help="Do not force encoder.backbone.pretrained=false while constructing the model.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Measure only the first N viewers.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifests = read_manifest_paths(Path(args.viewer_root))
    if args.limit is not None:
        manifests = manifests[: args.limit]
    if not manifests:
        raise SystemExit(f"No manifests found under {args.viewer_root}")

    overrides = [
        f"checkpoint={args.checkpoint}",
        f"model_config={args.model_config}",
    ]
    if args.device:
        overrides.append(f"device={args.device}")
    config = load_inference_config(args.inference_config, overrides)
    device = _resolve_device(config)
    print(f"device={device} manifests={len(manifests)} checkpoint={args.checkpoint}", flush=True)

    model, model_config = load_model_from_checkpoint(
        config.checkpoint,
        config.model_config,
        disable_pretrained=not args.keep_pretrained,
    )
    model.initializer.n_slots = config.n_slots
    model.to(device).eval()

    overall = None
    rows = []
    for idx, manifest_path in enumerate(manifests, 1):
        print(f"[{idx}/{len(manifests)}] {manifest_path.parent.name}", flush=True)
        video_path, pred_frames, acc = process_video(
            model,
            model_config,
            config,
            manifest_path,
            device,
            args.chunk_size,
        )
        if overall is None:
            overall = Accumulator(acc.n_patches)
        overall.merge(acc)
        summary = acc.summary()
        rows.append((Path(video_path).name, pred_frames, summary))
        print(
            f"  frames={pred_frames} top1={summary['top1']*100:.2f}% "
            f"top5={summary['top5']*100:.2f}% KL={summary['kl']:.4f} "
            f"TV={summary['tv']:.4f} MAE={summary['target_pred_mae']:.6f} "
            f"cos={summary['prob_cos']:.4f} "
            f"peak(pred/target)={summary['pred_peak']:.4f}/{summary['target_peak']:.4f}",
            flush=True,
        )

    print_csv(rows)
    print_overall(overall.summary())


if __name__ == "__main__":
    main()
