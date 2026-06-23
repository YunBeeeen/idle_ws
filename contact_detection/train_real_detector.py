"""Train a small real-log contact detector from manually labeled CSV episodes.

This script is intentionally separate from the simulation pipeline.  It is a
real-robot sanity experiment: use manually labeled CSV logs, keep episode-wise
splits, fit the scaler only on train episodes, and select the checkpoint and
decision threshold on validation F1.
"""

from __future__ import annotations

import argparse
import copy
import glob
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contact_dataset import build_window_end_indices
from models import GRUDetector, MLPDetector
from utils import (
    StandardScaler,
    binary_classification_metrics,
    build_input_features,
    load_config,
    load_real_log_csv,
    normalize_feature_mode,
    save_binary_nc_pc_confusion_matrix_figure,
    save_config_yaml,
    save_json,
    search_threshold_with_policy,
    select_torch_device,
    set_global_seed,
)


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset

        return torch, DataLoader, Dataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for real detector training. "
            "Install contact_detection/requirements.txt first."
        ) from exc


@dataclass
class EpisodeRecord:
    path: Path
    kind: str
    episode_id: int
    time: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def original_mode_uses_delta(feature_mode: str) -> bool:
    """Return whether the feature mode should produce the legacy 42D vector."""

    return normalize_feature_mode(feature_mode) == "original_42"


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(str(Path(pattern).expanduser())))
        if expanded:
            paths.extend(Path(path).resolve() for path in expanded)
        else:
            path = Path(pattern).expanduser()
            if path.exists():
                paths.append(path.resolve())
    return sorted(dict.fromkeys(paths))


def load_episode(
    path: Path,
    *,
    kind: str,
    episode_id: int,
    config: dict,
    feature_mode: str,
    use_delta_features: bool,
) -> tuple[EpisodeRecord, list[str]]:
    real = load_real_log_csv(path, config)
    labels = np.asarray(real.get("contact_label", np.zeros(real["time"].shape[0])), dtype=np.int64).reshape(-1)
    if kind == "no_contact":
        labels = np.zeros_like(labels, dtype=np.int64)
    if labels.shape[0] != real["time"].shape[0]:
        raise ValueError(f"Label length mismatch in {path}: {labels.shape[0]} vs {real['time'].shape[0]}")
    episode_arr = np.full(real["time"].shape[0], int(episode_id), dtype=np.int64)
    features, names = build_input_features(
        q=real["q"],
        qdot=real["qdot"],
        q_des=real["q_des"],
        tau_cmd=real["tau_cmd"],
        use_delta_features=use_delta_features,
        episode_id=episode_arr,
        tau_residual=real.get("tau_residual"),
        tau_residual_corrected=real.get("tau_residual_corrected"),
        feature_mode=feature_mode,
    )
    return (
        EpisodeRecord(
            path=path,
            kind=kind,
            episode_id=int(episode_id),
            time=np.asarray(real["time"], dtype=np.float64),
            features=np.asarray(features, dtype=np.float64),
            labels=labels.astype(np.int64),
        ),
        names,
    )


def split_episodes(
    episodes: list[EpisodeRecord],
    *,
    val_no_contact_count: int,
    val_contact_count: int,
    split_policy: str = "stratified_by_condition",
) -> tuple[set[int], set[int]]:
    no_contact = [ep for ep in episodes if ep.kind == "no_contact"]
    contact = [ep for ep in episodes if ep.kind == "contact"]
    if not no_contact or not contact:
        raise ValueError("Need at least one no-contact and one contact episode.")

    if str(split_policy) == "stratified_by_condition":
        train_ids: set[int] = set()
        val_ids: set[int] = set()
        for group in (no_contact, contact):
            grouped: dict[str, list[EpisodeRecord]] = {}
            for ep in group:
                grouped.setdefault(condition_key(ep.path), []).append(ep)
            for group_eps in grouped.values():
                group_eps = sorted(group_eps, key=lambda ep: ep.path.name)
                if len(group_eps) == 1:
                    train_ids.add(group_eps[0].episode_id)
                    continue
                val_count = max(1, int(round(len(group_eps) * 0.2)))
                val_count = min(val_count, len(group_eps) - 1)
                for ep in group_eps[:-val_count]:
                    train_ids.add(ep.episode_id)
                for ep in group_eps[-val_count:]:
                    val_ids.add(ep.episode_id)
        if not train_ids or not val_ids:
            raise ValueError("Stratified split produced an empty train or validation set.")
        return train_ids, val_ids

    def split_group(group: list[EpisodeRecord], requested_val_count: int) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
        if len(group) == 1:
            return group, group
        val_count = max(1, min(int(requested_val_count), len(group) - 1))
        return group[:-val_count], group[-val_count:]

    train_no, val_no = split_group(no_contact, val_no_contact_count)
    train_pc, val_pc = split_group(contact, val_contact_count)
    train_ids = {ep.episode_id for ep in train_no + train_pc}
    val_ids = {ep.episode_id for ep in val_no + val_pc}
    return train_ids, val_ids


def condition_key(path: Path) -> str:
    """Return an episode condition key without the numeric suffix.

    This keeps validation episode-wise while preventing a whole condition such
    as ``contact_ee_slow_sine_j23`` from landing only in validation.
    """

    stem = path.stem
    stem = re.sub(r"_\d{3}$", "", stem)
    stem = stem.replace("_near_current", "")
    return stem


def concatenate_episodes(episodes: list[EpisodeRecord]) -> dict[str, np.ndarray]:
    features = np.concatenate([ep.features for ep in episodes], axis=0)
    labels = np.concatenate([ep.labels for ep in episodes], axis=0)
    episode_id = np.concatenate(
        [np.full(ep.labels.shape[0], ep.episode_id, dtype=np.int64) for ep in episodes],
        axis=0,
    )
    local_time = np.concatenate([ep.time - float(ep.time[0]) for ep in episodes], axis=0)
    return {"features": features, "labels": labels, "episode_id": episode_id, "time": local_time}


def make_torch_datasets(torch_module, DatasetBase):
    class WindowDataset(DatasetBase):
        def __init__(self, scaled_features: np.ndarray, labels: np.ndarray, end_indices: np.ndarray, window_length: int):
            self.features = np.asarray(scaled_features, dtype=np.float32)
            self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)
            self.end_indices = np.asarray(end_indices, dtype=np.int64).reshape(-1)
            self.window_length = int(window_length)

        def __len__(self) -> int:
            return int(self.end_indices.shape[0])

        def __getitem__(self, index: int):
            end = int(self.end_indices[index])
            start = end - self.window_length + 1
            return self.features[start : end + 1], np.float32(self.labels[end])

    class StepDataset(DatasetBase):
        def __init__(self, scaled_features: np.ndarray, labels: np.ndarray, end_indices: np.ndarray):
            self.features = np.asarray(scaled_features, dtype=np.float32)
            self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)
            self.end_indices = np.asarray(end_indices, dtype=np.int64).reshape(-1)

        def __len__(self) -> int:
            return int(self.end_indices.shape[0])

        def __getitem__(self, index: int):
            end = int(self.end_indices[index])
            return self.features[end], np.float32(self.labels[end])

    return WindowDataset, StepDataset


def dataloader_labels(labels: np.ndarray, end_indices: np.ndarray) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64).reshape(-1)[np.asarray(end_indices, dtype=np.int64)]


def transition_keep_mask(
    *,
    labels: np.ndarray,
    episode_id: np.ndarray,
    time: np.ndarray,
    end_indices: np.ndarray,
    exclusion_s: float,
) -> np.ndarray:
    """Return True for windows whose end index is away from label transitions.

    Manual real contact labels are coarse.  A human push is usually ramped in
    and out, while the CSV label changes at a single timestamp.  Excluding a
    small band around each 0/1 transition keeps those ambiguous samples from
    dominating checkpoint and threshold selection.
    """

    end_arr = np.asarray(end_indices, dtype=np.int64).reshape(-1)
    if float(exclusion_s) <= 0.0 or end_arr.size == 0:
        return np.ones(end_arr.shape[0], dtype=bool)

    label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    episode_arr = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    keep = np.ones(end_arr.shape[0], dtype=bool)

    for ep in np.unique(episode_arr):
        sample_indices = np.flatnonzero(episode_arr == int(ep))
        if sample_indices.size < 2:
            continue
        local_labels = label_arr[sample_indices]
        change_local = np.flatnonzero(local_labels[1:] != local_labels[:-1]) + 1
        if change_local.size == 0:
            continue
        same_episode_windows = episode_arr[end_arr] == int(ep)
        for local_idx in change_local:
            transition_time = float(time_arr[sample_indices[int(local_idx)]])
            near_transition = np.abs(time_arr[end_arr] - transition_time) <= float(exclusion_s)
            keep &= ~(same_episode_windows & near_transition)
    return keep


def search_real_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    num_points: int,
    selection_policy: str,
    target_recall: float,
    fbeta_beta: float,
    max_validation_fpr: float,
) -> dict:
    """Threshold search with an extra real-robot FPR-constrained policy."""

    policy = str(selection_policy)
    if policy != "fpr_constrained_f1":
        return search_threshold_with_policy(
            scores,
            labels,
            num_points=int(num_points),
            selection_policy=policy,
            target_recall=float(target_recall),
            fbeta_beta=float(fbeta_beta),
        )

    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if score_arr.shape != label_arr.shape:
        raise ValueError(f"Score/label shape mismatch: {score_arr.shape} vs {label_arr.shape}")
    if score_arr.size == 0:
        raise ValueError("Cannot search threshold on empty arrays")

    unique = np.unique(score_arr)
    if unique.size <= int(num_points):
        candidates = unique
    else:
        candidates = np.linspace(float(np.min(score_arr)), float(np.max(score_arr)), int(num_points))

    best_threshold = float(candidates[0])
    best_metrics = binary_classification_metrics(label_arr, score_arr >= best_threshold)
    best_key: tuple[float, ...] | None = None
    fallback_threshold = best_threshold
    fallback_metrics = best_metrics
    fallback_key: tuple[float, ...] | None = None
    target_satisfied = False

    for threshold in candidates:
        metrics = binary_classification_metrics(label_arr, score_arr >= float(threshold))
        fpr = float(metrics["false_positive_rate"])
        key = (
            float(metrics["f1"]),
            float(metrics["recall"]),
            float(metrics["precision"]),
            -float(threshold),
        )
        fallback_candidate_key = (
            -fpr,
            float(metrics["f1"]),
            float(metrics["recall"]),
            float(metrics["precision"]),
            -float(threshold),
        )
        if fallback_key is None or fallback_candidate_key > fallback_key:
            fallback_key = fallback_candidate_key
            fallback_threshold = float(threshold)
            fallback_metrics = metrics
        if fpr > float(max_validation_fpr) + 1.0e-12:
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
            target_satisfied = True

    if not target_satisfied:
        best_threshold = float(fallback_threshold)
        best_metrics = fallback_metrics

    return {
        "threshold": float(best_threshold),
        "metrics": dict(best_metrics),
        "selection_policy": policy,
        "selection_score": float(best_metrics["f1"]),
        "target_recall": float(target_recall),
        "target_recall_satisfied": True,
        "fbeta_beta": float(fbeta_beta),
        "max_validation_fpr": float(max_validation_fpr),
        "max_validation_fpr_satisfied": bool(target_satisfied),
    }


def evaluate_model(model, loader, criterion, torch_module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    device = next(model.parameters()).device
    with torch_module.no_grad():
        for features, labels in loader:
            features = features.to(device=device, dtype=torch_module.float32)
            labels = labels.to(device=device, dtype=torch_module.float32)
            logits = model(features)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            probs.append(torch_module.sigmoid(logits).cpu().numpy())
            labels_out.append(labels.cpu().numpy())
    return (
        float(np.mean(losses)) if losses else 0.0,
        np.concatenate(probs, axis=0) if probs else np.zeros(0, dtype=np.float64),
        np.concatenate(labels_out, axis=0) if labels_out else np.zeros(0, dtype=np.float64),
    )


def train_one_model(
    *,
    model_name: str,
    model,
    train_loader,
    val_loader,
    pos_weight: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    threshold_points: int,
    threshold_selection_policy: str,
    target_recall: float,
    fbeta_beta: float,
    max_validation_fpr: float,
    device,
    torch_module,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    model.to(device)
    criterion = torch_module.nn.BCEWithLogitsLoss(
        pos_weight=torch_module.tensor([float(pos_weight)], dtype=torch_module.float32, device=device)
    )
    optimizer = torch_module.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    history: list[dict] = []
    best_state = None
    best_payload: dict | None = None
    best_key: tuple[float, float, int] | None = None
    best_val_probs = np.zeros(0, dtype=np.float64)
    best_val_labels = np.zeros(0, dtype=np.float64)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for features, labels in train_loader:
            features = features.to(device=device, dtype=torch_module.float32)
            labels = labels.to(device=device, dtype=torch_module.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss, val_probs, val_labels = evaluate_model(model, val_loader, criterion, torch_module)
        threshold_search = search_real_threshold(
            val_probs,
            val_labels,
            num_points=int(threshold_points),
            selection_policy=str(threshold_selection_policy),
            target_recall=float(target_recall),
            fbeta_beta=float(fbeta_beta),
            max_validation_fpr=float(max_validation_fpr),
        )
        val_metrics = dict(threshold_search["metrics"])
        threshold = float(threshold_search["threshold"])
        row = {
            "model": model_name,
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_false_positive_rate": float(val_metrics["false_positive_rate"]),
            "val_false_negative_rate": float(val_metrics["false_negative_rate"]),
            "val_threshold": threshold,
            "threshold_selection_policy": str(threshold_search["selection_policy"]),
            "max_validation_fpr": float(max_validation_fpr),
        }
        history.append(row)

        # Primary: highest validation F1. Tie-breaker: lower validation loss.
        key = (float(val_metrics["f1"]), -float(val_loss), -int(epoch))
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_payload = {
                "best_epoch": int(epoch),
                "best_val_loss": val_loss,
                "best_val_precision": float(val_metrics["precision"]),
                "best_val_recall": float(val_metrics["recall"]),
                "best_val_f1": float(val_metrics["f1"]),
                "decision_threshold": threshold,
                "validation_selected_threshold": threshold,
                "threshold_selection_metric": f"val_{threshold_search['selection_policy']}_selection_label",
                "threshold_selected_on": "selection_label_validation",
                "threshold_selection_policy": str(threshold_search["selection_policy"]),
                "max_validation_fpr": float(max_validation_fpr),
                "max_validation_fpr_satisfied": bool(threshold_search.get("max_validation_fpr_satisfied", True)),
            }
            best_val_probs = val_probs.copy()
            best_val_labels = val_labels.copy()

        print(
            f"[{model_name}] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.3f} "
            f"val_p={val_metrics['precision']:.3f} val_r={val_metrics['recall']:.3f} th={threshold:.3f}"
        )

    if best_state is None or best_payload is None:
        raise RuntimeError(f"No checkpoint was selected for {model_name}.")
    model.load_state_dict(best_state)
    best_payload["state_dict"] = best_state
    return best_payload, history, best_val_probs, best_val_labels


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, {"history": history, "best_epoch": max(history, key=lambda row: row["val_f1"])["epoch"]})


def validation_prediction_csv(path: Path, probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("row,label,probability,prediction\n")
        for idx, (label, prob) in enumerate(zip(labels.astype(int), probabilities.astype(float))):
            stream.write(f"{idx},{int(label)},{float(prob):.8f},{int(prob >= threshold)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="contact_detection/config_legacy_gru_mlp.yaml")
    parser.add_argument(
        "--no-contact-glob",
        action="append",
        default=["contact_detection/real_logs/no_contact/20260619/no_contact_home_bridge_hold_*.csv"],
    )
    parser.add_argument(
        "--contact-glob",
        action="append",
        default=["contact_detection/real_logs/contact/20260619/contact_ee_home_bridge_hold_*.csv"],
    )
    parser.add_argument("--output-dir", default="contact_detection/outputs_real/home_bridge_hold_residual_cmd_v1")
    parser.add_argument("--feature-mode", default="residual_cmd_v1")
    parser.add_argument("--model", choices=["gru", "mlp", "both"], default="both")
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Optional GRU checkpoint used to initialize real fine-tuning, e.g. a simulation-trained original_42 model.",
    )
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-no-contact-count", type=int, default=1)
    parser.add_argument("--val-contact-count", type=int, default=3)
    parser.add_argument(
        "--split-policy",
        choices=["stratified_by_condition", "tail"],
        default="stratified_by_condition",
    )
    parser.add_argument("--threshold-search-points", type=int, default=200)
    parser.add_argument(
        "--threshold-selection-policy",
        choices=["f1", "f2", "recall_constrained_f1", "fpr_constrained_f1"],
        default="f1",
        help="Validation threshold policy. fpr_constrained_f1 is useful for real trigger sanity checks.",
    )
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--fbeta-beta", type=float, default=2.0)
    parser.add_argument(
        "--max-validation-fpr",
        type=float,
        default=0.05,
        help="Maximum validation false-positive rate for --threshold-selection-policy fpr_constrained_f1.",
    )
    parser.add_argument(
        "--transition-exclusion-s",
        type=float,
        default=0.0,
        help="Exclude windows whose end sample is within this many seconds of a manual label transition.",
    )
    parser.add_argument(
        "--exclude-transition-val",
        action="store_true",
        help="Also exclude transition windows from validation threshold/checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--residual-offset-duration", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pretrained_checkpoint and str(args.model) == "mlp":
        raise ValueError("--pretrained-checkpoint is only supported for GRU fine-tuning. Use --model gru or --model both.")
    set_global_seed(int(args.seed))
    random.seed(int(args.seed))

    config = load_config(args.config)
    config.setdefault("real_inference", {})
    config["real_inference"]["resample_to_target_dt"] = False
    config["real_inference"]["residual_offset_duration_s"] = float(args.residual_offset_duration)
    config.setdefault("dataset", {})
    config["dataset"]["feature_mode"] = str(args.feature_mode)
    config["dataset"]["window_length"] = int(args.window_length)
    config["dataset"]["stride"] = int(args.stride)
    use_delta_features = original_mode_uses_delta(str(args.feature_mode))
    config["dataset"]["use_delta_features"] = bool(use_delta_features)

    no_paths = expand_paths(args.no_contact_glob)
    contact_paths = expand_paths(args.contact_glob)
    if not no_paths:
        raise FileNotFoundError(f"No no-contact CSVs matched: {args.no_contact_glob}")
    if not contact_paths:
        raise FileNotFoundError(f"No contact CSVs matched: {args.contact_glob}")

    episodes: list[EpisodeRecord] = []
    feature_names: list[str] | None = None
    for idx, path in enumerate(no_paths):
        episode, names = load_episode(
            path,
            kind="no_contact",
            episode_id=idx,
            config=config,
            feature_mode=str(args.feature_mode),
            use_delta_features=use_delta_features,
        )
        episodes.append(episode)
        feature_names = names
    offset = len(no_paths)
    for local_idx, path in enumerate(contact_paths):
        episode, names = load_episode(
            path,
            kind="contact",
            episode_id=offset + local_idx,
            config=config,
            feature_mode=str(args.feature_mode),
            use_delta_features=use_delta_features,
        )
        if int(np.sum(episode.labels == 1)) == 0:
            raise ValueError(f"Contact CSV has no positive contact_label rows: {path}")
        episodes.append(episode)
        feature_names = names
    if feature_names is None:
        raise RuntimeError("No features were loaded.")

    train_ids, val_ids = split_episodes(
        episodes,
        val_no_contact_count=int(args.val_no_contact_count),
        val_contact_count=int(args.val_contact_count),
        split_policy=str(args.split_policy),
    )
    data = concatenate_episodes(episodes)
    features = data["features"]
    labels = data["labels"]
    episode_id = data["episode_id"]
    all_end_indices = build_window_end_indices(episode_id, int(args.window_length), int(args.stride))
    train_mask = np.isin(episode_id, sorted(train_ids))
    val_mask = np.isin(episode_id, sorted(val_ids))
    train_end = all_end_indices[np.isin(episode_id[all_end_indices], sorted(train_ids))]
    val_end = all_end_indices[np.isin(episode_id[all_end_indices], sorted(val_ids))]
    if train_end.size == 0 or val_end.size == 0:
        raise ValueError("Train/validation windows are empty. Check episode lengths and split counts.")

    original_train_windows = int(train_end.size)
    original_val_windows = int(val_end.size)
    if float(args.transition_exclusion_s) > 0.0:
        train_keep = transition_keep_mask(
            labels=labels,
            episode_id=episode_id,
            time=data["time"],
            end_indices=train_end,
            exclusion_s=float(args.transition_exclusion_s),
        )
        train_end = train_end[train_keep]
        if bool(args.exclude_transition_val):
            val_keep = transition_keep_mask(
                labels=labels,
                episode_id=episode_id,
                time=data["time"],
                end_indices=val_end,
                exclusion_s=float(args.transition_exclusion_s),
            )
            val_end = val_end[val_keep]
        if train_end.size == 0 or val_end.size == 0:
            raise ValueError(
                "Transition exclusion removed all train or validation windows. "
                "Reduce --transition-exclusion-s or disable --exclude-transition-val."
            )

    scaler = StandardScaler.fit(features[train_mask])
    scaled = scaler.transform(features).astype(np.float32)

    torch, DataLoader, DatasetBase = _import_torch()
    device = select_torch_device(torch, str(args.device))
    WindowDataset, StepDataset = make_torch_datasets(torch, DatasetBase)
    pretrained_checkpoint: dict | None = None
    pretrained_path: Path | None = None
    if args.pretrained_checkpoint:
        pretrained_path = Path(str(args.pretrained_checkpoint)).expanduser().resolve()
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        pretrained_checkpoint = torch.load(pretrained_path, map_location="cpu")
        pretrained_input_dim = int(pretrained_checkpoint.get("input_dim", -1))
        if pretrained_input_dim != int(features.shape[1]):
            raise ValueError(
                "Pretrained checkpoint input_dim does not match real feature dimension: "
                f"{pretrained_input_dim} vs {features.shape[1]}. "
                "For sim-to-real fine-tuning, use the same feature mode, usually --feature-mode original_42."
            )
        pretrained_window = int(pretrained_checkpoint.get("window_length", int(args.window_length)))
        if pretrained_window != int(args.window_length):
            raise ValueError(
                "Pretrained checkpoint window_length does not match --window-length: "
                f"{pretrained_window} vs {args.window_length}."
            )
    train_labels = dataloader_labels(labels, train_end)
    neg = int(np.sum(train_labels == 0))
    pos = int(np.sum(train_labels == 1))
    pos_weight = float(neg / max(pos, 1))

    out_root = Path(args.output_dir).expanduser().resolve()
    models_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    figures_dir = out_root / "figures"
    for directory in (models_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    scaler.save(models_dir / "scaler.pkl")
    save_config_yaml(out_root / "experiment_config_used.yaml", config)

    metadata = {
        "experiment": "real_home_bridge_hold_sanity",
        "output_dir": str(out_root),
        "feature_mode": str(args.feature_mode),
        "feature_names": feature_names,
        "use_delta_features": bool(use_delta_features),
        "pretrained_checkpoint": str(pretrained_path) if pretrained_path is not None else None,
        "pretrain_source": "simulation_gru" if pretrained_path is not None else None,
        "fine_tune_target": "manual_real_robot_logs" if pretrained_path is not None else None,
        "tau_ext_input_policy": "label_only_never_feature",
        "residual_definition": "tau_residual = tau_meas - tau_cmd",
        "residual_expected_torque_mode": "command_total",
        "residual_offset_policy": "episode_initial_no_contact_mean",
        "window_length": int(args.window_length),
        "stride": int(args.stride),
        "split_policy": str(args.split_policy),
        "train_episode_ids": sorted(int(x) for x in train_ids),
        "val_episode_ids": sorted(int(x) for x in val_ids),
        "no_contact_csvs": [str(path) for path in no_paths],
        "contact_csvs": [str(path) for path in contact_paths],
        "train_windows": int(train_end.size),
        "val_windows": int(val_end.size),
        "train_windows_before_transition_exclusion": original_train_windows,
        "val_windows_before_transition_exclusion": original_val_windows,
        "transition_exclusion_s": float(args.transition_exclusion_s),
        "exclude_transition_val": bool(args.exclude_transition_val),
        "train_positive_windows": int(np.sum(train_labels == 1)),
        "train_negative_windows": int(np.sum(train_labels == 0)),
        "pos_weight": pos_weight,
        "selection_basis": f"validation_{args.threshold_selection_policy}",
        "threshold_selection_policy": str(args.threshold_selection_policy),
        "target_recall": float(args.target_recall),
        "fbeta_beta": float(args.fbeta_beta),
        "max_validation_fpr": float(args.max_validation_fpr),
        "test_set_usage": "not_used; this is validation-only real sanity training",
    }

    results: dict[str, dict] = {}
    requested_models = ["gru", "mlp"] if args.model == "both" else [str(args.model)]
    for model_name in requested_models:
        torch.manual_seed(int(args.seed) + (0 if model_name == "gru" else 1000))
        if model_name == "gru":
            train_dataset = WindowDataset(scaled, labels, train_end, int(args.window_length))
            val_dataset = WindowDataset(scaled, labels, val_end, int(args.window_length))
            gru_hidden_dim = int(args.hidden_dim)
            gru_num_layers = int(args.num_layers)
            gru_dropout = float(args.dropout)
            gru_bidirectional = False
            if pretrained_checkpoint is not None:
                gru_hidden_dim = int(pretrained_checkpoint.get("hidden_dim", gru_hidden_dim))
                gru_num_layers = int(pretrained_checkpoint.get("num_layers", gru_num_layers))
                gru_dropout = float(pretrained_checkpoint.get("dropout", gru_dropout))
                gru_bidirectional = bool(pretrained_checkpoint.get("bidirectional", False))
            model = GRUDetector(
                input_dim=int(features.shape[1]),
                hidden_dim=gru_hidden_dim,
                num_layers=gru_num_layers,
                dropout=gru_dropout,
                bidirectional=gru_bidirectional,
            )
            if pretrained_checkpoint is not None:
                state_dict = pretrained_checkpoint.get("state_dict")
                if not isinstance(state_dict, dict):
                    raise KeyError(f"Pretrained checkpoint has no state_dict: {pretrained_path}")
                model.load_state_dict(state_dict, strict=True)
            checkpoint_name = "gru_detector.pt"
            model_hidden_dim = gru_hidden_dim
            model_num_layers = gru_num_layers
            model_dropout = gru_dropout
            model_bidirectional = gru_bidirectional
        else:
            train_dataset = StepDataset(scaled, labels, train_end)
            val_dataset = StepDataset(scaled, labels, val_end)
            model = MLPDetector(
                input_dim=int(features.shape[1]),
                hidden_dim=int(args.hidden_dim),
                num_layers=2,
                dropout=float(args.dropout),
            )
            checkpoint_name = "mlp_detector.pt"
            model_hidden_dim = int(args.hidden_dim)
            model_num_layers = 2
            model_dropout = float(args.dropout)
            model_bidirectional = False

        train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False)
        best, history, val_probs, val_labels = train_one_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            threshold_points=int(args.threshold_search_points),
            threshold_selection_policy=str(args.threshold_selection_policy),
            target_recall=float(args.target_recall),
            fbeta_beta=float(args.fbeta_beta),
            max_validation_fpr=float(args.max_validation_fpr),
            device=device,
            torch_module=torch,
        )
        threshold = float(best["decision_threshold"])
        predictions = (val_probs >= threshold).astype(np.int64)
        val_metrics = binary_classification_metrics(val_labels, predictions)
        confusion = np.array([[val_metrics["tn"], val_metrics["fp"]], [val_metrics["fn"], val_metrics["tp"]]], dtype=int)
        validation_prediction_csv(metrics_dir / f"{model_name}_validation_predictions.csv", val_probs, val_labels, threshold)
        save_binary_nc_pc_confusion_matrix_figure(
            figures_dir / f"confusion_matrix_{model_name}.png",
            confusion,
            f"Real validation nc/pc ({model_name.upper()})",
            val_metrics,
        )
        checkpoint = {
            **best,
            "model_type": "binary",
            "model_arch": model_name,
            "input_dim": int(features.shape[1]),
            "hidden_dim": int(model_hidden_dim),
            "num_layers": int(model_num_layers),
            "dropout": float(model_dropout),
            "bidirectional": bool(model_bidirectional),
            "window_length": int(args.window_length),
            "stride": int(args.stride),
            "use_delta_features": bool(use_delta_features),
            "feature_mode": str(args.feature_mode),
            "feature_names": feature_names,
            "pretrained_checkpoint": str(pretrained_path) if (model_name == "gru" and pretrained_path is not None) else None,
            "pretrained_best_epoch": int(pretrained_checkpoint.get("best_epoch", -1)) if (model_name == "gru" and pretrained_checkpoint is not None) else None,
            "pretrained_best_val_f1": float(pretrained_checkpoint.get("best_val_f1", np.nan)) if (model_name == "gru" and pretrained_checkpoint is not None) else None,
            "pretrained_decision_threshold": float(pretrained_checkpoint.get("decision_threshold", np.nan)) if (model_name == "gru" and pretrained_checkpoint is not None) else None,
            "fine_tune_source": "simulation_gru_pretrained" if (model_name == "gru" and pretrained_path is not None) else "scratch",
            "best_val_f1_selection_label": float(best["best_val_f1"]),
            "best_val_loss_selection_label": float(best["best_val_loss"]),
            "best_val_precision_selection_label": float(best["best_val_precision"]),
            "best_val_recall_selection_label": float(best["best_val_recall"]),
            "decision_threshold_value": threshold,
            "label_basis_for_training": "manual_real_contact_label",
            "label_basis_for_checkpoint_selection": "manual_real_contact_label_validation",
            "label_basis_for_threshold_search": "manual_real_contact_label_validation",
            "label_basis_for_original_evaluation": "manual_real_contact_label",
            "threshold_applied_to_original_label_metrics": True,
            "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
            "real_threshold_selection_policy": str(args.threshold_selection_policy),
            "target_recall": float(args.target_recall),
            "fbeta_beta": float(args.fbeta_beta),
            "max_validation_fpr": float(args.max_validation_fpr),
            "transition_exclusion_s": float(args.transition_exclusion_s),
            "exclude_transition_val": bool(args.exclude_transition_val),
            "tau_ext_input_policy": "label_only_never_feature",
            "residual_definition": "tau_residual = tau_meas - tau_cmd",
            "residual_expected_torque_mode": "command_total",
            "residual_offset_policy": "episode_initial_no_contact_mean",
        }
        torch.save(checkpoint, models_dir / checkpoint_name)
        write_history(metrics_dir / ("train_log.json" if model_name == "gru" else "mlp_train_log.json"), history)
        results[model_name] = {
            "checkpoint": str(models_dir / checkpoint_name),
            "best_epoch": int(best["best_epoch"]),
            "decision_threshold": threshold,
            "validation_metrics": val_metrics,
            "confusion_matrix": confusion.tolist(),
        }

    summary = {**metadata, "models": results}
    save_json(metrics_dir / "real_train_summary.json", summary)
    print(f"Saved real detector summary: {metrics_dir / 'real_train_summary.json'}")
    for name, result in results.items():
        m = result["validation_metrics"]
        print(
            f"{name.upper()} val: precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"f1={m['f1']:.3f} fpr={m['false_positive_rate']:.3f} "
            f"threshold={result['decision_threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
