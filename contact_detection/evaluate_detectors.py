"""Evaluate threshold, MLP, and GRU contact detectors on the simulation test split.

논문/후속 실험 흐름에서 이 파일은 "시뮬레이션에서는 정답 label이 있으므로
정량 평가가 가능하다"는 단계에 해당한다. test split은 train/val과 다른
episode_id를 가지므로 같은 시계열 조각이 섞이는 leakage를 피한다.

저장하는 결과:
- Threshold / MLP / GRU 전체 test Precision, Recall, F1-score, confusion counts
- hold / slow_sine mode split metrics
- 동일 가중 평균 mode metrics
- low-torque / torque-bin analysis metrics
- confusion matrix, metric bar, prediction example, PR curve, threshold tradeoff

주의:
- tau_ext는 label/diagnosis/analysis 확인에만 쓰이며 모델 입력으로 쓰지 않는다.
- MLP는 GRU와 같은 end_indices의 single-step feature만 사용한다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from contact_dataset import ContactWindowDataset, transition_exclusion_sample_mask
from utils import (
    StandardScaler,
    apply_stage_config,
    binary_classification_metrics,
    compute_detection_delay,
    compute_episodewise_delta,
    ensure_output_dirs,
    iter_episode_slices,
    load_config,
    load_json,
    load_npz_dataset,
    output_root,
    save_config_yaml,
    save_binary_nc_pc_confusion_matrix_figure,
    save_json,
    save_metric_bar_figure,
    save_nc_pc_confusion_comparison_figure,
    save_precision_recall_curve_figure,
    save_sim_prediction_example,
    save_sim_prediction_examples,
    save_threshold_tradeoff_figure,
    select_torch_device,
    sigmoid,
    threshold_score_from_data,
)


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for evaluation. Install the dependencies from "
            "contact_detection/requirements.txt before running evaluate_detectors.py."
        ) from exc


def run_model_inference(model, loader, torch_module) -> np.ndarray:
    """Run test-set inference and return P(contact)."""
    model.eval()
    logits_list: list[np.ndarray] = []
    device = next(model.parameters()).device
    with torch_module.no_grad():
        for batch in loader:
            features = batch[0].to(device=device, dtype=torch_module.float32)
            logits = model(features)
            logits_list.append(logits.cpu().numpy())
    logits = np.concatenate(logits_list, axis=0) if logits_list else np.zeros(0, dtype=np.float64)
    return sigmoid(logits)


def binary_log_loss_from_probability(labels: np.ndarray, probability: np.ndarray) -> float:
    """Unweighted BCE loss for reporting original-label metrics."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"labels/probability shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        return 0.0
    p = np.clip(p, 1.0e-7, 1.0 - 1.0e-7)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def canonical_selection_label_basis(raw_basis: str | None, label_delay_ms: float = 0.0) -> str:
    """Normalize older artifact label-basis names for selection labels."""
    if float(label_delay_ms or 0.0) > 0.0:
        return "delayed_training_label"
    basis = str(raw_basis or "").strip()
    if basis in {"", "original_external_force_command_label", "original_label"}:
        return "original_command_label"
    return basis


def build_model_from_checkpoint(checkpoint: dict):
    from models import GRUDetector, MLPDetector

    model_arch = str(checkpoint.get("model_arch", "gru")).strip().lower()
    if model_arch == "gru":
        model = GRUDetector(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint["dropout"]),
            bidirectional=bool(checkpoint.get("bidirectional", False)),
        )
    elif model_arch == "mlp":
        model = MLPDetector(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_arch={model_arch!r}")
    model.load_state_dict(checkpoint["state_dict"])
    return model


def attach_checkpoint_metadata(metrics: dict, checkpoint: dict, model_path) -> None:
    label_delay_ms = float(checkpoint.get("label_delay_ms", 0.0) or 0.0)
    threshold_basis = canonical_selection_label_basis(
        checkpoint.get("label_basis_for_threshold_search"),
        label_delay_ms,
    )
    selection_basis = canonical_selection_label_basis(
        checkpoint.get("selection_label_basis", threshold_basis),
        label_delay_ms,
    )
    metrics["checkpoint_path"] = str(model_path)
    metrics["feature_mode"] = checkpoint.get("feature_mode", "original_42")
    metrics["feature_names"] = checkpoint.get("feature_names", [])
    metrics["residual_expected_torque_mode"] = checkpoint.get("residual_expected_torque_mode", "command_total")
    metrics["residual_offset_policy"] = checkpoint.get("residual_offset_policy", "episode_initial_no_contact_mean")
    metrics["tau_ext_input_policy"] = checkpoint.get("tau_ext_input_policy", "label_only_never_feature")
    metrics["label_delay_ms"] = label_delay_ms
    metrics["transition_exclusion_ms"] = float(checkpoint.get("transition_exclusion_ms", 0.0) or 0.0)
    metrics["checkpoint_selection_metric"] = checkpoint.get("checkpoint_selection_metric")
    metrics["decision_threshold_value"] = float(
        checkpoint.get("decision_threshold_value", checkpoint.get("decision_threshold", metrics.get("decision_threshold", 0.5)))
    )
    metrics["threshold_selection_metric"] = checkpoint.get("threshold_selection_metric", "val_f1_selection_label")
    metrics["threshold_selected_on"] = checkpoint.get("threshold_selected_on", "selection_label_validation")
    metrics["label_basis_for_threshold_search"] = threshold_basis
    metrics["label_basis_for_original_evaluation"] = checkpoint.get(
        "label_basis_for_original_evaluation",
        "original_external_force_command_label",
    )
    metrics["selection_label_basis"] = selection_basis
    metrics["label_basis_for_training"] = checkpoint.get(
        "label_basis_for_training",
        metrics["selection_label_basis"],
    )
    metrics["label_basis_for_training"] = canonical_selection_label_basis(
        metrics["label_basis_for_training"],
        label_delay_ms,
    )
    metrics["label_basis_for_checkpoint_selection"] = checkpoint.get(
        "label_basis_for_checkpoint_selection",
        metrics["selection_label_basis"],
    )
    metrics["label_basis_for_checkpoint_selection"] = canonical_selection_label_basis(
        metrics["label_basis_for_checkpoint_selection"],
        label_delay_ms,
    )
    metrics["threshold_applied_to_original_label_metrics"] = bool(
        checkpoint.get("threshold_applied_to_original_label_metrics", True)
    )
    metrics["original_label_metric_threshold_policy"] = checkpoint.get(
        "original_label_metric_threshold_policy",
        "reuse_selection_label_threshold",
    )
    metrics["checkpoint_best_epoch"] = int(
        checkpoint.get("best_epoch", checkpoint.get("source_checkpoint_best_epoch", -1))
    )
    metrics["checkpoint_best_epoch_selection_label"] = int(
        checkpoint.get(
            "best_epoch_selection_label",
            checkpoint.get("best_epoch", checkpoint.get("source_checkpoint_best_epoch", -1)),
        )
    )
    if "best_val_f1" in checkpoint:
        metrics["checkpoint_best_val_f1"] = float(checkpoint["best_val_f1"])
        metrics["checkpoint_best_val_f1_selection_label"] = float(
            checkpoint.get("best_val_f1_selection_label", checkpoint["best_val_f1"])
        )
    if "best_val_loss" in checkpoint:
        metrics["checkpoint_best_val_loss"] = float(checkpoint["best_val_loss"])
        metrics["checkpoint_best_val_loss_selection_label"] = float(
            checkpoint.get("best_val_loss_selection_label", checkpoint["best_val_loss"])
        )
    if "best_val_precision" in checkpoint:
        metrics["checkpoint_best_val_precision_selection_label"] = float(
            checkpoint.get("best_val_precision_selection_label", checkpoint["best_val_precision"])
        )
    if "best_val_recall" in checkpoint:
        metrics["checkpoint_best_val_recall_selection_label"] = float(
            checkpoint.get("best_val_recall_selection_label", checkpoint["best_val_recall"])
        )
    if "source_checkpoint_best_epoch" in checkpoint:
        metrics["source_checkpoint_best_epoch"] = int(checkpoint["source_checkpoint_best_epoch"])
    if "source_best_val_f1" in checkpoint:
        metrics["source_best_val_f1"] = float(checkpoint["source_best_val_f1"])
    if "trainval_epochs" in checkpoint:
        metrics["trainval_epochs"] = int(checkpoint["trainval_epochs"])


def ms_to_steps(ms: float, dt: float) -> int:
    return max(0, int(round(float(ms) / 1000.0 / float(dt))))


def default_ablation_tag(label_delay_ms: float, transition_exclusion_ms: float) -> str:
    parts: list[str] = []
    if float(label_delay_ms) > 0.0:
        parts.append(f"label_delay_{float(label_delay_ms):g}ms")
    if float(transition_exclusion_ms) > 0.0:
        parts.append(f"transition_exclusion_{float(transition_exclusion_ms):g}ms")
    return "__".join(parts) if parts else "baseline"


def resolve_artifact_dirs(config: dict, ablation_tag: str | None) -> tuple[dict[str, Path], dict[str, Path]]:
    base_dirs = ensure_output_dirs(output_root(config))
    if ablation_tag:
        artifact_dirs = ensure_output_dirs(base_dirs["root"] / "ablations" / str(ablation_tag))
        return base_dirs, artifact_dirs
    return base_dirs, base_dirs


def metrics_with_delay(
    labels: np.ndarray,
    predictions: np.ndarray,
    time: np.ndarray,
    episode_id: np.ndarray,
) -> dict[str, float | int | None]:
    metrics = binary_classification_metrics(labels, predictions)
    metrics["detection_delay"] = compute_detection_delay(time, labels, predictions, episode_id=episode_id)
    return metrics


def confusion_matrix_from_metrics(metrics: dict) -> np.ndarray:
    return np.asarray(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ],
        dtype=np.int64,
    )


def row_normalize_confusion_dict(matrix: np.ndarray) -> list[list[float]]:
    values = np.asarray(matrix, dtype=np.float64)
    row_sum = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0.0)
    return normalized.tolist()


def poster_confusion_model_payload(model_name: str, metrics: dict) -> dict:
    matrix = confusion_matrix_from_metrics(metrics)
    return {
        "model": model_name,
        "label_order": ["nc", "pc"],
        "layout": {
            "rows": "true_label",
            "columns": "predicted_label",
            "matrix": "[[TN, FP], [FN, TP]]",
        },
        "raw_count": {
            "tn": int(metrics["tn"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tp": int(metrics["tp"]),
            "matrix": matrix.astype(int).tolist(),
        },
        "row_normalized": {
            "matrix": row_normalize_confusion_dict(matrix),
            "true_nc": {
                "predicted_nc": float(matrix[0, 0] / max(int(matrix[0, :].sum()), 1)),
                "predicted_pc_false_positive_rate": float(matrix[0, 1] / max(int(matrix[0, :].sum()), 1)),
            },
            "true_pc": {
                "predicted_nc_false_negative_rate": float(matrix[1, 0] / max(int(matrix[1, :].sum()), 1)),
                "predicted_pc": float(matrix[1, 1] / max(int(matrix[1, :].sum()), 1)),
            },
        },
        "metric_summary": {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "accuracy": float(metrics["accuracy"]),
            "false_positive_rate": float(metrics["false_positive_rate"]),
            "false_negative_rate": float(metrics["false_negative_rate"]),
            "decision_threshold": metrics.get("decision_threshold"),
        },
        "checkpoint_path": metrics.get("checkpoint_path"),
    }


def build_poster_confusion_summary(stage: str, model_variant: str, metrics_by_model: dict[str, dict]) -> dict:
    return {
        "stage": stage,
        "model_variant": model_variant,
        "purpose": "poster_nc_pc_confusion_matrix",
        "label_mapping": {
            "0": "nc",
            "1": "pc",
            "nc": "no contact",
            "pc": "physical contact",
        },
        "matrix_layout": {
            "rows": "True label",
            "columns": "Predicted label",
            "matrix": "[[TN, FP], [FN, TP]]",
            "display": [["True nc / Pred nc", "True nc / Pred pc"], ["True pc / Pred nc", "True pc / Pred pc"]],
        },
        "test_set_usage": "test split is used only for final poster evaluation, not checkpoint/threshold/model selection",
        "used_for_selection": False,
        "sample_metric_label_basis": "original_external_force_command_label",
        "tau_ext_policy": "tau_ext is used only to generate labels and is never included in model inputs",
        "interpretation_note": (
            "Poster figures show both raw counts and row-normalized percentages so false positives "
            "and false negatives can be interpreted together with Precision/Recall/F1."
        ),
        "models": {
            name: poster_confusion_model_payload(name, metrics)
            for name, metrics in metrics_by_model.items()
        },
    }


def first_stable_detection(
    positive: np.ndarray,
    search_mask: np.ndarray,
    consecutive_samples: int,
) -> int | None:
    pos_arr = np.asarray(positive).astype(bool).reshape(-1)
    mask_arr = np.asarray(search_mask).astype(bool).reshape(-1)
    k = max(1, int(consecutive_samples))
    if pos_arr.shape != mask_arr.shape:
        raise ValueError(f"positive/search_mask shape mismatch: {pos_arr.shape} vs {mask_arr.shape}")
    if pos_arr.size < k:
        return None
    for idx in np.flatnonzero(mask_arr):
        end_idx = int(idx) + k
        if end_idx > pos_arr.size:
            break
        if np.all(mask_arr[idx:end_idx]) and np.all(pos_arr[idx:end_idx]):
            return int(idx)
    return None


def count_stable_false_alarm_segments(
    positive: np.ndarray,
    pure_mask: np.ndarray,
    consecutive_samples: int,
) -> int:
    pos_arr = np.asarray(positive).astype(bool).reshape(-1)
    pure_arr = np.asarray(pure_mask).astype(bool).reshape(-1)
    if pos_arr.shape != pure_arr.shape:
        raise ValueError(f"positive/pure_mask shape mismatch: {pos_arr.shape} vs {pure_arr.shape}")
    k = max(1, int(consecutive_samples))
    count = 0
    run_length = 0
    counted_current_run = False
    for is_positive, is_pure in zip(pos_arr, pure_arr):
        if is_pure and is_positive:
            run_length += 1
            if run_length >= k and not counted_current_run:
                count += 1
                counted_current_run = True
        else:
            run_length = 0
            counted_current_run = False
    return int(count)


def event_latency_metrics(
    data: dict[str, np.ndarray],
    end_indices: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    consecutive_samples: int,
    detection_margin_ms: float,
) -> dict:
    required = {"event_table_id", "event_table_start_index", "event_table_end_index"}
    if any(key not in data for key in required):
        return {
            "available": False,
            "note": "Dataset has no event table metadata. Regenerate datasets with the updated generator.",
        }

    full_time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    end_idx_arr = np.asarray(end_indices, dtype=np.int64).reshape(-1)
    prob_arr = np.asarray(probability, dtype=np.float64).reshape(-1)
    if prob_arr.shape[0] != end_idx_arr.shape[0]:
        raise ValueError(f"probability/end_indices length mismatch: {prob_arr.shape[0]} vs {end_idx_arr.shape[0]}")

    dt = float(np.median(np.diff(full_time[: min(full_time.size, 100)]))) if full_time.size > 1 else 0.0
    if not np.isfinite(dt) or dt <= 0.0:
        # time resets at episode boundaries; fall back to positive adjacent differences.
        diffs = np.diff(full_time)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        dt = float(np.median(diffs)) if diffs.size else 0.0
    margin_steps = ms_to_steps(float(detection_margin_ms), dt) if dt > 0.0 else 0
    positive = prob_arr >= float(threshold)
    event_ids = np.asarray(data["event_table_id"]).astype(np.int64)
    event_starts = np.asarray(data["event_table_start_index"]).astype(np.int64)
    event_ends = np.asarray(data["event_table_end_index"]).astype(np.int64)

    latencies_ms: list[float] = []
    missed = 0
    per_event: list[dict] = []
    excluded_for_false_alarm = np.zeros(end_idx_arr.shape[0], dtype=bool)
    for event_id, start_idx, end_idx in zip(event_ids, event_starts, event_ends):
        search_end = int(end_idx) + int(margin_steps)
        search_mask = (end_idx_arr >= int(start_idx)) & (end_idx_arr <= search_end)
        excluded_for_false_alarm |= (end_idx_arr >= max(0, int(start_idx) - int(margin_steps))) & (
            end_idx_arr <= search_end
        )
        detected_pos = first_stable_detection(positive, search_mask, consecutive_samples)
        if detected_pos is None:
            missed += 1
            per_event.append(
                {
                    "event_id": int(event_id),
                    "force_onset_index": int(start_idx),
                    "force_offset_index": int(end_idx),
                    "detected": False,
                    "latency_ms": None,
                }
            )
            continue
        latency_ms = float((full_time[end_idx_arr[detected_pos]] - full_time[int(start_idx)]) * 1000.0)
        latencies_ms.append(latency_ms)
        per_event.append(
            {
                "event_id": int(event_id),
                "force_onset_index": int(start_idx),
                "force_offset_index": int(end_idx),
                "detected": True,
                "detection_window_index": int(detected_pos),
                "detection_sample_index": int(end_idx_arr[detected_pos]),
                "latency_ms": latency_ms,
            }
        )

    pure_mask = ~excluded_for_false_alarm
    false_alarm_count = count_stable_false_alarm_segments(positive, pure_mask, consecutive_samples)
    pure_duration_s = float(np.sum(pure_mask) * dt) if dt > 0.0 else 0.0
    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    num_events = int(event_ids.size)
    return {
        "available": True,
        "threshold": float(threshold),
        "consecutive_samples": int(max(1, consecutive_samples)),
        "detection_margin_ms": float(detection_margin_ms),
        "detection_margin_steps": int(margin_steps),
        "number_of_contact_events": num_events,
        "number_of_detected_events": int(lat_arr.size),
        "number_of_missed_events": int(missed),
        "event_detection_rate": float(lat_arr.size / max(num_events, 1)),
        "latency_mean_ms": None if lat_arr.size == 0 else float(np.mean(lat_arr)),
        "latency_median_ms": None if lat_arr.size == 0 else float(np.median(lat_arr)),
        "latency_std_ms": None if lat_arr.size == 0 else float(np.std(lat_arr)),
        "latency_min_ms": None if lat_arr.size == 0 else float(np.min(lat_arr)),
        "latency_max_ms": None if lat_arr.size == 0 else float(np.max(lat_arr)),
        "false_alarm_count": int(false_alarm_count),
        "pure_no_contact_duration_s": pure_duration_s,
        "false_alarm_per_second": None if pure_duration_s <= 0.0 else float(false_alarm_count / pure_duration_s),
        "per_event": per_event,
    }


def threshold_sweep(
    probability: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    episode_id: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for threshold in thresholds:
        pred = (probability >= float(threshold)).astype(np.int64)
        metrics = binary_classification_metrics(labels, pred)
        delay = compute_detection_delay(time, labels, pred, episode_id=episode_id)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "accuracy": float(metrics["accuracy"]),
                "false_positive_rate": float(metrics["false_positive_rate"]),
                "false_negative_rate": float(metrics["false_negative_rate"]),
                "tp": int(metrics["tp"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "detected_events": int(delay["detected_events"]),
                "missed_events": int(delay["missed_events"]),
                "mean_delay_s": delay["mean_delay_s"],
            }
        )
    return rows


def choose_episode_examples(
    time: np.ndarray,
    labels: np.ndarray,
    threshold_pred: np.ndarray,
    mlp_prob: np.ndarray,
    gru_prob: np.ndarray,
    gru_pred: np.ndarray,
    episode_id: np.ndarray,
) -> list[dict[str, np.ndarray | str | float]]:
    stats: list[tuple[float, int, dict[str, float | int]]] = []
    for episode in np.unique(episode_id):
        mask = episode_id == episode
        if int(np.sum(labels[mask])) == 0:
            continue
        metrics = binary_classification_metrics(labels[mask], gru_pred[mask])
        stats.append((float(metrics["f1"]), int(episode), metrics))
    if not stats:
        return []

    stats.sort(key=lambda item: item[0])
    selected: list[tuple[str, tuple[float, int, dict[str, float | int]]]] = [
        ("Challenging / missed-case example", stats[0]),
        ("Representative median example", stats[len(stats) // 2]),
        ("Clean high-confidence example", stats[-1]),
    ]

    examples: list[dict[str, np.ndarray | str | float]] = []
    used: set[int] = set()
    for label_text, (_score, episode, metrics) in selected:
        if episode in used:
            continue
        used.add(episode)
        mask = episode_id == episode
        examples.append(
            {
                "title": (
                    f"{label_text} (episode {episode}, "
                    f"F1={float(metrics['f1']):.2f}, P={float(metrics['precision']):.2f}, "
                    f"R={float(metrics['recall']):.2f})"
                ),
                "time": time[mask],
                "label": labels[mask],
                "threshold_prediction": threshold_pred[mask],
                "mlp_probability": mlp_prob[mask],
                "gru_probability": gru_prob[mask],
            }
        )
    return examples


def infer_episode_modes(data: dict[str, np.ndarray]) -> np.ndarray:
    episode_id = np.asarray(data["episode_id"]).astype(np.int64)
    mode = np.empty(episode_id.shape[0], dtype="<U16")
    q_des = np.asarray(data["q_des"], dtype=np.float64)
    qdot_des = np.asarray(data["qdot_des"], dtype=np.float64)
    for start, end in iter_episode_slices(episode_id):
        is_hold = np.allclose(qdot_des[start:end], 0.0) and np.allclose(q_des[start:end], q_des[start])
        mode[start:end] = "hold" if is_hold else "slow_sine"
    return mode


def resolve_window_modes(data: dict[str, np.ndarray], end_indices: np.ndarray) -> tuple[np.ndarray, str]:
    if "trajectory_mode" in data:
        mode_full = np.asarray(data["trajectory_mode"]).astype("<U16")
        note = "Used stored trajectory_mode metadata from dataset generation."
    else:
        mode_full = infer_episode_modes(data)
        note = (
            "Dataset has no explicit trajectory_mode key; modes were inferred per episode from q_des/qdot_des. "
            "hold := constant q_des and zero qdot_des, slow_sine := time-varying q_des or nonzero qdot_des."
        )
    return mode_full[np.asarray(end_indices, dtype=np.int64)], note


def count_episode_modes(data: dict[str, np.ndarray]) -> dict[str, int]:
    if "trajectory_mode" in data:
        mode_full = np.asarray(data["trajectory_mode"]).astype("<U16")
    else:
        mode_full = infer_episode_modes(data)
    episode_id = np.asarray(data["episode_id"]).astype(np.int64)
    counts: dict[str, int] = {}
    for start, end in iter_episode_slices(episode_id):
        mode_name = str(mode_full[start])
        counts[mode_name] = counts.get(mode_name, 0) + 1
    return counts


def average_metric_dicts(metric_rows: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    if not metric_rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in metric_rows)))
    averaged: dict[str, float | int | None] = {}
    for key in keys:
        if key == "detection_delay":
            delay_means = [
                float(row["detection_delay"]["mean_delay_s"])
                for row in metric_rows
                if isinstance(row.get("detection_delay"), dict) and row["detection_delay"]["mean_delay_s"] is not None
            ]
            detected_events = [
                float(row["detection_delay"]["detected_events"])
                for row in metric_rows
                if isinstance(row.get("detection_delay"), dict)
            ]
            missed_events = [
                float(row["detection_delay"]["missed_events"])
                for row in metric_rows
                if isinstance(row.get("detection_delay"), dict)
            ]
            num_events = [
                float(row["detection_delay"]["num_events"])
                for row in metric_rows
                if isinstance(row.get("detection_delay"), dict)
            ]
            averaged[key] = {
                "num_events": float(np.mean(num_events)) if num_events else 0.0,
                "detected_events": float(np.mean(detected_events)) if detected_events else 0.0,
                "missed_events": float(np.mean(missed_events)) if missed_events else 0.0,
                "mean_delay_s": float(np.mean(delay_means)) if delay_means else None,
            }
            continue
        numeric_values = [row[key] for row in metric_rows if isinstance(row.get(key), (int, float))]
        if not numeric_values:
            continue
        averaged[key] = float(np.mean(np.asarray(numeric_values, dtype=np.float64)))
    averaged["aggregation"] = "equal_weight_mean_across_modes"
    averaged["count_note"] = "TP/FP/TN/FN are mean per-mode counts, not pooled counts."
    return averaged


def build_mode_split_payload(
    labels: np.ndarray,
    time: np.ndarray,
    episode_id: np.ndarray,
    mode_labels: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    threshold_gamma: float,
    mlp_threshold: float,
    gru_threshold: float,
    episode_counts: dict[str, dict[str, int]],
    mode_note: str,
) -> dict:
    mode_names = sorted(str(name) for name in np.unique(mode_labels))
    test_mode_metrics: dict[str, dict] = {}
    for mode_name in mode_names:
        mask = mode_labels == mode_name
        per_model_metrics = {
            detector_name: metrics_with_delay(labels[mask], prediction[mask], time[mask], episode_id[mask])
            for detector_name, prediction in predictions_by_model.items()
        }
        test_mode_metrics[mode_name] = {
            "num_test_episodes": int(len(np.unique(episode_id[mask]))),
            "num_windows": int(np.sum(mask)),
            **per_model_metrics,
        }

    equal_weight_average = {
        detector_name: average_metric_dicts(
            [test_mode_metrics[mode_name][detector_name] for mode_name in mode_names if detector_name in test_mode_metrics[mode_name]]
        )
        for detector_name in predictions_by_model
    }
    equal_weight_average["modes_included"] = mode_names

    return {
        "stage": None,
        "mode_inference_note": mode_note,
        "episode_counts": episode_counts,
        "threshold": {"gamma": float(threshold_gamma)},
        "mlp": {"decision_threshold": float(mlp_threshold)},
        "gru": {"decision_threshold": float(gru_threshold)},
        "test_mode_metrics": test_mode_metrics,
        "equal_weight_average": equal_weight_average,
    }


def value_in_bin(value: float, lower: float, upper: float, is_last: bool) -> bool:
    if is_last:
        return float(lower) <= float(value) <= float(upper)
    return float(lower) <= float(value) < float(upper)


def build_torque_bin_metrics(
    data: dict[str, np.ndarray],
    end_indices: np.ndarray,
    labels: np.ndarray,
    episode_id: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    bins: list[list[float]],
) -> dict:
    if "active_event_id" not in data or "event_table_id" not in data or "event_table_magnitude" not in data:
        return {
            "stage": None,
            "bin_definition": "unavailable",
            "note": "Dataset has no event metadata. Regenerate datasets with the updated generator.",
            "bins": [],
        }

    active_event_ids = np.asarray(data["active_event_id"]).astype(np.int64)[np.asarray(end_indices, dtype=np.int64)]
    event_ids = np.asarray(data["event_table_id"]).astype(np.int64)
    event_episode_ids = np.asarray(data["event_table_episode_id"]).astype(np.int64)
    event_modes = np.asarray(data["event_table_mode"]).astype("<U16")
    event_magnitudes = np.asarray(data["event_table_magnitude"], dtype=np.float64)

    rows: list[dict] = []
    for idx, bounds in enumerate(bins):
        lower = float(bounds[0])
        upper = float(bounds[1])
        is_last = idx == len(bins) - 1
        event_mask = np.asarray([value_in_bin(value, lower, upper, is_last) for value in event_magnitudes], dtype=bool)
        selected_event_ids = event_ids[event_mask]
        selected_episode_ids = np.unique(event_episode_ids[event_mask])
        positive_mask = np.isin(active_event_ids, selected_event_ids)
        negative_mask = (labels == 0) & np.isin(episode_id, selected_episode_ids)
        selected_mask = positive_mask | negative_mask

        if selected_event_ids.size == 0 or not np.any(selected_mask):
            rows.append(
                {
                    "bin_label": f"[{lower:.1f}, {upper:.1f}{']' if is_last else ')'}",
                    "lower_nm": lower,
                    "upper_nm": upper,
                    "num_events": int(selected_event_ids.size),
                    "num_episodes": int(selected_episode_ids.size),
                    "num_windows": int(np.sum(selected_mask)),
                    "modes_present": sorted(str(name) for name in np.unique(event_modes[event_mask])) if np.any(event_mask) else [],
                    "threshold": None,
                    "mlp": None,
                    "gru": None,
                }
            )
            continue

        row = {
            "bin_label": f"[{lower:.1f}, {upper:.1f}{']' if is_last else ')'}",
            "lower_nm": lower,
            "upper_nm": upper,
            "num_events": int(selected_event_ids.size),
            "num_episodes": int(selected_episode_ids.size),
            "num_windows": int(np.sum(selected_mask)),
            "modes_present": sorted(str(name) for name in np.unique(event_modes[event_mask])),
        }
        for detector_name, prediction in predictions_by_model.items():
            row[detector_name] = binary_classification_metrics(labels[selected_mask], prediction[selected_mask])
        rows.append(row)

    return {
        "stage": None,
        "bin_definition": (
            "Per-event representative ||tau_ext|| binning. Positives are windows whose active_event_id belongs to the "
            "selected bin; negatives are no-contact windows from the same episodes."
        ),
        "bins": rows,
    }


def save_example_payload(
    out_path,
    time: np.ndarray,
    labels: np.ndarray,
    threshold_pred: np.ndarray,
    mlp_prob: np.ndarray,
    gru_prob: np.ndarray,
    episode_id: np.ndarray,
    chosen_episode: int,
    e_norm: np.ndarray,
) -> dict[str, np.ndarray]:
    example_mask = episode_id == chosen_episode
    payload = {
        "time": time[example_mask],
        "label": labels[example_mask],
        "threshold_prediction": threshold_pred[example_mask],
        "mlp_probability": mlp_prob[example_mask],
        "gru_probability": gru_prob[example_mask],
        "episode_id": episode_id[example_mask],
        "e_norm": e_norm[example_mask],
    }
    np.savez_compressed(out_path, **payload)
    return payload


def run_original_split_evaluation(
    *,
    split_name: str,
    split_path: Path,
    config: dict,
    scaler: StandardScaler,
    threshold_payload: dict,
    mlp_model,
    gru_model,
    mlp_threshold: float,
    gru_threshold: float,
    torch_module,
    DataLoader,
    device,
    consecutive_samples: int,
    detection_margin_ms: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    raw = load_npz_dataset(split_path)
    bundle = ContactWindowDataset.from_npz(
        split_path,
        window_length=int(config["dataset"]["window_length"]),
        stride=int(config["dataset"]["stride"]),
        use_delta_features=bool(config["dataset"]["use_delta_features"]),
        scaler=scaler,
        feature_mode=str(config["dataset"].get("feature_mode", "original_42")),
    )
    step_dataset = bundle.dataset.aligned_single_step_dataset()
    labels = bundle.dataset.original_labels_for_windows().astype(np.int64)
    time = bundle.dataset.time_for_windows(raw["time"])
    episode_id = bundle.dataset.episodes_for_windows().astype(np.int64)
    end_indices = bundle.dataset.end_indices

    threshold_gamma = float(threshold_payload["gamma"])
    threshold_scores, threshold_meta = threshold_score_from_data(raw, end_indices, config)
    threshold_pred = (threshold_scores >= threshold_gamma).astype(np.int64)

    mlp_loader = DataLoader(
        step_dataset,
        batch_size=int(config["training"].get("mlp_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(config["training"].get("mlp_num_workers", config["training"].get("num_workers", 0))),
    )
    gru_loader = DataLoader(
        bundle.dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    mlp_prob = run_model_inference(mlp_model, mlp_loader, torch_module)
    gru_prob = run_model_inference(gru_model, gru_loader, torch_module)
    mlp_pred = (mlp_prob >= float(mlp_threshold)).astype(np.int64)
    gru_pred = (gru_prob >= float(gru_threshold)).astype(np.int64)
    threshold_sample_metrics = metrics_with_delay(labels, threshold_pred, time, episode_id)
    threshold_sample_metrics["loss"] = None
    mlp_sample_metrics = metrics_with_delay(labels, mlp_pred, time, episode_id)
    mlp_sample_metrics["loss"] = binary_log_loss_from_probability(labels, mlp_prob)
    gru_sample_metrics = metrics_with_delay(labels, gru_pred, time, episode_id)
    gru_sample_metrics["loss"] = binary_log_loss_from_probability(labels, gru_prob)

    payload = {
        "split": split_name,
        "label_basis": "original_external_force_command_label",
        "sample_metric_label_basis": "original_external_force_command_label",
        "event_metric_label_basis": "original_external_force_command_label",
        "threshold_policy": "reuse_selection_label_threshold",
        "threshold_selected_on": "selection_label_validation",
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        "threshold": {
            "decision_threshold": threshold_gamma,
            "threshold_metric": threshold_meta["threshold_metric"],
            "sample_level": threshold_sample_metrics,
            "event_level": event_latency_metrics(
                raw,
                end_indices,
                threshold_scores,
                threshold_gamma,
                consecutive_samples,
                detection_margin_ms,
            ),
        },
        "mlp": {
            "decision_threshold": float(mlp_threshold),
            "sample_level": mlp_sample_metrics,
            "event_level": event_latency_metrics(
                raw,
                end_indices,
                mlp_prob,
                mlp_threshold,
                consecutive_samples,
                detection_margin_ms,
            ),
        },
        "gru": {
            "decision_threshold": float(gru_threshold),
            "sample_level": gru_sample_metrics,
            "event_level": event_latency_metrics(
                raw,
                end_indices,
                gru_prob,
                gru_threshold,
                consecutive_samples,
                detection_margin_ms,
            ),
        },
    }
    arrays = {
        "labels": labels,
        "time": time,
        "episode_id": episode_id,
        "end_indices": end_indices,
        "threshold_pred": threshold_pred,
        "mlp_prob": mlp_prob,
        "mlp_pred": mlp_pred,
        "gru_prob": gru_prob,
        "gru_pred": gru_pred,
    }
    return payload, arrays


def transition_excluded_sample_metrics(
    raw: dict[str, np.ndarray],
    end_indices: np.ndarray,
    labels: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    transition_exclusion_ms: float,
) -> dict | None:
    full_time = np.asarray(raw["time"], dtype=np.float64).reshape(-1)
    diffs = np.diff(full_time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0 or float(transition_exclusion_ms) <= 0.0:
        return None
    dt = float(np.median(diffs))
    exclusion_steps = ms_to_steps(float(transition_exclusion_ms), dt)
    if exclusion_steps <= 0:
        return None
    sample_mask = transition_exclusion_sample_mask(raw["label"], raw["episode_id"], exclusion_steps)
    window_mask = sample_mask[np.asarray(end_indices, dtype=np.int64)]
    payload = {
        "transition_exclusion_ms": float(transition_exclusion_ms),
        "transition_exclusion_steps": int(exclusion_steps),
        "num_windows_total": int(len(labels)),
        "num_windows_kept": int(np.sum(window_mask)),
        "num_windows_excluded": int(len(labels) - np.sum(window_mask)),
    }
    for name, pred in predictions_by_model.items():
        payload[name] = binary_classification_metrics(labels[window_mask], pred[window_mask])
    return payload


def compute_feature_response_analysis(
    data: dict[str, np.ndarray],
    out_metrics_csv: Path,
    out_figure_path: Path,
    pre_ms: float = 100.0,
    post_ms: float = 200.0,
    threshold_sigma: float = 2.0,
) -> dict:
    full_time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    diffs = np.diff(full_time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    dt = float(np.median(diffs)) if diffs.size else 0.0
    if dt <= 0.0:
        return {"available": False, "note": "Could not estimate positive dt for feature response analysis."}

    pre_steps = ms_to_steps(pre_ms, dt)
    post_steps = ms_to_steps(post_ms, dt)
    offsets = np.arange(-pre_steps, post_steps + 1, dtype=np.int64)
    offset_ms = offsets.astype(np.float64) * dt * 1000.0

    q = np.asarray(data["q"], dtype=np.float64)
    q_des = np.asarray(data["q_des"], dtype=np.float64)
    qdot = np.asarray(data["qdot"], dtype=np.float64)
    tau_cmd = np.asarray(data["tau_cmd"], dtype=np.float64)
    tau_ext = np.asarray(data["tau_ext"], dtype=np.float64)
    episode_id = np.asarray(data["episode_id"]).reshape(-1)
    e_q = q_des - q
    features = {
        "norm_joint_error": np.linalg.norm(e_q, axis=1),
        "norm_delta_joint_error": np.linalg.norm(compute_episodewise_delta(e_q, episode_id), axis=1),
        "norm_qdot": np.linalg.norm(qdot, axis=1),
        "norm_delta_qdot": np.linalg.norm(compute_episodewise_delta(qdot, episode_id), axis=1),
        "norm_tau_cmd_change": np.linalg.norm(compute_episodewise_delta(tau_cmd, episode_id), axis=1),
        "norm_external_force": np.linalg.norm(tau_ext, axis=1),
    }

    event_starts = np.asarray(data.get("event_table_start_index", []), dtype=np.int64)
    valid_starts = []
    for idx in event_starts:
        start_idx = int(idx)
        lo = start_idx - pre_steps
        hi = start_idx + post_steps
        if lo < 0 or hi >= full_time.size:
            continue
        if episode_id[lo] != episode_id[start_idx] or episode_id[hi] != episode_id[start_idx]:
            continue
        valid_starts.append(start_idx)
    if not valid_starts:
        return {"available": False, "note": "No events have enough pre/post context for feature response analysis."}

    aligned: dict[str, np.ndarray] = {}
    for name, values in features.items():
        aligned[name] = np.stack([values[start + offsets] for start in valid_starts], axis=0)

    means = {name: np.mean(values, axis=0) for name, values in aligned.items()}
    response_delays: dict[str, dict] = {}
    baseline_mask = offset_ms < 0.0
    post_mask = offset_ms >= 0.0
    for name, mean_values in means.items():
        baseline_values = aligned[name][:, baseline_mask].reshape(-1)
        baseline_mean = float(np.mean(baseline_values))
        baseline_std = float(np.std(baseline_values))
        threshold = baseline_mean + float(threshold_sigma) * baseline_std
        crossings = np.flatnonzero(post_mask & (mean_values > threshold))
        response_delays[name] = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "threshold": float(threshold),
            "response_delay_ms": None if crossings.size == 0 else float(offset_ms[int(crossings[0])]),
        }

    force_mean = means["norm_external_force"]
    cross_correlation: dict[str, dict] = {}
    post_force = force_mean[post_mask] - float(np.mean(force_mean[post_mask]))
    for name in ("norm_delta_joint_error", "norm_qdot", "norm_delta_qdot", "norm_tau_cmd_change"):
        signal = means[name][post_mask]
        signal = signal - float(np.mean(signal))
        scores = []
        max_lag = min(post_steps, signal.size - 1)
        for lag in range(max_lag + 1):
            n = signal.size - lag
            if n <= 1:
                scores.append(float("-inf"))
            else:
                denom = float(np.linalg.norm(post_force[:n]) * np.linalg.norm(signal[lag : lag + n]))
                scores.append(0.0 if denom <= 0.0 else float(np.dot(post_force[:n], signal[lag : lag + n]) / denom))
        best_lag = int(np.argmax(np.asarray(scores, dtype=np.float64))) if scores else 0
        cross_correlation[name] = {
            "max_corr_delay_samples": best_lag,
            "max_corr_delay_ms": float(best_lag * dt * 1000.0),
            "max_corr_value": None if not scores else float(scores[best_lag]),
        }

    out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["offset_ms"] + [f"{name}_mean" for name in means]
        writer.writerow(header)
        for row_idx, offset_value in enumerate(offset_ms):
            writer.writerow([float(offset_value)] + [float(means[name][row_idx]) for name in means])

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for name in ("norm_external_force", "norm_delta_joint_error", "norm_qdot", "norm_tau_cmd_change"):
            values = means[name]
            max_value = float(np.max(values))
            scaled = values / max_value if max_value > 0.0 else values
            ax.plot(offset_ms, scaled, label=name)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time from force onset [ms]")
        ax.set_ylabel("Event-aligned mean (normalized)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_saved = True
    except Exception as exc:  # pragma: no cover - plotting should not break evaluation.
        figure_saved = False
        plot_error = str(exc)

    payload = {
        "available": True,
        "num_events_used": int(len(valid_starts)),
        "pre_ms": float(pre_ms),
        "post_ms": float(post_ms),
        "threshold_sigma": float(threshold_sigma),
        "csv_path": str(out_metrics_csv),
        "figure_path": str(out_figure_path) if figure_saved else None,
        "response_delays": response_delays,
        "cross_correlation": cross_correlation,
    }
    if not figure_saved:
        payload["plot_error"] = plot_error
    return payload


def ablation_rows_from_metrics(
    artifact_root: Path,
    ablation_tag: str,
    checkpoint_summary: dict | None,
    validation_metrics: dict,
    test_metrics: dict,
) -> list[dict]:
    rows: list[dict] = []
    for model_name in ("mlp", "gru"):
        ckpt = checkpoint_summary.get(model_name, {}) if checkpoint_summary else {}
        val_sample = validation_metrics[model_name]["sample_level"]
        val_event = validation_metrics[model_name]["event_level"]
        test_sample = test_metrics[model_name]
        test_event = test_metrics["event_level"][model_name]
        label_delay_ms = float(checkpoint_summary.get("label_delay_ms", 0.0)) if checkpoint_summary else 0.0
        transition_exclusion_ms = (
            float(checkpoint_summary.get("transition_exclusion_ms", 0.0)) if checkpoint_summary else 0.0
        )
        selection_label_basis = canonical_selection_label_basis(
            ckpt.get(
                "selection_label_basis",
                ckpt.get(
                    "label_basis_for_threshold_search",
                    checkpoint_summary.get("selection_label_basis", "original_command_label")
                    if checkpoint_summary
                    else "original_command_label",
                ),
            ),
            label_delay_ms,
        )
        threshold_basis = canonical_selection_label_basis(
            ckpt.get("label_basis_for_threshold_search", selection_label_basis),
            label_delay_ms,
        )
        row = {
            "ablation_tag": ablation_tag,
            "artifact_root": str(artifact_root),
            "model_type": model_name.upper(),
            "transition_exclusion_ms": transition_exclusion_ms,
            "label_delay_ms": label_delay_ms,
            "selection_label_basis": selection_label_basis,
            "label_basis_for_training": canonical_selection_label_basis(
                ckpt.get("label_basis_for_training", selection_label_basis),
                label_delay_ms,
            ),
            "label_basis_for_checkpoint_selection": canonical_selection_label_basis(
                ckpt.get("label_basis_for_checkpoint_selection", selection_label_basis),
                label_delay_ms,
            ),
            "label_basis_for_threshold_search": threshold_basis,
            "label_basis_for_original_evaluation": ckpt.get(
                "label_basis_for_original_evaluation",
                "original_external_force_command_label",
            ),
            "best_epoch": ckpt.get("best_epoch"),
            "best_epoch_selection_label": ckpt.get("best_epoch_selection_label", ckpt.get("best_epoch")),
            "best_val_f1_selection_label": ckpt.get("best_val_f1_selection_label", ckpt.get("best_val_f1")),
            "best_val_loss_selection_label": ckpt.get("best_val_loss_selection_label", ckpt.get("best_val_loss")),
            "best_val_precision_selection_label": ckpt.get(
                "best_val_precision_selection_label",
                ckpt.get("best_val_precision"),
            ),
            "best_val_recall_selection_label": ckpt.get(
                "best_val_recall_selection_label",
                ckpt.get("best_val_recall"),
            ),
            "decision_threshold_value": ckpt.get(
                "decision_threshold_value",
                ckpt.get("decision_threshold"),
            ),
            "threshold_selected_on": ckpt.get("threshold_selected_on", "selection_label_validation"),
            "threshold_selection_metric": ckpt.get("threshold_selection_metric", "val_f1_selection_label"),
            "threshold_applied_to_original_label_metrics": bool(
                ckpt.get("threshold_applied_to_original_label_metrics", True)
            ),
            "original_label_metric_threshold_policy": ckpt.get(
                "original_label_metric_threshold_policy",
                "reuse_selection_label_threshold",
            ),
            "validation_original_f1": val_sample.get("f1"),
            "validation_original_precision": val_sample.get("precision"),
            "validation_original_recall": val_sample.get("recall"),
            "validation_original_loss": val_sample.get("loss"),
            "validation_event_detection_rate": val_event.get("event_detection_rate"),
            "validation_latency_mean_ms": val_event.get("latency_mean_ms"),
            "validation_latency_median_ms": val_event.get("latency_median_ms"),
            "validation_latency_max_ms": val_event.get("latency_max_ms"),
            "validation_missed_event_count": val_event.get("number_of_missed_events"),
            "validation_false_alarm_per_second": val_event.get("false_alarm_per_second"),
            "test_precision": test_sample.get("precision"),
            "test_recall": test_sample.get("recall"),
            "test_f1": test_sample.get("f1"),
            "test_loss": test_sample.get("loss"),
            "test_event_detection_rate": test_event.get("event_detection_rate"),
            "test_latency_mean_ms": test_event.get("latency_mean_ms"),
            "test_latency_median_ms": test_event.get("latency_median_ms"),
            "test_latency_max_ms": test_event.get("latency_max_ms"),
            "test_missed_event_count": test_event.get("number_of_missed_events"),
            "test_false_alarm_count": test_event.get("false_alarm_count"),
            "test_false_alarm_per_second": test_event.get("false_alarm_per_second"),
            "test_metrics_used_for_selection": False,
            "recommended_by_validation_only": False,
        }
        rows.append(row)
    return rows


def recommendation_key(row: dict) -> tuple:
    return (
        float(row.get("validation_original_f1") or 0.0),
        float(row.get("validation_event_detection_rate") or 0.0),
        -float(row.get("validation_latency_mean_ms") or 1.0e9),
        -float(row.get("validation_missed_event_count") or 1.0e9),
        -float(row.get("validation_false_alarm_per_second") or 1.0e9),
    )


def save_ablation_summary(base_dirs: dict[str, Path], runs: list[dict]) -> dict:
    runs_sorted = sorted(runs, key=lambda row: (str(row.get("ablation_tag", "")), str(row.get("model_type", ""))))
    recommended = max(runs_sorted, key=recommendation_key) if runs_sorted else None
    if recommended is not None:
        for row in runs_sorted:
            row["recommended_by_validation_only"] = (
                row.get("ablation_tag") == recommended.get("ablation_tag")
                and row.get("model_type") == recommended.get("model_type")
            )
        recommended = next(row for row in runs_sorted if row["recommended_by_validation_only"])

    payload = {
        "selection_policy": (
            "Ablation recommendation uses validation original-label sample/event metrics only. "
            "Test metrics are recorded for final reporting and are not used for selection."
        ),
        "recommendation_priority": [
            "validation_original_f1",
            "validation_event_detection_rate",
            "lower_validation_latency_mean_ms",
            "lower_validation_missed_event_count",
            "lower_validation_false_alarm_per_second",
        ],
        "test_metrics_used_for_selection": False,
        "runs": runs_sorted,
        "recommended_by_validation_only": recommended,
    }
    save_json(base_dirs["metrics"] / "ablation_summary.json", payload)
    return payload


def rebuild_ablation_summary(base_dirs: dict[str, Path]) -> dict:
    ablations_root = base_dirs["root"] / "ablations"
    runs: list[dict] = []
    if ablations_root.exists():
        for artifact_root in sorted(path for path in ablations_root.iterdir() if path.is_dir()):
            metrics_dir = artifact_root / "metrics"
            checkpoint_path = metrics_dir / "checkpoint_summary.json"
            validation_path = metrics_dir / "validation_original_metrics.json"
            test_path = metrics_dir / "sim_test_metrics.json"
            if not (checkpoint_path.exists() and validation_path.exists() and test_path.exists()):
                continue
            checkpoint_summary = load_json(checkpoint_path)
            validation_metrics = load_json(validation_path)
            test_metrics = load_json(test_path)
            runs.extend(
                ablation_rows_from_metrics(
                    artifact_root,
                    artifact_root.name,
                    checkpoint_summary,
                    validation_metrics,
                    test_metrics,
                )
            )
    return save_ablation_summary(base_dirs, runs)


def update_ablation_summary(
    base_dirs: dict[str, Path],
    artifact_dirs: dict[str, Path],
    ablation_tag: str | None,
    checkpoint_summary: dict | None,
    validation_metrics: dict,
    test_metrics: dict,
) -> None:
    if ablation_tag is None:
        return
    # Rebuild from disk instead of appending blindly, so separate ablation runs
    # remain consistent even if one is re-run.
    rebuild_ablation_summary(base_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage override.")
    parser.add_argument(
        "--model-variant",
        choices=("best", "trainval_final"),
        default="best",
        help=(
            "Which learned artifacts to evaluate. 'best' uses mlp_detector.pt/gru_detector.pt. "
            "'trainval_final' uses the deployment-only train+val final artifacts and writes suffixed outputs."
        ),
    )
    parser.add_argument("--mlp-model-path", default="", help="Optional explicit MLP checkpoint path for poster/custom evaluation.")
    parser.add_argument("--gru-model-path", default="", help="Optional explicit GRU checkpoint path for poster/custom evaluation.")
    parser.add_argument("--scaler-path", default="", help="Optional explicit scaler path for poster/custom evaluation.")
    parser.add_argument("--threshold-path", default="", help="Optional explicit threshold.json path for poster/custom evaluation.")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional filename suffix for custom/poster outputs. Leading underscore is added automatically.",
    )
    parser.add_argument("--label-delay-ms", type=float, default=None, help="Ablation label delay used to infer artifact tag.")
    parser.add_argument(
        "--transition-exclusion-ms",
        type=float,
        default=None,
        help="Ablation transition exclusion used to infer artifact tag.",
    )
    parser.add_argument(
        "--ablation-tag",
        default=None,
        help="Evaluate artifacts under outputs/<stage>/ablations/<tag>/ instead of the default stage root.",
    )
    parser.add_argument(
        "--event-detection-consecutive-samples",
        type=int,
        default=None,
        help="Number of consecutive threshold-crossing samples required for event detection.",
    )
    parser.add_argument(
        "--detection-margin-ms",
        type=float,
        default=None,
        help="Search event detections until force offset plus this margin in ms.",
    )
    parser.add_argument(
        "--feature-response-window-pre-ms",
        type=float,
        default=None,
        help="Feature response alignment window before force onset in ms.",
    )
    parser.add_argument(
        "--feature-response-window-post-ms",
        type=float,
        default=None,
        help="Feature response alignment window after force onset in ms.",
    )
    parser.add_argument(
        "--scan-ablations",
        action="store_true",
        help="Scan outputs/<stage>/ablations/* metrics and rebuild root metrics/ablation_summary.json.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    training_cfg = config.setdefault("training", {})
    if args.label_delay_ms is not None:
        training_cfg["label_delay_ms"] = float(args.label_delay_ms)
    if args.transition_exclusion_ms is not None:
        training_cfg["transition_exclusion_ms"] = float(args.transition_exclusion_ms)
    ablation_tag = args.ablation_tag
    if ablation_tag is None and (
        float(training_cfg.get("label_delay_ms", 0.0) or 0.0) > 0.0
        or float(training_cfg.get("transition_exclusion_ms", 0.0) or 0.0) > 0.0
    ):
        ablation_tag = default_ablation_tag(
            float(training_cfg.get("label_delay_ms", 0.0) or 0.0),
            float(training_cfg.get("transition_exclusion_ms", 0.0) or 0.0),
        )
    base_dirs, out_dirs = resolve_artifact_dirs(config, ablation_tag)
    if args.scan_ablations:
        payload = rebuild_ablation_summary(base_dirs)
        print(f"Scanned ablations and saved summary to {base_dirs['metrics'] / 'ablation_summary.json'}")
        print(f"Found {len(payload.get('runs', []))} ablation model rows")
        return
    if ablation_tag is not None:
        config["ablation_tag"] = str(ablation_tag)
        config["ablation_output_dir"] = str(out_dirs["root"])
    save_config_yaml(out_dirs["root"] / "experiment_config_used.yaml", config)

    eval_cfg = config.setdefault("evaluation", {})
    consecutive_samples = int(
        args.event_detection_consecutive_samples
        if args.event_detection_consecutive_samples is not None
        else eval_cfg.get("event_detection_consecutive_samples", 3)
    )
    detection_margin_ms = float(
        args.detection_margin_ms if args.detection_margin_ms is not None else eval_cfg.get("detection_margin_ms", 50.0)
    )
    feature_response_pre_ms = float(
        args.feature_response_window_pre_ms
        if args.feature_response_window_pre_ms is not None
        else eval_cfg.get("feature_response_window_pre_ms", eval_cfg.get("feature_response_pre_ms", 100.0))
    )
    feature_response_post_ms = float(
        args.feature_response_window_post_ms
        if args.feature_response_window_post_ms is not None
        else eval_cfg.get("feature_response_window_post_ms", eval_cfg.get("feature_response_post_ms", 200.0))
    )

    val_path = base_dirs["datasets"] / "sim_val.npz"
    test_path = base_dirs["datasets"] / "sim_test.npz"
    threshold_path = Path(args.threshold_path).expanduser().resolve() if args.threshold_path.strip() else out_dirs["models"] / "threshold.json"
    scaler_path = Path(args.scaler_path).expanduser().resolve() if args.scaler_path.strip() else out_dirs["models"] / "scaler.pkl"
    if args.model_variant == "trainval_final":
        mlp_model_path = out_dirs["models"] / "mlp_detector_trainval_final.pt"
        gru_model_path = out_dirs["models"] / "gru_detector_trainval_final.pt"
        output_suffix = "_trainval_final"
    else:
        mlp_model_path = out_dirs["models"] / "mlp_detector.pt"
        gru_model_path = out_dirs["models"] / "gru_detector.pt"
        output_suffix = ""
    if args.mlp_model_path.strip():
        mlp_model_path = Path(args.mlp_model_path).expanduser().resolve()
    if args.gru_model_path.strip():
        gru_model_path = Path(args.gru_model_path).expanduser().resolve()
    if args.output_suffix.strip():
        custom_suffix = str(args.output_suffix).strip()
        if not custom_suffix.startswith("_"):
            custom_suffix = f"_{custom_suffix}"
        output_suffix = f"{output_suffix}{custom_suffix}"

    for path in (val_path, test_path, threshold_path, scaler_path, mlp_model_path, gru_model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation input is missing: {path}")

    threshold_payload = load_json(threshold_path)
    test_raw = load_npz_dataset(test_path)
    scaler = StandardScaler.load(scaler_path)

    test_bundle = ContactWindowDataset.from_npz(
        test_path,
        window_length=int(config["dataset"]["window_length"]),
        stride=int(config["dataset"]["stride"]),
        use_delta_features=bool(config["dataset"]["use_delta_features"]),
        scaler=scaler,
        feature_mode=str(config["dataset"].get("feature_mode", "original_42")),
    )
    test_step_dataset = test_bundle.dataset.aligned_single_step_dataset()
    if not np.array_equal(test_bundle.dataset.end_indices, test_step_dataset.selected_indices):
        raise RuntimeError("MLP evaluation alignment mismatch: single-step indices do not match GRU end_indices")

    labels = test_bundle.dataset.original_labels_for_windows().astype(np.int64)
    time = test_bundle.dataset.time_for_windows(test_raw["time"])
    episode_id = test_bundle.dataset.episodes_for_windows().astype(np.int64)
    end_indices = test_bundle.dataset.end_indices
    mode_labels, mode_note = resolve_window_modes(test_raw, end_indices)

    threshold_gamma = float(threshold_payload["gamma"])
    threshold_scores, threshold_meta = threshold_score_from_data(test_raw, end_indices, config)
    threshold_pred = (threshold_scores >= threshold_gamma).astype(np.int64)
    threshold_metrics = metrics_with_delay(labels, threshold_pred, time, episode_id)
    threshold_metrics["loss"] = None
    threshold_metrics["decision_threshold"] = threshold_gamma
    threshold_metrics["threshold_metric"] = threshold_meta["threshold_metric"]
    threshold_metrics["alpha"] = float(threshold_meta["alpha"])
    threshold_metrics["beta"] = float(threshold_meta["beta"])

    torch, DataLoader = _import_torch()
    device = select_torch_device(torch, config["training"].get("device", "auto"))

    mlp_checkpoint = torch.load(mlp_model_path, map_location="cpu")
    if str(mlp_checkpoint.get("model_type", "binary")) != "binary":
        raise ValueError(f"{mlp_model_path} is not a binary checkpoint.")
    mlp_model = build_model_from_checkpoint(mlp_checkpoint)
    mlp_model.to(device)
    print(f"Evaluating MLP on device={device}")
    mlp_loader = DataLoader(
        test_step_dataset,
        batch_size=int(config["training"].get("mlp_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(config["training"].get("mlp_num_workers", config["training"].get("num_workers", 0))),
    )
    mlp_prob = run_model_inference(mlp_model, mlp_loader, torch)
    configured_mlp_threshold = config["training"].get("mlp_decision_threshold")
    mlp_threshold = (
        float(configured_mlp_threshold)
        if configured_mlp_threshold is not None
        else float(mlp_checkpoint.get("decision_threshold", 0.5))
    )
    mlp_pred = (mlp_prob >= mlp_threshold).astype(np.int64)
    mlp_metrics = metrics_with_delay(labels, mlp_pred, time, episode_id)
    mlp_metrics["loss"] = binary_log_loss_from_probability(labels, mlp_prob)
    mlp_metrics["decision_threshold"] = mlp_threshold
    mlp_metrics["input_mode"] = str(mlp_checkpoint.get("input_mode", "single_step_aligned_to_window_end"))
    attach_checkpoint_metadata(mlp_metrics, mlp_checkpoint, mlp_model_path)
    if "validation_best_f1_threshold" in mlp_checkpoint:
        mlp_metrics["validation_best_f1_threshold"] = float(mlp_checkpoint["validation_best_f1_threshold"])

    gru_checkpoint = torch.load(gru_model_path, map_location="cpu")
    if str(gru_checkpoint.get("model_type", "binary")) != "binary":
        raise ValueError(f"{gru_model_path} is not a binary checkpoint.")
    gru_model = build_model_from_checkpoint(gru_checkpoint)
    gru_model.to(device)
    print(f"Evaluating GRU on device={device}")
    gru_loader = DataLoader(
        test_bundle.dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    gru_prob = run_model_inference(gru_model, gru_loader, torch)
    configured_gru_threshold = config["training"].get("gru_decision_threshold")
    gru_threshold = (
        float(configured_gru_threshold)
        if configured_gru_threshold is not None
        else float(gru_checkpoint.get("decision_threshold", 0.5))
    )
    gru_pred = (gru_prob >= gru_threshold).astype(np.int64)
    gru_metrics = metrics_with_delay(labels, gru_pred, time, episode_id)
    gru_metrics["loss"] = binary_log_loss_from_probability(labels, gru_prob)
    gru_metrics["decision_threshold"] = gru_threshold
    attach_checkpoint_metadata(gru_metrics, gru_checkpoint, gru_model_path)
    if "validation_best_f1_threshold" in gru_checkpoint:
        gru_metrics["validation_best_f1_threshold"] = float(gru_checkpoint["validation_best_f1_threshold"])

    event_level_metrics = {
        "threshold": event_latency_metrics(
            test_raw,
            end_indices,
            threshold_scores,
            threshold_gamma,
            consecutive_samples,
            detection_margin_ms,
        ),
        "mlp": event_latency_metrics(
            test_raw,
            end_indices,
            mlp_prob,
            mlp_threshold,
            consecutive_samples,
            detection_margin_ms,
        ),
        "gru": event_latency_metrics(
            test_raw,
            end_indices,
            gru_prob,
            gru_threshold,
            consecutive_samples,
            detection_margin_ms,
        ),
    }
    threshold_metrics["event_level"] = event_level_metrics["threshold"]
    mlp_metrics["event_level"] = event_level_metrics["mlp"]
    gru_metrics["event_level"] = event_level_metrics["gru"]
    transition_excluded_metrics = transition_excluded_sample_metrics(
        test_raw,
        end_indices,
        labels,
        {"threshold": threshold_pred, "mlp": mlp_pred, "gru": gru_pred},
        float(training_cfg.get("transition_exclusion_ms", 0.0) or 0.0),
    )

    validation_original_metrics, _validation_arrays = run_original_split_evaluation(
        split_name="sim_val",
        split_path=val_path,
        config=config,
        scaler=scaler,
        threshold_payload=threshold_payload,
        mlp_model=mlp_model,
        gru_model=gru_model,
        mlp_threshold=mlp_threshold,
        gru_threshold=gru_threshold,
        torch_module=torch,
        DataLoader=DataLoader,
        device=device,
        consecutive_samples=consecutive_samples,
        detection_margin_ms=detection_margin_ms,
    )
    validation_original_metrics.update(
        {
            "threshold_application_rule": (
                "Checkpoint and threshold were selected on selection-label validation F1; "
                "this original-label validation metric reuses that threshold without a new search."
            ),
            "label_basis_for_threshold_search": {
                "mlp": mlp_metrics["label_basis_for_threshold_search"],
                "gru": gru_metrics["label_basis_for_threshold_search"],
            },
            "label_basis_for_original_evaluation": "original_external_force_command_label",
            "threshold_selected_on": "selection_label_validation",
            "threshold_applied_to_original_label_metrics": True,
            "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        }
    )
    save_json(out_dirs["metrics"] / f"validation_original_metrics{output_suffix}.json", validation_original_metrics)

    mlp_sweep_thresholds = np.unique(
        np.concatenate([np.linspace(0.05, 0.95, 19, dtype=np.float64), np.asarray([mlp_threshold], dtype=np.float64)])
    )
    gru_sweep_thresholds = np.unique(
        np.concatenate([np.linspace(0.05, 0.95, 19, dtype=np.float64), np.asarray([gru_threshold], dtype=np.float64)])
    )
    mlp_sweep = threshold_sweep(mlp_prob, labels, time, episode_id, mlp_sweep_thresholds)
    gru_sweep = threshold_sweep(gru_prob, labels, time, episode_id, gru_sweep_thresholds)

    metrics_payload = {
        "stage": config["experiment_stage"],
        "model_variant": args.model_variant,
        "test_set_usage": "test split is used only for final performance evaluation, not model or threshold selection",
        "used_for_selection": False,
        "test_metrics_used_for_selection": False,
        "sample_metric_label_basis": "original_external_force_command_label",
        "event_metric_label_basis": "original_external_force_command_label",
        "threshold_policy": "reuse_selection_label_threshold",
        "threshold_application_rule": (
            "MLP/GRU decision thresholds are selected on selection-label validation data and reused unchanged "
            "for original-command-label validation/test metrics."
        ),
        "threshold_selected_on": "selection_label_validation",
        "label_basis_for_threshold_search": {
            "mlp": mlp_metrics["label_basis_for_threshold_search"],
            "gru": gru_metrics["label_basis_for_threshold_search"],
        },
        "label_basis_for_original_evaluation": "original_external_force_command_label",
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        "event_detection_consecutive_samples": int(consecutive_samples),
        "detection_margin_ms": float(detection_margin_ms),
        "feature_response_window_pre_ms": float(feature_response_pre_ms),
        "feature_response_window_post_ms": float(feature_response_post_ms),
        "validation_original_metrics_path": str(out_dirs["metrics"] / f"validation_original_metrics{output_suffix}.json"),
        "event_level": event_level_metrics,
        "transition_excluded_sample_metrics": transition_excluded_metrics,
        "threshold": threshold_metrics,
        "mlp": mlp_metrics,
        "gru": gru_metrics,
        "mlp_threshold_sweep": mlp_sweep,
        "gru_threshold_sweep": gru_sweep,
    }
    sim_metrics_path = out_dirs["metrics"] / f"sim_test_metrics{output_suffix}.json"
    save_json(sim_metrics_path, metrics_payload)
    event_latency_payload = {
        "stage": config["experiment_stage"],
        "model_variant": args.model_variant,
        "ablation_tag": ablation_tag,
        "label_basis": "original_external_force_command_label",
        "sample_metric_label_basis": "original_external_force_command_label",
        "event_metric_label_basis": "original_external_force_command_label",
        "threshold_policy": "reuse_selection_label_threshold",
        "threshold_selected_on": "selection_label_validation",
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        "event_detection_consecutive_samples": int(consecutive_samples),
        "detection_margin_ms": float(detection_margin_ms),
        "validation": {
            "threshold": validation_original_metrics["threshold"]["event_level"],
            "mlp": validation_original_metrics["mlp"]["event_level"],
            "gru": validation_original_metrics["gru"]["event_level"],
        },
        "test": event_level_metrics,
    }
    save_json(out_dirs["metrics"] / f"event_latency_metrics{output_suffix}.json", event_latency_payload)

    split_episode_counts: dict[str, dict[str, int]] = {}
    for split_name in ("sim_train", "sim_val", "sim_test"):
        split_path = base_dirs["datasets"] / f"{split_name}.npz"
        if split_path.exists():
            split_episode_counts[split_name] = count_episode_modes(load_npz_dataset(split_path))

    mode_split_payload = build_mode_split_payload(
        labels=labels,
        time=time,
        episode_id=episode_id,
        mode_labels=mode_labels,
        predictions_by_model={"threshold": threshold_pred, "mlp": mlp_pred, "gru": gru_pred},
        threshold_gamma=threshold_gamma,
        mlp_threshold=mlp_threshold,
        gru_threshold=gru_threshold,
        episode_counts=split_episode_counts,
        mode_note=mode_note,
    )
    mode_split_payload["stage"] = config["experiment_stage"]
    mode_split_payload["model_variant"] = args.model_variant
    save_json(out_dirs["metrics"] / f"mode_split_metrics{output_suffix}.json", mode_split_payload)

    torque_bins = config.get("evaluation", {}).get(
        "torque_bins",
        [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.5]],
    )
    torque_bin_payload = build_torque_bin_metrics(
        test_raw,
        end_indices,
        labels,
        episode_id,
        {"threshold": threshold_pred, "mlp": mlp_pred, "gru": gru_pred},
        torque_bins,
    )
    torque_bin_payload["stage"] = config["experiment_stage"]
    torque_bin_payload["model_variant"] = args.model_variant
    save_json(out_dirs["metrics"] / f"torque_bin_metrics{output_suffix}.json", torque_bin_payload)

    feature_response_payload = compute_feature_response_analysis(
        test_raw,
        out_dirs["metrics"] / f"feature_response_aligned{output_suffix}.csv",
        out_dirs["figures"] / f"feature_response_aligned{output_suffix}.png",
        pre_ms=feature_response_pre_ms,
        post_ms=feature_response_post_ms,
        threshold_sigma=float(eval_cfg.get("feature_response_threshold_sigma", 2.0)),
    )
    feature_response_payload["stage"] = config["experiment_stage"]
    feature_response_payload["model_variant"] = args.model_variant
    feature_response_payload["ablation_tag"] = ablation_tag
    save_json(out_dirs["metrics"] / f"feature_response_analysis{output_suffix}.json", feature_response_payload)

    poster_metrics_by_model = {
        "Threshold": threshold_metrics,
        "MLP": mlp_metrics,
        "GRU": gru_metrics,
    }
    poster_confusion_summary = build_poster_confusion_summary(
        config["experiment_stage"],
        args.model_variant,
        poster_metrics_by_model,
    )
    save_json(out_dirs["metrics"] / f"confusion_matrix_summary{output_suffix}.json", poster_confusion_summary)

    confusion_threshold = confusion_matrix_from_metrics(threshold_metrics)
    confusion_mlp = confusion_matrix_from_metrics(mlp_metrics)
    confusion_gru = confusion_matrix_from_metrics(gru_metrics)
    save_binary_nc_pc_confusion_matrix_figure(
        out_dirs["figures"] / f"confusion_matrix_threshold{output_suffix}.png",
        confusion_threshold,
        title="Threshold nc/pc confusion matrix",
        metrics=threshold_metrics,
    )
    save_binary_nc_pc_confusion_matrix_figure(
        out_dirs["figures"] / f"confusion_matrix_gru{output_suffix}.png",
        confusion_gru,
        title="GRU nc/pc confusion matrix",
        metrics=gru_metrics,
    )
    save_binary_nc_pc_confusion_matrix_figure(
        out_dirs["figures"] / f"confusion_matrix_mlp{output_suffix}.png",
        confusion_mlp,
        title="MLP nc/pc confusion matrix",
        metrics=mlp_metrics,
    )
    save_nc_pc_confusion_comparison_figure(
        out_dirs["figures"] / f"confusion_matrix_comparison{output_suffix}.png",
        {
            "Threshold": confusion_threshold,
            "MLP": confusion_mlp,
            "GRU": confusion_gru,
        },
        metrics_by_model=poster_metrics_by_model,
        title="Test-set nc/pc confusion matrix comparison",
    )
    save_metric_bar_figure(out_dirs["figures"] / f"sim_metric_bar{output_suffix}.png", metrics_payload)
    save_threshold_tradeoff_figure(
        out_dirs["figures"] / f"gru_threshold_tradeoff{output_suffix}.png",
        gru_sweep,
        comparison_sweeps={"MLP": mlp_sweep},
    )
    save_precision_recall_curve_figure(
        out_dirs["figures"] / f"precision_recall_curve_mlp_gru{output_suffix}.png",
        {"MLP": mlp_sweep, "GRU": gru_sweep},
    )

    positive_episodes = []
    for episode in np.unique(episode_id):
        mask = episode_id == episode
        if int(np.sum(labels[mask])) > 0:
            metrics = binary_classification_metrics(labels[mask], gru_pred[mask])
            positive_episodes.append((float(metrics["f1"]), int(episode)))
    positive_episodes.sort()
    chosen_episode = positive_episodes[len(positive_episodes) // 2][1] if positive_episodes else int(episode_id[0])
    e_norm_full = np.linalg.norm(test_raw["q_des"] - test_raw["q"], axis=1)[end_indices]
    example_payload = save_example_payload(
        out_dirs["metrics"] / f"sim_prediction_example_data{output_suffix}.npz",
        time,
        labels,
        threshold_pred,
        mlp_prob,
        gru_prob,
        episode_id,
        chosen_episode,
        e_norm_full,
    )
    save_sim_prediction_example(
        out_dirs["figures"] / f"sim_prediction_example{output_suffix}.png",
        example_payload["time"],
        example_payload["label"],
        example_payload["threshold_prediction"],
        example_payload["gru_probability"],
        mlp_probability=example_payload["mlp_probability"],
        mlp_decision_threshold=mlp_threshold,
        gru_decision_threshold=gru_threshold,
        e_norm=example_payload["e_norm"],
    )
    examples = choose_episode_examples(time, labels, threshold_pred, mlp_prob, gru_prob, gru_pred, episode_id)
    if examples:
        save_sim_prediction_examples(
            out_dirs["figures"] / f"sim_prediction_examples{output_suffix}.png",
            examples,
            mlp_decision_threshold=mlp_threshold,
            gru_decision_threshold=gru_threshold,
        )

    checkpoint_summary_path = out_dirs["metrics"] / "checkpoint_summary.json"
    checkpoint_summary = load_json(checkpoint_summary_path) if checkpoint_summary_path.exists() else None
    update_ablation_summary(
        base_dirs,
        out_dirs,
        ablation_tag,
        checkpoint_summary,
        validation_original_metrics,
        metrics_payload,
    )

    print(f"Saved simulation test metrics to {sim_metrics_path}")


if __name__ == "__main__":
    main()
