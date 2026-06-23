"""Train threshold, MLP, and GRU binary contact detectors.

논문/후속 실험 흐름에서 이 파일은 baseline과 학습 모델을 만드는 단계다.

1. train split으로 dimension-wise scaler를 fit하고 저장한다.
2. validation split에서 threshold baseline gamma를 선택한다.
3. single-step MLP baseline을 BCEWithLogitsLoss로 학습한다.
4. temporal GRU detector를 BCEWithLogitsLoss로 학습한다.
5. class imbalance는 pos_weight=N_negative/N_positive로 보정한다.
6. validation F1-score가 가장 높은 epoch의 checkpoint를 저장한다.

중요한 validity rule:
- 모델 입력은 contact_dataset.py가 만든 [q, qdot, e_q, tau_cmd, optional delta]뿐이다.
- tau_ext/external force는 feature/scaler/checkpoint에 들어가지 않는다.
- train/val은 이미 episode 단위로 분리된 npz를 사용한다.
- MLP도 GRU와 같은 window end index에 해당하는 시점만 사용해 공정 비교한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from contact_dataset import ContactWindowDataset
from utils import (
    apply_stage_config,
    build_input_features,
    ensure_output_dirs,
    load_config,
    load_json,
    load_npz_dataset,
    load_real_log_csv,
    output_root,
    save_config_yaml,
    save_json,
    save_training_curve,
    search_threshold_with_policy,
    select_torch_device,
    set_global_seed,
    threshold_score_from_data,
)


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install the dependencies from "
            "contact_detection/requirements.txt before running train_detectors.py."
        ) from exc


def evaluate_model(model, loader, criterion, torch_module) -> tuple[float, np.ndarray, np.ndarray]:
    """Validation loop returning loss, P(contact), and labels."""
    model.eval()
    losses: list[float] = []
    prob_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    device = next(model.parameters()).device
    with torch_module.no_grad():
        for batch in loader:
            features, labels = batch[:2]
            features = features.to(device=device, dtype=torch_module.float32)
            labels = labels.to(device=device, dtype=torch_module.float32)
            logits = model(features)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            prob_chunks.append(torch_module.sigmoid(logits).cpu().numpy())
            label_chunks.append(labels.cpu().numpy())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    probabilities = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.zeros(0, dtype=np.float64)
    labels_np = np.concatenate(label_chunks, axis=0) if label_chunks else np.zeros(0, dtype=np.float64)
    return mean_loss, probabilities, labels_np


def training_value(training_cfg: dict, key: str, model_prefix: str, default=None):
    prefixed_key = f"{model_prefix}_{key}"
    if prefixed_key in training_cfg:
        return training_cfg[prefixed_key]
    if key in training_cfg:
        return training_cfg[key]
    return default


def model_seed(config: dict, training_cfg: dict, model_prefix: str) -> int:
    """Return the seed used to isolate one model's init and shuffling."""

    base_seed = int(config.get("seed", 42))
    configured_seed = training_value(training_cfg, "seed", model_prefix, None)
    if configured_seed is not None:
        return int(configured_seed)
    seed_offset = int(training_value(training_cfg, "seed_offset", model_prefix, 0) or 0)
    return base_seed + seed_offset


def real_no_contact_dataset_from_csv(
    csv_path: str | Path,
    config: dict,
    scaler,
    *,
    window_length: int,
    stride: int,
    use_delta_features: bool,
    feature_mode: str,
) -> ContactWindowDataset:
    """Build a no-contact hard-negative window dataset from a real robot CSV."""

    real_data = load_real_log_csv(csv_path, config)
    episode_id = np.asarray(real_data.get("episode_id", np.zeros(real_data["time"].shape[0])), dtype=np.int64)
    labels = np.zeros(real_data["time"].shape[0], dtype=np.float32)
    features, _names = build_input_features(
        q=real_data["q"],
        qdot=real_data["qdot"],
        q_des=real_data["q_des"],
        tau_cmd=real_data["tau_cmd"],
        use_delta_features=use_delta_features,
        episode_id=episode_id,
        tau_residual=real_data.get("tau_residual"),
        tau_residual_corrected=real_data.get("tau_residual_corrected"),
        feature_mode=feature_mode,
    )
    return ContactWindowDataset(
        features=features,
        labels=labels,
        episode_id=episode_id,
        window_length=window_length,
        stride=stride,
        scaler=scaler,
        fit_scaler=False,
    )


def build_optimizer(torch_module, model, training_cfg: dict, model_prefix: str):
    optimizer_name = str(training_value(training_cfg, "optimizer", model_prefix, "adam")).strip().lower()
    lr = float(training_value(training_cfg, "lr", model_prefix, 1.0e-3))
    weight_decay = float(training_value(training_cfg, "weight_decay", model_prefix, 0.0))
    if optimizer_name == "adam":
        optimizer = torch_module.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch_module.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer for {model_prefix}: {optimizer_name!r}. Use 'adam' or 'adamw'.")
    return optimizer, optimizer_name, lr, weight_decay


def checkpoint_score_from_epoch(
    metric_name: str,
    val_loss: float,
    val_metrics: dict,
    val_selection_score: float,
) -> tuple[float, float, str]:
    """Return a comparable checkpoint score.

    Higher ``score`` is always better. ``raw_value`` keeps the human-readable
    value in train_log.json, so val_loss remains a loss instead of a negated
    score.
    """
    metric = str(metric_name).strip().lower()
    if metric == "val_loss":
        return -float(val_loss), float(val_loss), metric
    if metric in {"selection_score", "val_selection_score"}:
        return float(val_selection_score), float(val_selection_score), "selection_score"
    if metric == "val_f1":
        return float(val_metrics["f1"]), float(val_metrics["f1"]), metric
    if metric == "val_f2":
        return float(val_metrics["f2"]), float(val_metrics["f2"]), metric
    if metric == "val_recall":
        return float(val_metrics["recall"]), float(val_metrics["recall"]), metric
    raise ValueError(
        "checkpoint_selection_metric must be one of: "
        "val_loss, selection_score, val_f1, val_f2, val_recall; "
        f"got {metric_name!r}"
    )


def checkpoint_rank_from_epoch(
    metric_name: str,
    val_loss: float,
    val_metrics: dict,
    val_selection_score: float,
) -> tuple[tuple[float, ...], float, str]:
    """Return a lexicographic checkpoint rank.

    The primary score follows ``checkpoint_selection_metric``.  For the default
    validation-F1 selection, lower validation loss is used only as a tie-breaker
    when the F1 value is exactly tied.
    """

    score, raw_value, normalized_metric = checkpoint_score_from_epoch(
        metric_name,
        val_loss,
        val_metrics,
        val_selection_score,
    )
    if normalized_metric == "val_f1":
        return (float(score), -float(val_loss)), raw_value, normalized_metric
    return (float(score),), raw_value, normalized_metric


def checkpoint_rank_is_better(
    candidate_rank: tuple[float, ...],
    best_rank: tuple[float, ...] | None,
    min_delta: float,
) -> bool:
    if best_rank is None:
        return True
    primary_delta = float(candidate_rank[0]) - float(best_rank[0])
    if primary_delta > float(min_delta):
        return True
    # Tie-break only when the primary metric is effectively identical.  This
    # keeps "highest validation F1" as the real selection rule.
    if np.isclose(primary_delta, 0.0, atol=1.0e-12, rtol=0.0):
        return candidate_rank[1:] > best_rank[1:]
    return False


def checkpoint_summary_entry(model_name: str, checkpoint: dict, model_path: str) -> dict:
    selection_label_basis = checkpoint.get(
        "selection_label_basis",
        checkpoint.get("label_basis_for_threshold_search", "original_command_label"),
    )
    return {
        "model": model_name,
        "model_path": model_path,
        "feature_mode": checkpoint.get("feature_mode", "original_42"),
        "feature_names": checkpoint.get("feature_names", []),
        "residual_expected_torque_mode": checkpoint.get("residual_expected_torque_mode", "command_total"),
        "residual_offset_policy": checkpoint.get("residual_offset_policy", "episode_initial_no_contact_mean"),
        "tau_ext_input_policy": checkpoint.get("tau_ext_input_policy", "label_only_never_feature"),
        "model_seed": checkpoint.get("model_seed"),
        "isolate_model_random_seed": bool(checkpoint.get("isolate_model_random_seed", True)),
        "checkpoint_selection_metric": checkpoint.get("checkpoint_selection_metric"),
        "checkpoint_selected_on": checkpoint.get("checkpoint_selected_on", "selection_label_validation"),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_epoch_selection_label": int(
            checkpoint.get("best_epoch_selection_label", checkpoint.get("best_epoch", -1))
        ),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "best_val_precision": float(checkpoint.get("best_val_precision", float("nan"))),
        "best_val_recall": float(checkpoint.get("best_val_recall", float("nan"))),
        "best_val_f1": float(checkpoint.get("best_val_f1", float("nan"))),
        "best_val_loss_selection_label": float(
            checkpoint.get("best_val_loss_selection_label", checkpoint.get("best_val_loss", float("nan")))
        ),
        "best_val_precision_selection_label": float(
            checkpoint.get("best_val_precision_selection_label", checkpoint.get("best_val_precision", float("nan")))
        ),
        "best_val_recall_selection_label": float(
            checkpoint.get("best_val_recall_selection_label", checkpoint.get("best_val_recall", float("nan")))
        ),
        "best_val_f1_selection_label": float(
            checkpoint.get("best_val_f1_selection_label", checkpoint.get("best_val_f1", float("nan")))
        ),
        "decision_threshold": float(checkpoint.get("decision_threshold", 0.5)),
        "decision_threshold_value": float(
            checkpoint.get("decision_threshold_value", checkpoint.get("decision_threshold", 0.5))
        ),
        "validation_selected_threshold": float(checkpoint.get("validation_selected_threshold", 0.5)),
        "threshold_selection_metric": checkpoint.get("threshold_selection_metric", "val_f1_selection_label"),
        "threshold_selected_on": checkpoint.get("threshold_selected_on", "selection_label_validation"),
        "selection_label_basis": selection_label_basis,
        "label_basis_for_training": checkpoint.get("label_basis_for_training", selection_label_basis),
        "label_basis_for_checkpoint_selection": checkpoint.get(
            "label_basis_for_checkpoint_selection",
            selection_label_basis,
        ),
        "label_basis_for_threshold_search": checkpoint.get(
            "label_basis_for_threshold_search",
            "original_command_label",
        ),
        "label_basis_for_original_evaluation": checkpoint.get(
            "label_basis_for_original_evaluation",
            label_basis_for_original_evaluation(),
        ),
        "threshold_applied_to_original_label_metrics": bool(
            checkpoint.get("threshold_applied_to_original_label_metrics", True)
        ),
        "original_label_metric_threshold_policy": checkpoint.get(
            "original_label_metric_threshold_policy",
            "reuse_selection_label_threshold",
        ),
    }


def threshold_selection_metric_name(selection_policy: str) -> str:
    policy = str(selection_policy).strip().lower()
    if policy == "f1":
        return "val_f1_selection_label"
    if policy == "f2":
        return "val_f2_selection_label"
    if policy == "recall_constrained_f1":
        return "val_recall_constrained_f1_selection_label"
    return f"val_{policy}_selection_label"


def label_basis_for_threshold_search(training_cfg: dict) -> str:
    label_delay_ms = float(training_cfg.get("label_delay_ms", 0.0) or 0.0)
    if label_delay_ms > 0.0:
        return "delayed_training_label"
    return "original_command_label"


def label_basis_for_original_evaluation() -> str:
    return "original_external_force_command_label"


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


def train_fixed_epoch_model(
    *,
    model_name: str,
    model_arch: str,
    model_prefix: str,
    model,
    train_loader,
    torch_module,
    training_cfg: dict,
    dataset_cfg: dict,
    feature_names: list[str],
    source_checkpoint: dict,
    num_epochs: int,
    pos_weight_value: float,
    stage_name: str,
    extra_checkpoint_fields: dict | None = None,
) -> tuple[dict, dict]:
    """Retrain on train+val for a fixed number of epochs.

    This path is for deployment artifacts only.  It does not use validation or
    test data to choose a checkpoint.  The epoch count and decision threshold
    come from the original validation-selected checkpoint.
    """

    if int(num_epochs) < 1:
        raise ValueError(f"Final train+val epoch count must be >= 1, got {num_epochs}")

    device = select_torch_device(torch_module, training_value(training_cfg, "device", model_prefix, "auto"))
    model.to(device)
    criterion = torch_module.nn.BCEWithLogitsLoss(
        pos_weight=torch_module.tensor(pos_weight_value, dtype=torch_module.float32, device=device)
    )
    optimizer, optimizer_name, lr, weight_decay = build_optimizer(torch_module, model, training_cfg, model_prefix)
    grad_clip_norm = training_value(training_cfg, "grad_clip_norm", model_prefix, training_cfg.get("grad_clip_norm"))
    history: list[dict] = []

    print(f"Training final train+val {model_name} on device={device} for {int(num_epochs)} epochs")
    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            features, labels = batch[:2]
            features = features.to(device=device, dtype=torch_module.float32)
            labels = labels.to(device=device, dtype=torch_module.float32)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm is not None:
                torch_module.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_log = {"epoch": epoch, "train_loss": train_loss}
        history.append(epoch_log)
        print(json.dumps({"model": f"{model_name}_trainval_final", **epoch_log}))

    checkpoint = {key: value for key, value in source_checkpoint.items() if key != "state_dict"}
    checkpoint.update(
        {
            "model_type": "binary",
            "model_arch": model_arch,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_dim": int(getattr(model, "input_dim", len(feature_names))),
            "feature_names": feature_names,
            "feature_mode": str(dataset_cfg.get("feature_mode", "original_42")),
            "use_delta_features": bool(dataset_cfg["use_delta_features"]),
            "residual_expected_torque_mode": "command_total",
            "residual_offset_policy": "episode_initial_no_contact_mean",
            "tau_ext_input_policy": "label_only_never_feature",
            "window_length": int(dataset_cfg["window_length"]),
            "stride": int(dataset_cfg["stride"]),
            "final_trainval_model": True,
            "final_training_split": "train+val",
            "trainval_epochs": int(num_epochs),
            "source_checkpoint_best_epoch": int(source_checkpoint.get("best_epoch", num_epochs)),
            "source_checkpoint_selection_metric": source_checkpoint.get("checkpoint_selection_metric"),
            "source_best_val_loss": float(source_checkpoint.get("best_val_loss", float("nan"))),
            "source_best_val_precision": float(source_checkpoint.get("best_val_precision", float("nan"))),
            "source_best_val_recall": float(source_checkpoint.get("best_val_recall", float("nan"))),
            "source_best_val_f1": float(source_checkpoint.get("best_val_f1", float("nan"))),
            "decision_threshold": float(source_checkpoint.get("decision_threshold", 0.5)),
            "decision_threshold_value": float(
                source_checkpoint.get("decision_threshold_value", source_checkpoint.get("decision_threshold", 0.5))
            ),
            "threshold_source": "validation_selected_threshold_from_original_checkpoint",
            "threshold_selection_metric": source_checkpoint.get("threshold_selection_metric", "val_f1_selection_label"),
            "threshold_selected_on": source_checkpoint.get("threshold_selected_on", "selection_label_validation"),
            "checkpoint_selected_on": source_checkpoint.get("checkpoint_selected_on", "selection_label_validation"),
            "selection_label_basis": source_checkpoint.get(
                "selection_label_basis",
                source_checkpoint.get("label_basis_for_threshold_search", "original_command_label"),
            ),
            "label_basis_for_training": source_checkpoint.get(
                "label_basis_for_training",
                source_checkpoint.get("label_basis_for_threshold_search", "original_command_label"),
            ),
            "label_basis_for_checkpoint_selection": source_checkpoint.get(
                "label_basis_for_checkpoint_selection",
                source_checkpoint.get("label_basis_for_threshold_search", "original_command_label"),
            ),
            "label_basis_for_threshold_search": source_checkpoint.get(
                "label_basis_for_threshold_search",
                "original_command_label",
            ),
            "label_basis_for_original_evaluation": source_checkpoint.get(
                "label_basis_for_original_evaluation",
                label_basis_for_original_evaluation(),
            ),
            "threshold_applied_to_original_label_metrics": bool(
                source_checkpoint.get("threshold_applied_to_original_label_metrics", True)
            ),
            "original_label_metric_threshold_policy": source_checkpoint.get(
                "original_label_metric_threshold_policy",
                "reuse_selection_label_threshold",
            ),
            "checkpoint_selection_metric": "trainval_fixed_epochs_from_validation_f1_best_epoch",
            "training_device": str(device),
            "optimizer": optimizer_name,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "pos_weight": float(pos_weight_value),
            "stage": stage_name,
            "test_set_usage": "not_used_for_training_checkpoint_or_threshold_selection",
        }
    )
    if extra_checkpoint_fields:
        checkpoint.update(extra_checkpoint_fields)

    log_payload = {
        "history": history,
        "model": model_name,
        "model_type": "binary",
        "model_arch": model_arch,
        "final_training_split": "train+val",
        "trainval_epochs": int(num_epochs),
        "source_checkpoint_best_epoch": int(source_checkpoint.get("best_epoch", num_epochs)),
        "source_checkpoint_selection_metric": source_checkpoint.get("checkpoint_selection_metric"),
        "source_best_val_loss": float(source_checkpoint.get("best_val_loss", float("nan"))),
        "source_best_val_precision": float(source_checkpoint.get("best_val_precision", float("nan"))),
        "source_best_val_recall": float(source_checkpoint.get("best_val_recall", float("nan"))),
        "source_best_val_f1": float(source_checkpoint.get("best_val_f1", float("nan"))),
        "decision_threshold": float(source_checkpoint.get("decision_threshold", 0.5)),
        "feature_mode": str(dataset_cfg.get("feature_mode", "original_42")),
        "residual_expected_torque_mode": "command_total",
        "residual_offset_policy": "episode_initial_no_contact_mean",
        "tau_ext_input_policy": "label_only_never_feature",
        "decision_threshold_value": float(
            source_checkpoint.get("decision_threshold_value", source_checkpoint.get("decision_threshold", 0.5))
        ),
        "threshold_source": "validation_selected_threshold_from_original_checkpoint",
        "threshold_selection_metric": source_checkpoint.get("threshold_selection_metric", "val_f1_selection_label"),
        "threshold_selected_on": source_checkpoint.get("threshold_selected_on", "selection_label_validation"),
        "checkpoint_selected_on": source_checkpoint.get("checkpoint_selected_on", "selection_label_validation"),
        "selection_label_basis": source_checkpoint.get(
            "selection_label_basis",
            source_checkpoint.get("label_basis_for_threshold_search", "original_command_label"),
        ),
        "label_basis_for_training": source_checkpoint.get(
            "label_basis_for_training",
            source_checkpoint.get("label_basis_for_threshold_search", "original_command_label"),
        ),
        "label_basis_for_checkpoint_selection": source_checkpoint.get(
            "label_basis_for_checkpoint_selection",
            source_checkpoint.get("label_basis_for_threshold_search", "original_command_label"),
        ),
        "label_basis_for_threshold_search": source_checkpoint.get(
            "label_basis_for_threshold_search",
            "original_command_label",
        ),
        "label_basis_for_original_evaluation": source_checkpoint.get(
            "label_basis_for_original_evaluation",
            label_basis_for_original_evaluation(),
        ),
        "threshold_applied_to_original_label_metrics": bool(
            source_checkpoint.get("threshold_applied_to_original_label_metrics", True)
        ),
        "original_label_metric_threshold_policy": source_checkpoint.get(
            "original_label_metric_threshold_policy",
            "reuse_selection_label_threshold",
        ),
        "pos_weight": float(pos_weight_value),
        "optimizer": optimizer_name,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "training_device": str(device),
        "stage": stage_name,
        "label_delay_ms": float(training_cfg.get("label_delay_ms", 0.0) or 0.0),
        "transition_exclusion_ms": float(training_cfg.get("transition_exclusion_ms", 0.0) or 0.0),
        "test_set_usage": "test split is not used; it remains final performance check only",
    }
    return checkpoint, log_payload


def train_binary_model(
    *,
    model_name: str,
    model_arch: str,
    model_prefix: str,
    model,
    train_loader,
    val_loader,
    torch_module,
    training_cfg: dict,
    dataset_cfg: dict,
    feature_names: list[str],
    pos_weight_value: float,
    selection_policy: str,
    target_recall: float,
    fbeta_beta: float,
    stage_name: str,
    extra_checkpoint_fields: dict | None = None,
) -> tuple[dict, dict, list[dict]]:
    device = select_torch_device(torch_module, training_value(training_cfg, "device", model_prefix, "auto"))
    model.to(device)
    criterion = torch_module.nn.BCEWithLogitsLoss(
        pos_weight=torch_module.tensor(pos_weight_value, dtype=torch_module.float32, device=device)
    )
    optimizer, optimizer_name, lr, weight_decay = build_optimizer(torch_module, model, training_cfg, model_prefix)
    print(f"Training {model_name} on device={device}")

    patience = int(training_value(training_cfg, "early_stopping_patience", model_prefix, training_cfg["early_stopping_patience"]))
    num_epochs = int(training_value(training_cfg, "epochs", model_prefix, training_cfg["epochs"]))
    min_delta = float(training_value(training_cfg, "early_stopping_min_delta", model_prefix, training_cfg.get("early_stopping_min_delta", 0.0)))
    warmup_epochs = int(
        training_value(
            training_cfg,
            "early_stopping_warmup_epochs",
            model_prefix,
            training_cfg.get("early_stopping_warmup_epochs", 0),
        )
    )
    grad_clip_norm = training_value(training_cfg, "grad_clip_norm", model_prefix, training_cfg.get("grad_clip_norm"))
    configured_threshold = training_value(training_cfg, "decision_threshold", model_prefix, None)
    checkpoint_selection_metric = str(
        training_value(
            training_cfg,
            "checkpoint_selection_metric",
            model_prefix,
            training_cfg.get("checkpoint_selection_metric", "val_f1"),
        )
    )

    best_checkpoint_rank: tuple[float, ...] | None = None
    best_checkpoint_score = -np.inf
    best_checkpoint_raw_value = float("nan")
    best_selection_score = -np.inf
    best_val_metrics: dict[str, float | int] | None = None
    best_val_loss = float("inf")
    best_epoch = -1
    best_threshold = 0.5
    best_state = None
    history: list[dict] = []
    warnings_list: list[str] = []
    stale_epochs = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            features, labels = batch[:2]
            features = features.to(device=device, dtype=torch_module.float32)
            labels = labels.to(device=device, dtype=torch_module.float32)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm is not None:
                torch_module.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss, val_prob, val_labels = evaluate_model(model, val_loader, criterion, torch_module)
        val_search = search_threshold_with_policy(
            val_prob,
            val_labels,
            num_points=int(training_cfg.get("threshold_search_points", 200)),
            selection_policy=selection_policy,
            target_recall=target_recall,
            fbeta_beta=fbeta_beta,
        )
        val_threshold = float(val_search["threshold"])
        val_metrics = val_search["metrics"]
        val_selection_score = float(val_search["selection_score"])
        checkpoint_rank, checkpoint_raw_value, checkpoint_metric_name = checkpoint_rank_from_epoch(
            checkpoint_selection_metric,
            val_loss,
            val_metrics,
            val_selection_score,
        )
        checkpoint_score = float(checkpoint_rank[0])

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_f2": float(val_metrics["f2"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_false_positive_rate": float(val_metrics["false_positive_rate"]),
            "val_false_negative_rate": float(val_metrics["false_negative_rate"]),
            "val_threshold": val_threshold,
            "val_selection_score": val_selection_score,
            "checkpoint_selection_metric": checkpoint_metric_name,
            "checkpoint_selection_value": checkpoint_raw_value,
            "checkpoint_tie_breaker": "lower_val_loss" if checkpoint_metric_name == "val_f1" else None,
            "selection_policy": selection_policy,
            "target_recall": target_recall,
            "target_recall_satisfied": bool(float(val_metrics["recall"]) >= float(target_recall)),
            "target_recall_candidate_exists": bool(val_search.get("target_recall_satisfied", False)),
            "operating_threshold": float(configured_threshold) if configured_threshold is not None else val_threshold,
        }
        if float(val_metrics["recall"]) == 0.0:
            warning = f"WARNING: validation recall is 0 for {model_name} at epoch {epoch}"
            warnings.warn(warning)
            warnings_list.append(warning)
        if len(history) > 0 and train_loss < float(history[-1]["train_loss"]) and val_loss > float(history[-1]["val_loss"]):
            warning = (
                f"WARNING: possible overfitting signal for {model_name} at epoch {epoch}: "
                "train_loss decreased while val_loss increased. This is a val_loss warning only; "
                "checkpoint selection follows validation F1 unless checkpoint_selection_metric is changed."
            )
            warnings.warn(warning)
            warnings_list.append(warning)
        history.append(epoch_log)
        print(json.dumps({"model": model_name, **epoch_log}))

        if checkpoint_rank_is_better(checkpoint_rank, best_checkpoint_rank, min_delta):
            best_checkpoint_rank = checkpoint_rank
            best_checkpoint_score = checkpoint_score
            best_checkpoint_raw_value = checkpoint_raw_value
            best_selection_score = val_selection_score
            best_val_metrics = {key: float(value) for key, value in val_metrics.items() if isinstance(value, (int, float))}
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_threshold = val_threshold
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if epoch >= warmup_epochs and stale_epochs >= patience:
                print(f"Early stopping triggered for {model_name} at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"Training {model_name} completed without a valid checkpoint")
    if best_val_metrics is None:
        raise RuntimeError(f"Training {model_name} completed without validation metrics")

    decision_threshold = float(configured_threshold) if configured_threshold is not None else float(best_threshold)
    threshold_metric_name = threshold_selection_metric_name(selection_policy)
    threshold_label_basis = label_basis_for_threshold_search(training_cfg)
    checkpoint = {
        "model_type": "binary",
        "model_arch": model_arch,
        "state_dict": best_state,
        "input_dim": int(getattr(model, "input_dim", len(feature_names))),
        "decision_threshold": decision_threshold,
        "decision_threshold_value": decision_threshold,
        "validation_selected_threshold": float(best_threshold),
        "validation_best_f1_threshold": float(best_threshold),
        "threshold_selection_metric": threshold_metric_name,
        "threshold_selected_on": "selection_label_validation",
        "checkpoint_selected_on": "selection_label_validation",
        "label_basis_for_threshold_search": threshold_label_basis,
        "label_basis_for_original_evaluation": label_basis_for_original_evaluation(),
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        "feature_names": feature_names,
        "feature_mode": str(dataset_cfg.get("feature_mode", "original_42")),
        "use_delta_features": bool(dataset_cfg["use_delta_features"]),
        "residual_expected_torque_mode": "command_total",
        "residual_offset_policy": "episode_initial_no_contact_mean",
        "tau_ext_input_policy": "label_only_never_feature",
        "window_length": int(dataset_cfg["window_length"]),
        "stride": int(dataset_cfg["stride"]),
        "best_epoch": int(best_epoch),
        "best_epoch_selection_label": int(best_epoch),
        "best_val_precision": float(best_val_metrics["precision"]),
        "best_val_recall": float(best_val_metrics["recall"]),
        "best_val_f1": float(best_val_metrics["f1"]),
        "best_val_f2": float(best_val_metrics["f2"]),
        "best_val_loss": float(best_val_loss),
        "best_val_precision_selection_label": float(best_val_metrics["precision"]),
        "best_val_recall_selection_label": float(best_val_metrics["recall"]),
        "best_val_f1_selection_label": float(best_val_metrics["f1"]),
        "best_val_loss_selection_label": float(best_val_loss),
        "checkpoint_selection_metric": checkpoint_metric_name,
        "best_checkpoint_selection_value": float(best_checkpoint_raw_value),
        "best_selection_score": float(best_selection_score),
        "threshold_selection_policy": selection_policy,
        "target_recall": float(target_recall),
        "fbeta_beta": float(fbeta_beta),
        "training_device": str(device),
        "optimizer": optimizer_name,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "pos_weight": float(pos_weight_value),
        "early_stopping_min_delta": float(min_delta),
        "early_stopping_warmup_epochs": int(warmup_epochs),
        "stage": stage_name,
        "label_delay_ms": float(training_cfg.get("label_delay_ms", 0.0) or 0.0),
        "transition_exclusion_ms": float(training_cfg.get("transition_exclusion_ms", 0.0) or 0.0),
        "exclude_transition_val": bool(training_cfg.get("exclude_transition_val", False)),
        "selection_label_basis": threshold_label_basis,
        "label_basis_for_training": threshold_label_basis,
        "label_basis_for_checkpoint_selection": threshold_label_basis,
    }
    if extra_checkpoint_fields:
        checkpoint.update(extra_checkpoint_fields)

    train_log_payload = {
        "history": history,
        "best_epoch": best_epoch,
        "best_epoch_selection_label": int(best_epoch),
        "best_val_precision": float(best_val_metrics["precision"]),
        "best_val_recall": float(best_val_metrics["recall"]),
        "best_val_f1": float(best_val_metrics["f1"]),
        "best_val_f2": float(best_val_metrics["f2"]),
        "best_val_loss": float(best_val_loss),
        "best_val_precision_selection_label": float(best_val_metrics["precision"]),
        "best_val_recall_selection_label": float(best_val_metrics["recall"]),
        "best_val_f1_selection_label": float(best_val_metrics["f1"]),
        "best_val_loss_selection_label": float(best_val_loss),
        "best_selection_score": float(best_selection_score),
        "checkpoint_selection_metric": checkpoint_metric_name,
        "checkpoint_selected_on": "selection_label_validation",
        "best_checkpoint_selection_value": float(best_checkpoint_raw_value),
        "best_probability_threshold": best_threshold,
        "operating_probability_threshold": decision_threshold,
        "decision_threshold_value": decision_threshold,
        "threshold_selection_metric": threshold_metric_name,
        "threshold_selected_on": "selection_label_validation",
        "label_basis_for_threshold_search": threshold_label_basis,
        "label_basis_for_original_evaluation": label_basis_for_original_evaluation(),
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        "pos_weight": pos_weight_value,
        "threshold_selection_policy": selection_policy,
        "target_recall": float(target_recall),
        "fbeta_beta": float(fbeta_beta),
        "optimizer": optimizer_name,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "early_stopping_min_delta": float(min_delta),
        "early_stopping_warmup_epochs": int(warmup_epochs),
        "training_device": str(device),
        "model_type": "binary",
        "model_arch": model_arch,
        "feature_mode": str(dataset_cfg.get("feature_mode", "original_42")),
        "feature_names": feature_names,
        "residual_expected_torque_mode": "command_total",
        "residual_offset_policy": "episode_initial_no_contact_mean",
        "tau_ext_input_policy": "label_only_never_feature",
        "warnings": warnings_list,
        "stage": stage_name,
        "label_delay_ms": float(training_cfg.get("label_delay_ms", 0.0) or 0.0),
        "transition_exclusion_ms": float(training_cfg.get("transition_exclusion_ms", 0.0) or 0.0),
        "exclude_transition_val": bool(training_cfg.get("exclude_transition_val", False)),
        "selection_label_basis": threshold_label_basis,
        "label_basis_for_training": threshold_label_basis,
        "label_basis_for_checkpoint_selection": threshold_label_basis,
    }
    return checkpoint, train_log_payload, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage override.")
    parser.add_argument(
        "--train-final-trainval",
        action="store_true",
        help=(
            "After the normal train/val experiment, retrain deployment-only MLP/GRU models on train+val "
            "for the validation-F1 best epoch count and save them as separate artifacts."
        ),
    )
    parser.add_argument("--label-delay-ms", type=float, default=None, help="Shift train/val selection labels later by this many ms.")
    parser.add_argument(
        "--transition-exclusion-ms",
        type=float,
        default=None,
        help="Exclude train windows whose end index falls within this many ms of command-label transitions.",
    )
    parser.add_argument(
        "--exclude-transition-val",
        action="store_true",
        help="Also exclude validation windows around transitions for model/threshold selection.",
    )
    parser.add_argument(
        "--ablation-tag",
        default=None,
        help="Optional ablation artifact tag. Results are saved under outputs/<stage>/ablations/<tag>/.",
    )
    parser.add_argument(
        "--reuse-gru-checkpoint",
        default=None,
        help=(
            "Optional path to an existing GRU checkpoint. When set, train_detectors.py trains the MLP baseline "
            "but reuses this GRU checkpoint instead of retraining GRU."
        ),
    )
    parser.add_argument(
        "--reuse-gru-train-log",
        default=None,
        help="Optional train_log.json path that corresponds to --reuse-gru-checkpoint.",
    )
    parser.add_argument(
        "--real-no-contact-csv",
        action="append",
        default=[],
        help="Real robot no-contact CSV to append to the training split as hard-negative windows. Can be repeated.",
    )
    parser.add_argument(
        "--real-no-contact-val-csv",
        action="append",
        default=[],
        help="Optional real no-contact CSV to append to validation as hard-negative windows. Can be repeated.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    training_cfg = config.setdefault("training", {})
    if args.label_delay_ms is not None:
        training_cfg["label_delay_ms"] = float(args.label_delay_ms)
    else:
        training_cfg["label_delay_ms"] = float(training_cfg.get("label_delay_ms", 0.0) or 0.0)
    if args.transition_exclusion_ms is not None:
        training_cfg["transition_exclusion_ms"] = float(args.transition_exclusion_ms)
    else:
        training_cfg["transition_exclusion_ms"] = float(training_cfg.get("transition_exclusion_ms", 0.0) or 0.0)
    if args.exclude_transition_val:
        training_cfg["exclude_transition_val"] = True
    else:
        training_cfg["exclude_transition_val"] = bool(training_cfg.get("exclude_transition_val", False))
    training_cfg["isolate_model_random_seed"] = bool(training_cfg.get("isolate_model_random_seed", True))
    reuse_gru_checkpoint_path = args.reuse_gru_checkpoint or training_cfg.get("reuse_gru_checkpoint_path")
    reuse_gru_train_log_path = args.reuse_gru_train_log or training_cfg.get("reuse_gru_train_log_path")

    ablation_tag = args.ablation_tag
    if ablation_tag is None and (
        float(training_cfg["label_delay_ms"]) > 0.0 or float(training_cfg["transition_exclusion_ms"]) > 0.0
    ):
        ablation_tag = default_ablation_tag(training_cfg["label_delay_ms"], training_cfg["transition_exclusion_ms"])

    set_global_seed(int(config.get("seed", 42)))
    base_dirs, out_dirs = resolve_artifact_dirs(config, ablation_tag)
    if ablation_tag is not None:
        config["ablation_tag"] = str(ablation_tag)
        config["ablation_output_dir"] = str(out_dirs["root"])
    save_config_yaml(out_dirs["root"] / "experiment_config_used.yaml", config)

    train_path = base_dirs["datasets"] / "sim_train.npz"
    val_path = base_dirs["datasets"] / "sim_val.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Simulation datasets are missing. Run generate_sim_dataset.py before training detectors."
        )

    dataset_cfg = config["dataset"]
    feature_mode = str(dataset_cfg.get("feature_mode", "original_42"))
    dt = float(config["simulation"]["dt"])
    label_delay_steps = ms_to_steps(float(training_cfg["label_delay_ms"]), dt)
    transition_exclusion_steps = ms_to_steps(float(training_cfg["transition_exclusion_ms"]), dt)
    train_bundle = ContactWindowDataset.from_npz(
        train_path,
        window_length=int(dataset_cfg["window_length"]),
        stride=int(dataset_cfg["stride"]),
        use_delta_features=bool(dataset_cfg["use_delta_features"]),
        fit_scaler=True,
        label_delay_steps=label_delay_steps,
        transition_exclusion_steps=transition_exclusion_steps,
        exclude_transition_windows=transition_exclusion_steps > 0,
        feature_mode=feature_mode,
    )
    scaler = train_bundle.scaler
    scaler_path = out_dirs["models"] / "scaler.pkl"
    scaler.save(scaler_path)

    val_bundle = ContactWindowDataset.from_npz(
        val_path,
        window_length=int(dataset_cfg["window_length"]),
        stride=int(dataset_cfg["stride"]),
        use_delta_features=bool(dataset_cfg["use_delta_features"]),
        scaler=scaler,
        label_delay_steps=label_delay_steps,
        transition_exclusion_steps=transition_exclusion_steps,
        exclude_transition_windows=bool(training_cfg["exclude_transition_val"]) and transition_exclusion_steps > 0,
        feature_mode=feature_mode,
    )

    train_step_dataset = train_bundle.dataset.aligned_single_step_dataset()
    val_step_dataset = val_bundle.dataset.aligned_single_step_dataset()

    real_train_window_datasets = [
        real_no_contact_dataset_from_csv(
            path,
            config,
            scaler,
            window_length=int(dataset_cfg["window_length"]),
            stride=int(dataset_cfg["stride"]),
            use_delta_features=bool(dataset_cfg["use_delta_features"]),
            feature_mode=feature_mode,
        )
        for path in args.real_no_contact_csv
    ]
    real_val_window_datasets = [
        real_no_contact_dataset_from_csv(
            path,
            config,
            scaler,
            window_length=int(dataset_cfg["window_length"]),
            stride=int(dataset_cfg["stride"]),
            use_delta_features=bool(dataset_cfg["use_delta_features"]),
            feature_mode=feature_mode,
        )
        for path in args.real_no_contact_val_csv
    ]
    real_train_step_datasets = [dataset.aligned_single_step_dataset() for dataset in real_train_window_datasets]
    real_val_step_datasets = [dataset.aligned_single_step_dataset() for dataset in real_val_window_datasets]

    train_window_indices = train_bundle.dataset.end_indices
    if not np.array_equal(train_window_indices, train_step_dataset.selected_indices):
        raise RuntimeError("MLP alignment mismatch: train single-step indices do not match GRU end_indices")
    if not np.array_equal(val_bundle.dataset.end_indices, val_step_dataset.selected_indices):
        raise RuntimeError("MLP alignment mismatch: val single-step indices do not match GRU end_indices")

    selection_policy = str(config["training"].get("threshold_selection_policy", "f1"))
    target_recall = float(config["training"].get("target_recall", 0.85))
    fbeta_beta = float(config["training"].get("fbeta_beta", 2.0))

    val_raw = load_npz_dataset(val_path)
    val_score, threshold_meta = threshold_score_from_data(val_raw, val_bundle.dataset.end_indices, config)
    val_label = val_bundle.dataset.labels_for_windows()
    baseline_result = search_threshold_with_policy(
        val_score,
        val_label,
        num_points=int(config["training"].get("threshold_search_points", 200)),
        selection_policy=selection_policy,
        target_recall=target_recall,
        fbeta_beta=fbeta_beta,
    )
    threshold_payload = {
        **threshold_meta,
        "gamma": float(baseline_result["threshold"]),
        "val_precision": float(baseline_result["metrics"]["precision"]),
        "val_recall": float(baseline_result["metrics"]["recall"]),
        "val_f1": float(baseline_result["metrics"]["f1"]),
        "val_f2": float(baseline_result["metrics"]["f2"]),
        "val_false_positive_rate": float(baseline_result["metrics"]["false_positive_rate"]),
        "val_false_negative_rate": float(baseline_result["metrics"]["false_negative_rate"]),
        "selection_policy": baseline_result["selection_policy"],
        "selection_score": float(baseline_result["selection_score"]),
        "target_recall": float(baseline_result["target_recall"]),
        "target_recall_satisfied": bool(baseline_result["target_recall_satisfied"]),
        "fbeta_beta": float(baseline_result["fbeta_beta"]),
        "validation_metrics": baseline_result["metrics"],
        "label_delay_ms": float(training_cfg["label_delay_ms"]),
        "transition_exclusion_ms": float(training_cfg["transition_exclusion_ms"]),
        "selection_label_basis": label_basis_for_threshold_search(training_cfg),
        "label_basis_for_training": label_basis_for_threshold_search(training_cfg),
        "label_basis_for_checkpoint_selection": label_basis_for_threshold_search(training_cfg),
        "threshold_selected_on": "selection_label_validation",
        "checkpoint_selected_on": "selection_label_validation",
        "label_basis_for_threshold_search": label_basis_for_threshold_search(training_cfg),
        "label_basis_for_original_evaluation": label_basis_for_original_evaluation(),
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
    }
    save_json(out_dirs["models"] / "threshold.json", threshold_payload)

    torch, DataLoader = _import_torch()
    from models import GRUDetector, MLPDetector
    isolate_model_random_seed = bool(config["training"].get("isolate_model_random_seed", True))
    mlp_seed = model_seed(config, config["training"], "mlp")
    gru_seed = model_seed(config, config["training"], "gru")

    train_label_chunks = [train_bundle.dataset.labels_for_windows()]
    train_label_chunks.extend(dataset.labels_for_windows() for dataset in real_train_window_datasets)
    train_labels = np.concatenate(train_label_chunks, axis=0)
    pos_count = int(np.sum(train_labels == 1))
    neg_count = int(np.sum(train_labels == 0))
    if config["training"].get("pos_weight_override") is not None:
        pos_weight_value = float(config["training"]["pos_weight_override"])
    else:
        pos_weight_value = float(neg_count / max(pos_count, 1))

    train_window_dataset_for_loader = (
        torch.utils.data.ConcatDataset([train_bundle.dataset, *real_train_window_datasets])
        if real_train_window_datasets
        else train_bundle.dataset
    )
    val_window_dataset_for_loader = (
        torch.utils.data.ConcatDataset([val_bundle.dataset, *real_val_window_datasets])
        if real_val_window_datasets
        else val_bundle.dataset
    )
    train_step_dataset_for_loader = (
        torch.utils.data.ConcatDataset([train_step_dataset, *real_train_step_datasets])
        if real_train_step_datasets
        else train_step_dataset
    )
    val_step_dataset_for_loader = (
        torch.utils.data.ConcatDataset([val_step_dataset, *real_val_step_datasets])
        if real_val_step_datasets
        else val_step_dataset
    )

    train_window_loader = DataLoader(
        train_window_dataset_for_loader,
        batch_size=int(training_value(config["training"], "batch_size", "gru", config["training"]["batch_size"])),
        shuffle=True,
        num_workers=int(training_value(config["training"], "num_workers", "gru", config["training"].get("num_workers", 0))),
    )
    val_window_loader = DataLoader(
        val_window_dataset_for_loader,
        batch_size=int(training_value(config["training"], "batch_size", "gru", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(training_value(config["training"], "num_workers", "gru", config["training"].get("num_workers", 0))),
    )
    train_step_loader = DataLoader(
        train_step_dataset_for_loader,
        batch_size=int(training_value(config["training"], "batch_size", "mlp", config["training"]["batch_size"])),
        shuffle=True,
        num_workers=int(training_value(config["training"], "num_workers", "mlp", config["training"].get("num_workers", 0))),
    )
    val_step_loader = DataLoader(
        val_step_dataset_for_loader,
        batch_size=int(training_value(config["training"], "batch_size", "mlp", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(training_value(config["training"], "num_workers", "mlp", config["training"].get("num_workers", 0))),
    )
    val_step_loader_sim_only = DataLoader(
        val_step_dataset,
        batch_size=int(training_value(config["training"], "batch_size", "mlp", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(training_value(config["training"], "num_workers", "mlp", config["training"].get("num_workers", 0))),
    )
    val_window_loader_sim_only = DataLoader(
        val_bundle.dataset,
        batch_size=int(training_value(config["training"], "batch_size", "gru", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(training_value(config["training"], "num_workers", "gru", config["training"].get("num_workers", 0))),
    )

    if isolate_model_random_seed:
        set_global_seed(mlp_seed)
    mlp_model = MLPDetector(
        input_dim=train_step_dataset.input_dim,
        hidden_dim=int(training_value(config["training"], "hidden_dim", "mlp", config["training"]["hidden_dim"])),
        num_layers=int(training_value(config["training"], "num_layers", "mlp", config["training"].get("mlp_num_layers", 2))),
        dropout=float(training_value(config["training"], "dropout", "mlp", config["training"]["dropout"])),
    )
    mlp_model.input_dim = train_step_dataset.input_dim  # type: ignore[attr-defined]
    mlp_checkpoint, mlp_train_log, mlp_history = train_binary_model(
        model_name="MLP",
        model_arch="mlp",
        model_prefix="mlp",
        model=mlp_model,
        train_loader=train_step_loader,
        val_loader=val_step_loader,
        torch_module=torch,
        training_cfg=config["training"],
        dataset_cfg=dataset_cfg,
        feature_names=train_bundle.feature_names,
        pos_weight_value=pos_weight_value,
        selection_policy=selection_policy,
        target_recall=target_recall,
        fbeta_beta=fbeta_beta,
        stage_name=config["experiment_stage"],
        extra_checkpoint_fields={
            "hidden_dim": int(training_value(config["training"], "hidden_dim", "mlp", config["training"]["hidden_dim"])),
            "num_layers": int(training_value(config["training"], "num_layers", "mlp", config["training"].get("mlp_num_layers", 2))),
            "dropout": float(training_value(config["training"], "dropout", "mlp", config["training"]["dropout"])),
            "input_mode": "single_step_aligned_to_window_end",
            "alignment_source": "window_end_indices",
            "model_seed": int(mlp_seed),
            "isolate_model_random_seed": isolate_model_random_seed,
        },
    )
    mlp_train_log["train_dataset_samples"] = int(len(train_step_dataset))
    mlp_train_log["val_dataset_samples"] = int(len(val_step_dataset))
    mlp_train_log["real_no_contact_train_samples"] = int(sum(len(dataset) for dataset in real_train_step_datasets))
    mlp_train_log["real_no_contact_val_samples"] = int(sum(len(dataset) for dataset in real_val_step_datasets))
    mlp_train_log["train_excluded_transition_windows"] = int(train_bundle.dataset.excluded_window_count)
    mlp_train_log["val_excluded_transition_windows"] = int(val_bundle.dataset.excluded_window_count)
    mlp_train_log["model_seed"] = int(mlp_seed)
    mlp_train_log["isolate_model_random_seed"] = isolate_model_random_seed
    torch.save(mlp_checkpoint, out_dirs["models"] / "mlp_detector.pt")
    save_json(out_dirs["metrics"] / "mlp_train_log.json", mlp_train_log)
    save_training_curve(out_dirs["figures"] / "mlp_training_curve.png", mlp_history)

    if reuse_gru_checkpoint_path:
        reuse_path = Path(str(reuse_gru_checkpoint_path)).expanduser().resolve()
        if not reuse_path.exists():
            raise FileNotFoundError(f"reuse_gru_checkpoint_path does not exist: {reuse_path}")
        shutil.copy2(reuse_path, out_dirs["models"] / "gru_detector.pt")
        gru_checkpoint = torch.load(reuse_path, map_location="cpu")
        gru_model = GRUDetector(
            input_dim=int(gru_checkpoint.get("input_dim", train_bundle.dataset.input_dim)),
            hidden_dim=int(gru_checkpoint["hidden_dim"]),
            num_layers=int(gru_checkpoint["num_layers"]),
            dropout=float(gru_checkpoint["dropout"]),
            bidirectional=bool(gru_checkpoint.get("bidirectional", False)),
        )
        gru_model.input_dim = int(gru_checkpoint.get("input_dim", train_bundle.dataset.input_dim))  # type: ignore[attr-defined]
        gru_model.load_state_dict(gru_checkpoint["state_dict"])
        gru_history = []
        if reuse_gru_train_log_path:
            log_path = Path(str(reuse_gru_train_log_path)).expanduser().resolve()
            if not log_path.exists():
                raise FileNotFoundError(f"reuse_gru_train_log_path does not exist: {log_path}")
            gru_train_log = load_json(log_path)
        else:
            gru_train_log = {
                "history": [],
                "best_epoch": int(gru_checkpoint.get("best_epoch", -1)),
                "best_val_precision": float(gru_checkpoint.get("best_val_precision", float("nan"))),
                "best_val_recall": float(gru_checkpoint.get("best_val_recall", float("nan"))),
                "best_val_f1": float(gru_checkpoint.get("best_val_f1", float("nan"))),
                "best_val_f2": float(gru_checkpoint.get("best_val_f2", float("nan"))),
                "pos_weight": float(gru_checkpoint.get("pos_weight", pos_weight_value)),
                "stage": config["experiment_stage"],
            }
        gru_train_log["reused_gru_checkpoint"] = True
        gru_train_log["reused_gru_checkpoint_path"] = str(reuse_path)
        if reuse_gru_train_log_path:
            gru_train_log["reused_gru_train_log_path"] = str(Path(str(reuse_gru_train_log_path)).expanduser().resolve())
        gru_train_log["note"] = (
            "GRU was not retrained in this run. The legacy GRU checkpoint was reused so that adding the MLP "
            "baseline does not change the original GRU result."
        )
        save_json(out_dirs["metrics"] / "train_log.json", gru_train_log)
    else:
        if isolate_model_random_seed:
            set_global_seed(gru_seed)
        gru_model = GRUDetector(
            input_dim=train_bundle.dataset.input_dim,
            hidden_dim=int(training_value(config["training"], "hidden_dim", "gru", config["training"]["hidden_dim"])),
            num_layers=int(training_value(config["training"], "num_layers", "gru", config["training"]["num_layers"])),
            dropout=float(training_value(config["training"], "dropout", "gru", config["training"]["dropout"])),
            bidirectional=False,
        )
        gru_model.input_dim = train_bundle.dataset.input_dim  # type: ignore[attr-defined]
        gru_checkpoint, gru_train_log, gru_history = train_binary_model(
            model_name="GRU",
            model_arch="gru",
            model_prefix="gru",
            model=gru_model,
            train_loader=train_window_loader,
            val_loader=val_window_loader,
            torch_module=torch,
            training_cfg=config["training"],
            dataset_cfg=dataset_cfg,
            feature_names=train_bundle.feature_names,
            pos_weight_value=pos_weight_value,
            selection_policy=selection_policy,
            target_recall=target_recall,
            fbeta_beta=fbeta_beta,
            stage_name=config["experiment_stage"],
            extra_checkpoint_fields={
                "hidden_dim": int(training_value(config["training"], "hidden_dim", "gru", config["training"]["hidden_dim"])),
                "num_layers": int(training_value(config["training"], "num_layers", "gru", config["training"]["num_layers"])),
                "dropout": float(training_value(config["training"], "dropout", "gru", config["training"]["dropout"])),
                "bidirectional": False,
                "model_seed": int(gru_seed),
                "isolate_model_random_seed": isolate_model_random_seed,
            },
        )
        gru_train_log["train_dataset_windows"] = int(len(train_bundle.dataset))
        gru_train_log["val_dataset_windows"] = int(len(val_bundle.dataset))
        gru_train_log["real_no_contact_train_windows"] = int(sum(len(dataset) for dataset in real_train_window_datasets))
        gru_train_log["real_no_contact_val_windows"] = int(sum(len(dataset) for dataset in real_val_window_datasets))
        gru_train_log["train_excluded_transition_windows"] = int(train_bundle.dataset.excluded_window_count)
        gru_train_log["val_excluded_transition_windows"] = int(val_bundle.dataset.excluded_window_count)
        gru_train_log["model_seed"] = int(gru_seed)
        gru_train_log["isolate_model_random_seed"] = isolate_model_random_seed
        torch.save(gru_checkpoint, out_dirs["models"] / "gru_detector.pt")
        save_json(out_dirs["metrics"] / "train_log.json", gru_train_log)
        save_training_curve(out_dirs["figures"] / "training_curve.png", gru_history)

    from evaluate_detectors import (
        binary_log_loss_from_probability,
        event_latency_metrics,
        metrics_with_delay,
        run_model_inference,
    )

    eval_cfg = config.get("evaluation", {})
    event_consecutive_samples = int(eval_cfg.get("event_detection_consecutive_samples", 3))
    detection_margin_ms = float(eval_cfg.get("detection_margin_ms", 50.0))
    val_original_labels = val_bundle.dataset.original_labels_for_windows().astype(np.int64)
    val_time = val_bundle.dataset.time_for_windows(val_raw["time"])
    val_episode_id = val_bundle.dataset.episodes_for_windows().astype(np.int64)
    val_threshold_pred_original = (val_score >= float(threshold_payload["gamma"])).astype(np.int64)
    val_threshold_original_sample = metrics_with_delay(
        val_original_labels,
        val_threshold_pred_original,
        val_time,
        val_episode_id,
    )
    val_threshold_original_sample["loss"] = None

    eval_device = select_torch_device(torch, config["training"].get("device", "auto"))
    mlp_model.load_state_dict(mlp_checkpoint["state_dict"])
    mlp_model.to(eval_device)
    mlp_val_prob_original = run_model_inference(mlp_model, val_step_loader_sim_only, torch)
    mlp_val_pred_original = (
        mlp_val_prob_original >= float(mlp_checkpoint.get("decision_threshold", 0.5))
    ).astype(np.int64)
    mlp_validation_original_sample = metrics_with_delay(
        val_original_labels,
        mlp_val_pred_original,
        val_time,
        val_episode_id,
    )
    mlp_validation_original_sample["loss"] = binary_log_loss_from_probability(
        val_original_labels,
        mlp_val_prob_original,
    )
    mlp_validation_original_event = event_latency_metrics(
        val_raw,
        val_bundle.dataset.end_indices,
        mlp_val_prob_original,
        float(mlp_checkpoint.get("decision_threshold", 0.5)),
        event_consecutive_samples,
        detection_margin_ms,
    )

    gru_model.load_state_dict(gru_checkpoint["state_dict"])
    gru_model.to(eval_device)
    gru_val_prob_original = run_model_inference(gru_model, val_window_loader_sim_only, torch)
    gru_val_pred_original = (
        gru_val_prob_original >= float(gru_checkpoint.get("decision_threshold", 0.5))
    ).astype(np.int64)
    gru_validation_original_sample = metrics_with_delay(
        val_original_labels,
        gru_val_pred_original,
        val_time,
        val_episode_id,
    )
    gru_validation_original_sample["loss"] = binary_log_loss_from_probability(
        val_original_labels,
        gru_val_prob_original,
    )
    gru_validation_original_event = event_latency_metrics(
        val_raw,
        val_bundle.dataset.end_indices,
        gru_val_prob_original,
        float(gru_checkpoint.get("decision_threshold", 0.5)),
        event_consecutive_samples,
        detection_margin_ms,
    )
    validation_original_summary = {
        "split": "sim_val",
        "sample_metric_label_basis": label_basis_for_original_evaluation(),
        "event_metric_label_basis": label_basis_for_original_evaluation(),
        "threshold_selected_on": "selection_label_validation",
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        "event_detection_consecutive_samples": event_consecutive_samples,
        "detection_margin_ms": detection_margin_ms,
        "threshold": {
            "sample_level": val_threshold_original_sample,
            "event_level": event_latency_metrics(
                val_raw,
                val_bundle.dataset.end_indices,
                val_score,
                float(threshold_payload["gamma"]),
                event_consecutive_samples,
                detection_margin_ms,
            ),
        },
        "mlp": {
            "sample_level": mlp_validation_original_sample,
            "event_level": mlp_validation_original_event,
        },
        "gru": {
            "sample_level": gru_validation_original_sample,
            "event_level": gru_validation_original_event,
        },
    }

    checkpoint_summary = {
        "stage": config["experiment_stage"],
        "ablation_tag": ablation_tag,
        "artifact_root": str(out_dirs["root"]),
        "dataset_root": str(base_dirs["root"]),
        "feature_mode": str(dataset_cfg.get("feature_mode", "original_42")),
        "feature_names": train_bundle.feature_names,
        "residual_expected_torque_mode": "command_total",
        "residual_offset_policy": "episode_initial_no_contact_mean",
        "tau_ext_input_policy": "label_only_never_feature",
        "real_no_contact_train_csv": [str(Path(path).expanduser().resolve()) for path in args.real_no_contact_csv],
        "real_no_contact_val_csv": [str(Path(path).expanduser().resolve()) for path in args.real_no_contact_val_csv],
        "label_delay_ms": float(training_cfg["label_delay_ms"]),
        "label_delay_steps": int(label_delay_steps),
        "transition_exclusion_ms": float(training_cfg["transition_exclusion_ms"]),
        "transition_exclusion_steps": int(transition_exclusion_steps),
        "exclude_transition_val": bool(training_cfg["exclude_transition_val"]),
        "selection_label_basis": label_basis_for_threshold_search(config["training"]),
        "label_basis_for_training": label_basis_for_threshold_search(config["training"]),
        "label_basis_for_checkpoint_selection": label_basis_for_threshold_search(config["training"]),
        "train_excluded_transition_windows": int(train_bundle.dataset.excluded_window_count),
        "val_excluded_transition_windows": int(val_bundle.dataset.excluded_window_count),
        "random_seed_policy": (
            "MLP and GRU random seeds are reset independently before model initialization and training. "
            "This prevents adding the MLP baseline from changing the GRU initialization or shuffled batch order."
        ),
        "isolate_model_random_seed": isolate_model_random_seed,
        "mlp_seed": int(mlp_seed),
        "gru_seed": int(gru_seed),
        "reuse_gru_checkpoint": bool(reuse_gru_checkpoint_path),
        "reuse_gru_checkpoint_path": str(Path(str(reuse_gru_checkpoint_path)).expanduser().resolve())
        if reuse_gru_checkpoint_path
        else None,
        "reuse_gru_train_log_path": str(Path(str(reuse_gru_train_log_path)).expanduser().resolve())
        if reuse_gru_train_log_path
        else None,
        "checkpoint_selection_rule": (
            "MLP/GRU best checkpoints are selected on the validation split by the highest validation F1-score. "
            "If validation F1 ties exactly, lower validation loss is used as the tie-breaker."
        ),
        "threshold_selection_rule": (
            "Decision thresholds are selected on the validation split using training.threshold_selection_policy. "
            "The test split is not used for threshold selection."
        ),
        "threshold_application_rule": (
            "For label-delay ablations, each run selects its checkpoint and decision threshold on that run's "
            "selection-label validation F1. The same threshold is then reused for original-command-label "
            "validation/test metrics so label policies can be compared without reselecting thresholds."
        ),
        "threshold_selected_on": "selection_label_validation",
        "checkpoint_selected_on": "selection_label_validation",
        "label_basis_for_threshold_search": label_basis_for_threshold_search(config["training"]),
        "label_basis_for_original_evaluation": label_basis_for_original_evaluation(),
        "threshold_applied_to_original_label_metrics": True,
        "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
        "test_set_usage": "test split is not used for training, checkpoint selection, or threshold selection",
        "threshold": {
            "model_path": str(out_dirs["models"] / "threshold.json"),
            "gamma": float(threshold_payload["gamma"]),
            "decision_threshold_value": float(threshold_payload["gamma"]),
            "val_precision": float(threshold_payload["val_precision"]),
            "val_recall": float(threshold_payload["val_recall"]),
            "val_f1": float(threshold_payload["val_f1"]),
            "selection_policy": threshold_payload["selection_policy"],
            "threshold_selection_metric": threshold_selection_metric_name(str(threshold_payload["selection_policy"])),
            "threshold_selected_on": "selection_label_validation",
            "checkpoint_selected_on": "selection_label_validation",
            "selection_label_basis": label_basis_for_threshold_search(config["training"]),
            "label_basis_for_training": label_basis_for_threshold_search(config["training"]),
            "label_basis_for_checkpoint_selection": label_basis_for_threshold_search(config["training"]),
            "label_basis_for_threshold_search": label_basis_for_threshold_search(config["training"]),
            "label_basis_for_original_evaluation": label_basis_for_original_evaluation(),
            "threshold_applied_to_original_label_metrics": True,
            "original_label_metric_threshold_policy": "reuse_selection_label_threshold",
            "validation_original_sample_metrics": val_threshold_original_sample,
            "validation_original_event_metrics": validation_original_summary["threshold"]["event_level"],
        },
        "mlp": checkpoint_summary_entry("MLP", mlp_checkpoint, str(out_dirs["models"] / "mlp_detector.pt")),
        "gru": checkpoint_summary_entry("GRU", gru_checkpoint, str(out_dirs["models"] / "gru_detector.pt")),
        "validation_original_metrics": validation_original_summary,
    }
    checkpoint_summary["mlp"]["validation_original_sample_metrics"] = mlp_validation_original_sample
    checkpoint_summary["mlp"]["validation_original_event_metrics"] = mlp_validation_original_event
    checkpoint_summary["gru"]["validation_original_sample_metrics"] = gru_validation_original_sample
    checkpoint_summary["gru"]["validation_original_event_metrics"] = gru_validation_original_event
    save_json(out_dirs["metrics"] / "checkpoint_summary.json", checkpoint_summary)

    if args.train_final_trainval:
        # Re-seed before deployment-only retraining so the final artifacts are
        # reproducible and independent from the preceding validation experiment.
        set_global_seed(int(config.get("seed", 42)) + 1000)

        combined_window_dataset = torch.utils.data.ConcatDataset([train_bundle.dataset, val_bundle.dataset])
        combined_step_dataset = torch.utils.data.ConcatDataset([train_step_dataset, val_step_dataset])
        combined_labels = np.concatenate(
            [train_bundle.dataset.labels_for_windows(), val_bundle.dataset.labels_for_windows()],
            axis=0,
        )
        combined_pos_count = int(np.sum(combined_labels == 1))
        combined_neg_count = int(np.sum(combined_labels == 0))
        if config["training"].get("pos_weight_override") is not None:
            final_pos_weight_value = float(config["training"]["pos_weight_override"])
        else:
            final_pos_weight_value = float(combined_neg_count / max(combined_pos_count, 1))

        final_step_loader = DataLoader(
            combined_step_dataset,
            batch_size=int(training_value(config["training"], "batch_size", "mlp", config["training"]["batch_size"])),
            shuffle=True,
            num_workers=int(
                training_value(config["training"], "num_workers", "mlp", config["training"].get("num_workers", 0))
            ),
        )
        final_window_loader = DataLoader(
            combined_window_dataset,
            batch_size=int(training_value(config["training"], "batch_size", "gru", config["training"]["batch_size"])),
            shuffle=True,
            num_workers=int(
                training_value(config["training"], "num_workers", "gru", config["training"].get("num_workers", 0))
            ),
        )

        final_mlp_model = MLPDetector(
            input_dim=train_step_dataset.input_dim,
            hidden_dim=int(training_value(config["training"], "hidden_dim", "mlp", config["training"]["hidden_dim"])),
            num_layers=int(
                training_value(config["training"], "num_layers", "mlp", config["training"].get("mlp_num_layers", 2))
            ),
            dropout=float(training_value(config["training"], "dropout", "mlp", config["training"]["dropout"])),
        )
        final_mlp_model.input_dim = train_step_dataset.input_dim  # type: ignore[attr-defined]
        final_mlp_checkpoint, final_mlp_log = train_fixed_epoch_model(
            model_name="MLP",
            model_arch="mlp",
            model_prefix="mlp",
            model=final_mlp_model,
            train_loader=final_step_loader,
            torch_module=torch,
            training_cfg=config["training"],
            dataset_cfg=dataset_cfg,
            feature_names=train_bundle.feature_names,
            source_checkpoint=mlp_checkpoint,
            num_epochs=int(mlp_checkpoint["best_epoch"]),
            pos_weight_value=final_pos_weight_value,
            stage_name=config["experiment_stage"],
            extra_checkpoint_fields={
                "hidden_dim": int(training_value(config["training"], "hidden_dim", "mlp", config["training"]["hidden_dim"])),
                "num_layers": int(
                    training_value(config["training"], "num_layers", "mlp", config["training"].get("mlp_num_layers", 2))
                ),
                "dropout": float(training_value(config["training"], "dropout", "mlp", config["training"]["dropout"])),
                "input_mode": "single_step_aligned_to_window_end",
                "alignment_source": "window_end_indices",
            },
        )

        final_gru_model = GRUDetector(
            input_dim=train_bundle.dataset.input_dim,
            hidden_dim=int(training_value(config["training"], "hidden_dim", "gru", config["training"]["hidden_dim"])),
            num_layers=int(training_value(config["training"], "num_layers", "gru", config["training"]["num_layers"])),
            dropout=float(training_value(config["training"], "dropout", "gru", config["training"]["dropout"])),
            bidirectional=False,
        )
        final_gru_model.input_dim = train_bundle.dataset.input_dim  # type: ignore[attr-defined]
        final_gru_checkpoint, final_gru_log = train_fixed_epoch_model(
            model_name="GRU",
            model_arch="gru",
            model_prefix="gru",
            model=final_gru_model,
            train_loader=final_window_loader,
            torch_module=torch,
            training_cfg=config["training"],
            dataset_cfg=dataset_cfg,
            feature_names=train_bundle.feature_names,
            source_checkpoint=gru_checkpoint,
            num_epochs=int(gru_checkpoint["best_epoch"]),
            pos_weight_value=final_pos_weight_value,
            stage_name=config["experiment_stage"],
            extra_checkpoint_fields={
                "hidden_dim": int(training_value(config["training"], "hidden_dim", "gru", config["training"]["hidden_dim"])),
                "num_layers": int(training_value(config["training"], "num_layers", "gru", config["training"]["num_layers"])),
                "dropout": float(training_value(config["training"], "dropout", "gru", config["training"]["dropout"])),
                "bidirectional": False,
            },
        )

        final_mlp_path = out_dirs["models"] / "mlp_detector_trainval_final.pt"
        final_gru_path = out_dirs["models"] / "gru_detector_trainval_final.pt"
        torch.save(final_mlp_checkpoint, final_mlp_path)
        torch.save(final_gru_checkpoint, final_gru_path)
        final_summary = {
            "stage": config["experiment_stage"],
            "purpose": "deployment-only final retraining on train+val",
            "test_set_usage": "test split is not used; evaluate separately only as final performance check",
            "scaler": {
                "path": str(scaler_path),
                "fit_split": "train",
                "note": "The original train-fitted scaler is reused; scaler.pkl is not overwritten.",
            },
            "mlp": {
                **checkpoint_summary_entry("MLP", final_mlp_checkpoint, str(final_mlp_path)),
                "trainval_epochs": int(final_mlp_checkpoint["trainval_epochs"]),
                "source_checkpoint_best_epoch": int(final_mlp_checkpoint["source_checkpoint_best_epoch"]),
                "source_best_val_f1": float(final_mlp_checkpoint["source_best_val_f1"]),
                "train_log": final_mlp_log,
            },
            "gru": {
                **checkpoint_summary_entry("GRU", final_gru_checkpoint, str(final_gru_path)),
                "trainval_epochs": int(final_gru_checkpoint["trainval_epochs"]),
                "source_checkpoint_best_epoch": int(final_gru_checkpoint["source_checkpoint_best_epoch"]),
                "source_best_val_f1": float(final_gru_checkpoint["source_best_val_f1"]),
                "train_log": final_gru_log,
            },
        }
        save_json(out_dirs["metrics"] / "final_trainval_summary.json", final_summary)

    residual_summary = {
        "stage": config["experiment_stage"],
        "artifact_root": str(out_dirs["root"]),
        "feature_mode": str(dataset_cfg.get("feature_mode", "original_42")),
        "feature_names": train_bundle.feature_names,
        "input_dim": int(train_bundle.dataset.input_dim),
        "use_delta_features": bool(dataset_cfg["use_delta_features"]),
        "residual_expected_torque_mode": "command_total",
        "residual_definition": "tau_residual = tau_meas - tau_cmd = tau_meas - (kp*(q_des-q)+kd*(qdot_des-qdot)+tau_ff)",
        "residual_offset_policy": "episode_initial_no_contact_mean",
        "tau_ext_input_policy": "label_only_never_feature",
        "real_no_contact_train_csv": [str(Path(path).expanduser().resolve()) for path in args.real_no_contact_csv],
        "real_no_contact_val_csv": [str(Path(path).expanduser().resolve()) for path in args.real_no_contact_val_csv],
        "real_no_contact_train_windows": int(sum(len(dataset) for dataset in real_train_window_datasets)),
        "real_no_contact_val_windows": int(sum(len(dataset) for dataset in real_val_window_datasets)),
        "note": (
            "Residual features are intended for real robot logs. tau_ext from simulation remains label/analysis-only "
            "and is not accepted as a residual feature substitute."
        ),
    }
    save_json(out_dirs["metrics"] / "residual_feature_summary.json", residual_summary)

    print(f"Training complete. Saved models to {out_dirs['models']}")


if __name__ == "__main__":
    main()
