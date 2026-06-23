"""Validation-only hyperparameter tuning for MLP/GRU contact detectors.

This runner deliberately reuses an existing generated dataset.  It does not
call generate_sim_dataset.py, so MLP/GRU comparisons are not mixed with dataset
changes.  Each trial writes into outputs/<stage>/ablations/<tag>/, leaving the
default model artifacts untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from utils import (
    apply_stage_config,
    ensure_output_dirs,
    load_config,
    load_json,
    output_root,
    save_config_yaml,
    save_json,
)


SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "mlp": {
        "mlp_hidden_dim": [32, 64, 128, 256],
        "mlp_num_layers": [1, 2, 3],
        "mlp_dropout": [0.0, 0.1, 0.2, 0.3],
        "mlp_lr": [1.0e-3, 5.0e-4, 2.0e-4],
        "mlp_weight_decay": [0.0, 1.0e-5, 1.0e-4, 1.0e-3],
        "pos_weight_override": [None, 1.0, 2.0, 3.0],
    },
    "gru": {
        "gru_hidden_dim": [32, 64, 128],
        "gru_num_layers": [1, 2],
        "gru_dropout": [0.0, 0.1, 0.2, 0.3],
        "gru_lr": [1.0e-3, 5.0e-4, 2.0e-4],
        "gru_weight_decay": [0.0, 1.0e-5, 1.0e-4, 1.0e-3],
        "pos_weight_override": [None, 1.0, 2.0, 3.0],
    },
}


def sanitize_tag(text: str) -> str:
    allowed = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "study"


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def dataset_file_info(path: Path, compute_hash: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info
    stat = path.stat()
    info.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime_unix": float(stat.st_mtime),
        }
    )
    if compute_hash:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        info["sha256"] = digest.hexdigest()
    return info


def dataset_inventory(stage_root: Path, compute_hash: bool = False) -> dict[str, Any]:
    dataset_dir = stage_root / "datasets"
    return {
        "reuse_policy": (
            "Hyperparameter tuning reuses these existing npz files. "
            "No dataset generation is performed inside tune_detectors.py."
        ),
        "dataset_dir": str(dataset_dir),
        "sim_train": dataset_file_info(dataset_dir / "sim_train.npz", compute_hash),
        "sim_val": dataset_file_info(dataset_dir / "sim_val.npz", compute_hash),
        "sim_test": dataset_file_info(dataset_dir / "sim_test.npz", compute_hash),
    }


def require_tuning_datasets(stage_root: Path) -> None:
    missing = [
        path
        for path in (stage_root / "datasets" / "sim_train.npz", stage_root / "datasets" / "sim_val.npz")
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Tuning requires a fixed generated dataset, but these files are missing:\n"
            f"{formatted}\n"
            "Run/copy dataset generation once first, then tune on the same files."
        )


def sample_random_params(model_name: str, rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in SEARCH_SPACES[model_name].items()}


def suggest_optuna_params(model_name: str, trial) -> dict[str, Any]:
    return {
        key: trial.suggest_categorical(key, values)
        for key, values in SEARCH_SPACES[model_name].items()
    }


def apply_trial_params(config: dict, model_name: str, params: dict[str, Any]) -> dict:
    trial_config = copy.deepcopy(config)
    training_cfg = trial_config.setdefault("training", {})
    for key, value in params.items():
        training_cfg[key] = value

    # MLP tuning can reuse a configured legacy GRU checkpoint so the trial only
    # changes the MLP baseline.  GRU tuning must retrain GRU, so disable reuse.
    if model_name == "gru":
        training_cfg["reuse_gru_checkpoint_path"] = None
        training_cfg["reuse_gru_train_log_path"] = None

    training_cfg["checkpoint_selection_metric"] = "val_f1"
    return trial_config


def read_trial_metric(artifact_root: Path, model_name: str) -> dict[str, Any]:
    log_path = artifact_root / "metrics" / ("mlp_train_log.json" if model_name == "mlp" else "train_log.json")
    if not log_path.exists():
        raise FileNotFoundError(f"Trial log not found: {log_path}")
    log_payload = load_json(log_path)
    return {
        "train_log_path": str(log_path),
        "best_epoch": int(log_payload.get("best_epoch_selection_label", log_payload.get("best_epoch", -1))),
        "best_val_f1_selection_label": float(
            log_payload.get("best_val_f1_selection_label", log_payload.get("best_val_f1", 0.0))
        ),
        "best_val_loss_selection_label": float(
            log_payload.get("best_val_loss_selection_label", log_payload.get("best_val_loss", float("inf")))
        ),
        "best_val_precision_selection_label": float(
            log_payload.get("best_val_precision_selection_label", log_payload.get("best_val_precision", 0.0))
        ),
        "best_val_recall_selection_label": float(
            log_payload.get("best_val_recall_selection_label", log_payload.get("best_val_recall", 0.0))
        ),
        "decision_threshold_value": float(
            log_payload.get(
                "decision_threshold_value",
                log_payload.get("operating_probability_threshold", log_payload.get("best_probability_threshold", 0.5)),
            )
        ),
        "threshold_selected_on": log_payload.get("threshold_selected_on", "selection_label_validation"),
        "threshold_selection_metric": log_payload.get("threshold_selection_metric", "val_f1_selection_label"),
        "label_basis_for_threshold_search": log_payload.get("label_basis_for_threshold_search", "original_command_label"),
    }


def trial_rank(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row.get("best_val_f1_selection_label", 0.0)),
        -float(row.get("best_val_loss_selection_label", float("inf"))),
    )


def best_rows_by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") not in {"completed", "reused_existing_trial"}:
            continue
        model_name = str(row["model_type"]).lower()
        if model_name not in best or trial_rank(row) > trial_rank(best[model_name]):
            best[model_name] = row
    return best


def save_tuning_summary(
    *,
    summary_path: Path,
    study_name: str,
    sampler_requested: str,
    sampler_used: str,
    config_path: Path,
    stage: str,
    stage_root: Path,
    rows: list[dict[str, Any]],
    dataset_info: dict[str, Any],
) -> dict[str, Any]:
    best = best_rows_by_model(rows)
    payload = {
        "study_name": study_name,
        "sampler_requested": sampler_requested,
        "sampler_used": sampler_used,
        "config_path": str(config_path),
        "stage": stage,
        "stage_root": str(stage_root),
        "selection_policy": (
            "Hyperparameter trials are ranked by validation F1 on the selection label. "
            "If validation F1 ties, lower validation loss is the tie-breaker. "
            "The test split is not used for hyperparameter selection."
        ),
        "primary_metric": "best_val_f1_selection_label",
        "tie_breaker": "lower_best_val_loss_selection_label",
        "test_metrics_used_for_selection": False,
        "dataset": dataset_info,
        "search_spaces": SEARCH_SPACES,
        "runs": sorted(rows, key=lambda row: (str(row.get("model_type")), int(row.get("trial_index", -1)))),
        "best_by_model_validation_only": best,
    }
    save_json(summary_path, payload)
    return payload


def run_command(command: list[str], *, dry_run: bool) -> int:
    print(command_text(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def run_trial(
    *,
    model_name: str,
    trial_index: int,
    params: dict[str, Any],
    base_config: dict,
    base_output_dir: Path,
    config_dir: Path,
    stage: str,
    study_name: str,
    stage_root: Path,
    summary_rows: list[dict[str, Any]],
    summary_path: Path,
    dataset_info: dict[str, Any],
    sampler_requested: str,
    sampler_used: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    scripts_dir = Path(__file__).resolve().parent
    tag = sanitize_tag(f"tune_{study_name}__{model_name}_{trial_index:03d}")
    artifact_root = stage_root / "ablations" / tag
    metrics_dir = artifact_root / "metrics"
    metric_log = metrics_dir / ("mlp_train_log.json" if model_name == "mlp" else "train_log.json")
    config_path = config_dir / f"{tag}.yaml"

    trial_config = apply_trial_params(base_config, model_name, params)
    trial_config["output_dir"] = str(base_output_dir)
    trial_config["stage_output_dirs"] = True
    trial_config["experiment_stage"] = stage
    trial_config.setdefault("training", {})["tuning_model_type"] = model_name
    trial_config["hyperparameter_tuning"] = {
        "enabled": True,
        "study_name": study_name,
        "trial_index": int(trial_index),
        "model_type": model_name,
        "selection_policy": "validation_f1_selection_label_only",
        "test_metrics_used_for_selection": False,
        "dataset_reuse_policy": "reuse_existing_stage_npz_dataset",
    }
    save_config_yaml(config_path, trial_config)

    row: dict[str, Any] = {
        "model_type": model_name.upper(),
        "trial_index": int(trial_index),
        "ablation_tag": tag,
        "artifact_root": str(artifact_root),
        "config_path": str(config_path),
        "params": params,
        "status": "pending",
        "test_metrics_used_for_selection": False,
    }

    if metric_log.exists() and not overwrite and not dry_run:
        row.update(read_trial_metric(artifact_root, model_name))
        row["status"] = "reused_existing_trial"
        summary_rows.append(row)
        save_tuning_summary(
            summary_path=summary_path,
            study_name=study_name,
            sampler_requested=sampler_requested,
            sampler_used=sampler_used,
            config_path=Path(base_config["_config_path"]),
            stage=stage,
            stage_root=stage_root,
            rows=summary_rows,
            dataset_info=dataset_info,
        )
        return row

    command = [
        sys.executable,
        str(scripts_dir / "train_detectors.py"),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--ablation-tag",
        tag,
    ]
    row["command"] = command_text(command)
    return_code = run_command(command, dry_run=dry_run)
    row["return_code"] = int(return_code)

    if dry_run:
        row["status"] = "dry_run"
    elif return_code != 0:
        row["status"] = "failed"
        summary_rows.append(row)
        save_tuning_summary(
            summary_path=summary_path,
            study_name=study_name,
            sampler_requested=sampler_requested,
            sampler_used=sampler_used,
            config_path=Path(base_config["_config_path"]),
            stage=stage,
            stage_root=stage_root,
            rows=summary_rows,
            dataset_info=dataset_info,
        )
        raise RuntimeError(f"Tuning trial failed: {tag}")
    else:
        row.update(read_trial_metric(artifact_root, model_name))
        row["status"] = "completed"

    summary_rows.append(row)
    save_tuning_summary(
        summary_path=summary_path,
        study_name=study_name,
        sampler_requested=sampler_requested,
        sampler_used=sampler_used,
        config_path=Path(base_config["_config_path"]),
        stage=stage,
        stage_root=stage_root,
        rows=summary_rows,
        dataset_info=dataset_info,
    )
    return row


def evaluate_best_rows(best_rows: dict[str, dict[str, Any]], stage: str, dry_run: bool) -> None:
    scripts_dir = Path(__file__).resolve().parent
    evaluated_tags: set[str] = set()
    for row in best_rows.values():
        tag = str(row["ablation_tag"])
        if tag in evaluated_tags:
            continue
        evaluated_tags.add(tag)
        command = [
            sys.executable,
            str(scripts_dir / "evaluate_detectors.py"),
            "--config",
            str(row["config_path"]),
            "--stage",
            stage,
            "--ablation-tag",
            tag,
        ]
        return_code = run_command(command, dry_run=dry_run)
        if return_code != 0:
            raise RuntimeError(f"Evaluation of validation-selected best trial failed: {tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base config path. The generated dataset under this output root is reused.")
    parser.add_argument("--stage", default="randomized_sim", help="Stage whose fixed dataset should be reused.")
    parser.add_argument("--models", nargs="+", choices=("mlp", "gru"), default=["mlp", "gru"])
    parser.add_argument("--n-trials", type=int, default=None, help="Apply the same number of trials to all selected models.")
    parser.add_argument("--n-trials-mlp", type=int, default=12)
    parser.add_argument("--n-trials-gru", type=int, default=12)
    parser.add_argument("--sampler", choices=("random", "optuna"), default="random")
    parser.add_argument("--require-optuna", action="store_true", help="Fail instead of falling back to random search when Optuna is missing.")
    parser.add_argument("--study-name", default=None, help="Name used in trial tags and the tuning summary file.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluate-best", action="store_true", help="After validation-only tuning, run test evaluation for the selected best trial(s).")
    parser.add_argument("--hash-datasets", action="store_true", help="Record SHA256 hashes for train/val/test npz files.")
    parser.add_argument("--dry-run", action="store_true", help="Write trial configs and print commands without training.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run an existing trial tag instead of reusing its saved train log.")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    staged_config = copy.deepcopy(raw_config)
    apply_stage_config(staged_config, args.stage)
    stage_root = output_root(staged_config)
    require_tuning_datasets(stage_root)
    base_output_dir = stage_root.parent if bool(staged_config.get("stage_output_dirs", True)) else stage_root

    study_name = sanitize_tag(args.study_name or time.strftime("validation_tuning_%Y%m%d_%H%M%S"))
    base_dirs = ensure_output_dirs(stage_root)
    study_root = stage_root / "hparam_tuning" / study_name
    config_dir = study_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_dirs["metrics"] / f"hparam_tuning_{study_name}.json"
    dataset_info = dataset_inventory(stage_root, compute_hash=bool(args.hash_datasets))

    optuna_available = False
    sampler_used = args.sampler
    if args.sampler == "optuna":
        try:
            import optuna  # type: ignore

            optuna_available = True
        except ImportError:
            if args.require_optuna:
                raise ImportError(
                    "Optuna is not installed. Install optional dependency 'optuna' or run with --sampler random."
                )
            sampler_used = "random_fallback_because_optuna_is_not_installed"
            optuna = None  # type: ignore

    seed = int(args.seed if args.seed is not None else raw_config.get("seed", 42))
    rng = random.Random(seed)
    summary_rows: list[dict[str, Any]] = []

    n_trials_by_model = {
        "mlp": int(args.n_trials if args.n_trials is not None else args.n_trials_mlp),
        "gru": int(args.n_trials if args.n_trials is not None else args.n_trials_gru),
    }

    print(f"Fixed dataset root: {stage_root / 'datasets'}")
    print(f"Tuning summary: {summary_path}")
    print(f"Sampler requested={args.sampler}, used={sampler_used}, optuna_available={optuna_available}")

    if args.sampler == "optuna" and optuna_available:
        for model_name in args.models:
            n_trials = n_trials_by_model[model_name]
            if n_trials <= 0:
                continue
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
                study_name=f"{study_name}_{model_name}",
            )

            def objective(trial, tuned_model=model_name):
                params = suggest_optuna_params(tuned_model, trial)
                row = run_trial(
                    model_name=tuned_model,
                    trial_index=int(trial.number),
                    params=params,
                    base_config=raw_config,
                    base_output_dir=base_output_dir,
                    config_dir=config_dir,
                    stage=args.stage,
                    study_name=study_name,
                    stage_root=stage_root,
                    summary_rows=summary_rows,
                    summary_path=summary_path,
                    dataset_info=dataset_info,
                    sampler_requested=args.sampler,
                    sampler_used=sampler_used,
                    dry_run=bool(args.dry_run),
                    overwrite=bool(args.overwrite),
                )
                trial.set_user_attr("row", row)
                return float(row.get("best_val_f1_selection_label", 0.0))

            study.optimize(objective, n_trials=n_trials)
    else:
        for model_name in args.models:
            for trial_index in range(n_trials_by_model[model_name]):
                params = sample_random_params(model_name, rng)
                run_trial(
                    model_name=model_name,
                    trial_index=trial_index,
                    params=params,
                    base_config=raw_config,
                    base_output_dir=base_output_dir,
                    config_dir=config_dir,
                    stage=args.stage,
                    study_name=study_name,
                    stage_root=stage_root,
                    summary_rows=summary_rows,
                    summary_path=summary_path,
                    dataset_info=dataset_info,
                    sampler_requested=args.sampler,
                    sampler_used=sampler_used,
                    dry_run=bool(args.dry_run),
                    overwrite=bool(args.overwrite),
                )

    payload = save_tuning_summary(
        summary_path=summary_path,
        study_name=study_name,
        sampler_requested=args.sampler,
        sampler_used=sampler_used,
        config_path=Path(raw_config["_config_path"]),
        stage=args.stage,
        stage_root=stage_root,
        rows=summary_rows,
        dataset_info=dataset_info,
    )
    if args.evaluate_best and not args.dry_run:
        evaluate_best_rows(payload["best_by_model_validation_only"], args.stage, dry_run=False)
    print(f"Saved validation-only tuning summary to {summary_path}")


if __name__ == "__main__":
    main()
