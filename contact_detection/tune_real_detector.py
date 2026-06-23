"""Validation-only hyperparameter tuning for real-log contact GRU models.

This runner is intentionally separate from the simulation tuner.  It repeatedly
calls ``train_real_detector.py`` on the same real CSV set, stores every trial in
its own output directory, and ranks candidates using validation metrics only.

No live robot result or test/review CSV is used for tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import save_json


DEFAULT_NO_CONTACT_GLOBS = [
    "contact_detection/real_logs/no_contact/20260618/static_hold_*.csv",
    "contact_detection/real_logs/no_contact/20260619/no_contact_home_bridge_hold_*.csv",
    "contact_detection/real_logs/no_contact/20260619/slow_sine_*.csv",
]
DEFAULT_CONTACT_GLOBS = [
    "contact_detection/real_logs/contact/20260619/contact_ee_static_hold_*.csv",
    "contact_detection/real_logs/contact/20260619/contact_ee_home_bridge_hold_*.csv",
    "contact_detection/real_logs/contact/20260619/contact_ee_slow_sine_*.csv",
]


@dataclass(frozen=True)
class TrialParams:
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    window_length: int
    transition_exclusion_s: float
    max_validation_fpr: float


def sanitize_name(text: str) -> str:
    allowed = [ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(text)]
    return "".join(allowed).strip("_") or "real_tuning"


def random_params(rng: random.Random) -> TrialParams:
    return TrialParams(
        hidden_dim=int(rng.choice([16, 24, 32, 48, 64, 96])),
        num_layers=int(rng.choice([1, 2])),
        dropout=float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.4])),
        lr=float(10 ** rng.uniform(-4.2, -2.7)),
        weight_decay=float(10 ** rng.uniform(-5.0, -2.2)),
        batch_size=int(rng.choice([128, 256, 512])),
        window_length=int(rng.choice([20, 30, 40, 50])),
        transition_exclusion_s=float(rng.choice([0.3, 0.5, 0.8, 1.0])),
        max_validation_fpr=float(rng.choice([0.03, 0.05, 0.08, 0.10])),
    )


def optuna_params(trial: Any) -> TrialParams:
    return TrialParams(
        hidden_dim=int(trial.suggest_categorical("hidden_dim", [16, 24, 32, 48, 64, 96])),
        num_layers=int(trial.suggest_categorical("num_layers", [1, 2])),
        dropout=float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4])),
        lr=float(trial.suggest_float("lr", 1.0e-4, 2.0e-3, log=True)),
        weight_decay=float(trial.suggest_float("weight_decay", 1.0e-5, 6.0e-3, log=True)),
        batch_size=int(trial.suggest_categorical("batch_size", [128, 256, 512])),
        window_length=int(trial.suggest_categorical("window_length", [20, 30, 40, 50])),
        transition_exclusion_s=float(trial.suggest_categorical("transition_exclusion_s", [0.3, 0.5, 0.8, 1.0])),
        max_validation_fpr=float(trial.suggest_categorical("max_validation_fpr", [0.03, 0.05, 0.08, 0.10])),
    )


def build_train_command(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    params: TrialParams,
    seed: int,
) -> list[str]:
    script = Path(__file__).resolve().parent / "train_real_detector.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--feature-mode",
        str(args.feature_mode),
        "--model",
        "gru",
        "--split-policy",
        str(args.split_policy),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(params.batch_size),
        "--hidden-dim",
        str(params.hidden_dim),
        "--num-layers",
        str(params.num_layers),
        "--dropout",
        f"{params.dropout:.8g}",
        "--lr",
        f"{params.lr:.10g}",
        "--weight-decay",
        f"{params.weight_decay:.10g}",
        "--window-length",
        str(params.window_length),
        "--stride",
        str(int(args.stride)),
        "--transition-exclusion-s",
        f"{params.transition_exclusion_s:.8g}",
        "--threshold-selection-policy",
        str(args.threshold_selection_policy),
        "--max-validation-fpr",
        f"{params.max_validation_fpr:.8g}",
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--residual-offset-duration",
        f"{float(args.residual_offset_duration):.8g}",
    ]
    if bool(args.exclude_transition_val):
        cmd.append("--exclude-transition-val")
    for pattern in args.no_contact_glob:
        cmd.extend(["--no-contact-glob", str(pattern)])
    for pattern in args.contact_glob:
        cmd.extend(["--contact-glob", str(pattern)])
    return cmd


def read_trial_summary(output_dir: Path, trial_index: int, params: TrialParams) -> dict[str, Any]:
    summary_path = output_dir / "metrics" / "real_train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing trial summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row = payload["models"]["gru"]
    metrics = row["validation_metrics"]
    return {
        "trial_index": int(trial_index),
        "status": "completed",
        "output_dir": str(output_dir),
        "model_path": str(output_dir / "models" / "gru_detector.pt"),
        "scaler_path": str(output_dir / "models" / "scaler.pkl"),
        "feature_mode": payload.get("feature_mode"),
        "params": params.__dict__,
        "best_epoch": int(row["best_epoch"]),
        "decision_threshold": float(row["decision_threshold"]),
        "validation_precision": float(metrics["precision"]),
        "validation_recall": float(metrics["recall"]),
        "validation_f1": float(metrics["f1"]),
        "validation_false_positive_rate": float(metrics["false_positive_rate"]),
        "validation_false_negative_rate": float(metrics["false_negative_rate"]),
        "confusion_matrix": row.get("confusion_matrix"),
    }


def trial_rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("validation_f1", -1.0)),
        -float(row.get("validation_false_positive_rate", 1.0)),
        float(row.get("validation_recall", -1.0)),
        float(row.get("validation_precision", -1.0)),
    )


def summarize(
    *,
    args: argparse.Namespace,
    study_name: str,
    study_dir: Path,
    rows: list[dict[str, Any]],
    sampler_used: str,
    optuna_available: bool,
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") in {"completed", "reused"}]
    best = max(completed, key=trial_rank) if completed else None
    return {
        "study_name": study_name,
        "study_dir": str(study_dir),
        "sampler_requested": str(args.sampler),
        "sampler_used": sampler_used,
        "optuna_available": bool(optuna_available),
        "feature_mode": str(args.feature_mode),
        "model_type": "gru",
        "selection_basis": (
            "validation only: rank by validation F1, tie-break by lower validation FPR, "
            "then higher recall and precision"
        ),
        "test_or_live_used_for_selection": False,
        "threshold_selection_policy": str(args.threshold_selection_policy),
        "exclude_transition_val": bool(args.exclude_transition_val),
        "no_contact_glob": list(args.no_contact_glob),
        "contact_glob": list(args.contact_glob),
        "best_by_validation_only": best,
        "runs": sorted(rows, key=lambda row: int(row.get("trial_index", -1))),
    }


def run_one_trial(
    *,
    args: argparse.Namespace,
    study_dir: Path,
    trial_index: int,
    params: TrialParams,
    seed: int,
) -> dict[str, Any]:
    output_dir = study_dir / f"trial_{trial_index:03d}"
    summary_path = output_dir / "metrics" / "real_train_summary.json"
    if summary_path.exists() and not bool(args.overwrite):
        row = read_trial_summary(output_dir, trial_index, params)
        row["status"] = "reused"
        return row

    cmd = build_train_command(args=args, output_dir=output_dir, params=params, seed=seed)
    print("\n[trial]", trial_index)
    print(" ".join(cmd))
    if bool(args.dry_run):
        return {
            "trial_index": int(trial_index),
            "status": "dry_run",
            "output_dir": str(output_dir),
            "params": params.__dict__,
            "command": cmd,
        }

    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False)
    if result.returncode != 0:
        return {
            "trial_index": int(trial_index),
            "status": "failed",
            "output_dir": str(output_dir),
            "params": params.__dict__,
            "returncode": int(result.returncode),
            "command": cmd,
        }
    return read_trial_summary(output_dir, trial_index, params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="contact_detection/config_legacy_gru_mlp.yaml")
    parser.add_argument("--feature-mode", default="real_no_eq_no_dqdot_v1")
    parser.add_argument("--study-name", default="")
    parser.add_argument("--output-root", default="contact_detection/outputs_real/hparam_tuning")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--split-policy", choices=["stratified_by_condition", "tail"], default="stratified_by_condition")
    parser.add_argument("--threshold-selection-policy", choices=["f1", "f2", "recall_constrained_f1", "fpr_constrained_f1"], default="fpr_constrained_f1")
    parser.add_argument("--exclude-transition-val", action="store_true", default=True)
    parser.add_argument("--no-exclude-transition-val", dest="exclude_transition_val", action="store_false")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--residual-offset-duration", type=float, default=2.0)
    parser.add_argument("--sampler", choices=["optuna", "random"], default="optuna")
    parser.add_argument("--require-optuna", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-contact-glob", action="append", default=DEFAULT_NO_CONTACT_GLOBS)
    parser.add_argument("--contact-glob", action="append", default=DEFAULT_CONTACT_GLOBS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_name = sanitize_name(args.study_name or f"real_24d_optuna_{time.strftime('%Y%m%d_%H%M%S')}")
    study_dir = Path(args.output_root).expanduser().resolve() / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_path = study_dir / "real_hparam_tuning_summary.json"

    rows: list[dict[str, Any]] = []
    sampler_used = "random"
    optuna_available = False
    optuna = None
    if args.sampler == "optuna":
        try:
            import optuna as optuna_module  # type: ignore

            optuna = optuna_module
            optuna_available = True
            sampler_used = "optuna_tpe"
        except Exception as exc:
            if bool(args.require_optuna):
                raise RuntimeError("Optuna was required but could not be imported.") from exc
            sampler_used = "random_fallback_because_optuna_is_unavailable"

    if optuna is not None:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
            study_name=study_name,
        )

        def objective(trial: Any) -> float:
            params = optuna_params(trial)
            row = run_one_trial(
                args=args,
                study_dir=study_dir,
                trial_index=int(trial.number),
                params=params,
                seed=int(args.seed) + int(trial.number),
            )
            rows.append(row)
            trial.set_user_attr("row", row)
            if row.get("status") == "failed":
                return -1.0
            return float(row.get("validation_f1", -1.0))

        study.optimize(objective, n_trials=int(args.n_trials))
    else:
        rng = random.Random(int(args.seed))
        for trial_index in range(int(args.n_trials)):
            params = random_params(rng)
            rows.append(
                run_one_trial(
                    args=args,
                    study_dir=study_dir,
                    trial_index=trial_index,
                    params=params,
                    seed=int(args.seed) + int(trial_index),
                )
            )

    summary = summarize(
        args=args,
        study_name=study_name,
        study_dir=study_dir,
        rows=rows,
        sampler_used=sampler_used,
        optuna_available=optuna_available,
    )
    save_json(summary_path, summary)
    print(f"\nSaved real tuning summary: {summary_path}")
    best = summary.get("best_by_validation_only")
    if best:
        print(
            "Best validation-only trial: "
            f"#{best['trial_index']} F1={best['validation_f1']:.3f} "
            f"P={best['validation_precision']:.3f} R={best['validation_recall']:.3f} "
            f"FPR={best['validation_false_positive_rate']:.3f}"
        )
        print(f"model_path={best['model_path']}")
        print(f"scaler_path={best['scaler_path']}")


if __name__ == "__main__":
    main()
