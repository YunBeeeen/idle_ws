"""Summarize real no-contact logs for residual-torque detector design.

This script does not train a model.  It is a diagnostic step for deciding
whether the real robot detector should move away from raw tracking-error
features and toward residual torque features.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    apply_stage_config,
    compute_episodewise_delta,
    load_config,
    load_real_log_csv,
    save_json,
)


def finite_stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "max": None, "p95": None, "p99": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
    }


def norm_stats(matrix: np.ndarray | None) -> dict[str, float | None]:
    if matrix is None:
        return {"mean": None, "std": None, "max": None, "p95": None, "p99": None}
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        return finite_stats(arr)
    return finite_stats(np.linalg.norm(arr, axis=1))


def per_joint_abs_p95(matrix: np.ndarray | None) -> list[float | None]:
    if matrix is None:
        return []
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        return []
    result: list[float | None] = []
    for joint_idx in range(arr.shape[1]):
        values = np.abs(arr[:, joint_idx])
        values = values[np.isfinite(values)]
        result.append(None if values.size == 0 else float(np.percentile(values, 95.0)))
    return result


def resolve_csv_path(text: str, manifest_path: Path | None) -> Path:
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if manifest_path is not None:
        manifest_candidate = (manifest_path.parent / candidate).resolve()
        if manifest_candidate.exists():
            return manifest_candidate
    return cwd_candidate


def load_manifest_csvs(path: str | Path) -> list[Path]:
    manifest_path = Path(path).expanduser().resolve()
    csv_paths: list[Path] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            csv_text = row.get("csv_path", "").strip()
            if csv_text:
                csv_paths.append(resolve_csv_path(csv_text, manifest_path))
    return csv_paths


def load_probability_summary(probability_summary_dir: Path | None, episode_id: str) -> dict[str, Any]:
    if probability_summary_dir is None:
        return {}
    summary_path = probability_summary_dir / f"{episode_id}_summary.json"
    prob_csv_path = probability_summary_dir / f"{episode_id}_probability.csv"
    result: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        result.update(
            {
                "legacy_feature_mode": summary.get("feature_mode"),
                "legacy_decision_threshold": summary.get("decision_threshold"),
                "legacy_probability_mean": summary.get("mean_contact_probability"),
                "legacy_probability_max": summary.get("max_contact_probability"),
                "legacy_probability_p95": summary.get("p95_contact_probability"),
                "legacy_probability_p99": summary.get("p99_contact_probability"),
            }
        )
    if prob_csv_path.exists():
        probs: list[float] = []
        preds: list[int] = []
        with prob_csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                prob_text = row.get("contact_probability", "")
                pred_text = row.get("contact_prediction", "")
                if not prob_text:
                    continue
                prob = float(prob_text)
                if not np.isfinite(prob):
                    continue
                probs.append(prob)
                if pred_text:
                    preds.append(int(float(pred_text)))
        if probs:
            prob_arr = np.asarray(probs, dtype=np.float64)
            result.setdefault("legacy_probability_mean", float(np.mean(prob_arr)))
            result.setdefault("legacy_probability_max", float(np.max(prob_arr)))
            result.setdefault("legacy_probability_p95", float(np.percentile(prob_arr, 95.0)))
            result.setdefault("legacy_probability_p99", float(np.percentile(prob_arr, 99.0)))
        if preds:
            pred_arr = np.asarray(preds, dtype=np.int64)
            result["legacy_no_contact_false_alarm_fraction"] = float(np.mean(pred_arr == 1))
    return result


def estimate_sample_hz(time: np.ndarray) -> float | None:
    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    if time_arr.size < 3:
        return None
    dt = np.diff(time_arr)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def summarize_one(csv_path: Path, config: dict, probability_summary_dir: Path | None) -> dict[str, Any]:
    real = load_real_log_csv(csv_path, config)
    time = np.asarray(real["time"], dtype=np.float64).reshape(-1)
    q = np.asarray(real["q"], dtype=np.float64)
    qdot = np.asarray(real["qdot"], dtype=np.float64)
    q_des = np.asarray(real["q_des"], dtype=np.float64)
    tau_cmd = np.asarray(real["tau_cmd"], dtype=np.float64)
    tau_meas = np.asarray(real["tau_meas"], dtype=np.float64) if "tau_meas" in real else None
    tau_residual = np.asarray(real["tau_residual"], dtype=np.float64) if "tau_residual" in real else None
    tau_residual_corrected = (
        np.asarray(real["tau_residual_corrected"], dtype=np.float64) if "tau_residual_corrected" in real else None
    )

    e_q = q_des - q
    delta_e_q = compute_episodewise_delta(e_q)
    delta_qdot = compute_episodewise_delta(qdot)
    delta_tau_cmd = compute_episodewise_delta(tau_cmd)
    delta_tau_residual = (
        compute_episodewise_delta(tau_residual_corrected)
        if tau_residual_corrected is not None
        else compute_episodewise_delta(tau_residual)
        if tau_residual is not None
        else None
    )

    episode_id = csv_path.stem
    row: dict[str, Any] = {
        "episode_id": episode_id,
        "csv_path": str(csv_path),
        "num_rows": int(time.shape[0]),
        "duration_s": None if time.size == 0 else float(time[-1] - time[0]),
        "sample_hz_estimate": estimate_sample_hz(time),
        "tau_ext_input_policy": "label_only_never_feature",
        "residual_definition": "tau_residual = tau_meas - tau_cmd",
        "residual_offset_policy": "episode_initial_no_contact_mean",
        "e_q_norm": norm_stats(e_q),
        "delta_e_q_norm": norm_stats(delta_e_q),
        "qdot_norm": norm_stats(qdot),
        "delta_qdot_norm": norm_stats(delta_qdot),
        "tau_cmd_norm": norm_stats(tau_cmd),
        "delta_tau_cmd_norm": norm_stats(delta_tau_cmd),
        "tau_meas_norm": norm_stats(tau_meas),
        "tau_residual_norm": norm_stats(tau_residual),
        "tau_residual_corrected_norm": norm_stats(tau_residual_corrected),
        "delta_tau_residual_norm": norm_stats(delta_tau_residual),
        "tau_residual_corrected_abs_p95_by_joint": per_joint_abs_p95(tau_residual_corrected),
    }
    row.update(load_probability_summary(probability_summary_dir, episode_id))
    return row


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "episode_id": row["episode_id"],
        "csv_path": row["csv_path"],
        "num_rows": row["num_rows"],
        "duration_s": row["duration_s"],
        "sample_hz_estimate": row["sample_hz_estimate"],
        "legacy_no_contact_false_alarm_fraction": row.get("legacy_no_contact_false_alarm_fraction"),
        "legacy_probability_mean": row.get("legacy_probability_mean"),
        "legacy_probability_p95": row.get("legacy_probability_p95"),
        "legacy_probability_max": row.get("legacy_probability_max"),
    }
    for key in (
        "e_q_norm",
        "delta_e_q_norm",
        "qdot_norm",
        "delta_qdot_norm",
        "tau_cmd_norm",
        "tau_residual_corrected_norm",
        "delta_tau_residual_norm",
    ):
        stats = row.get(key, {})
        flat[f"{key}_mean"] = stats.get("mean")
        flat[f"{key}_p95"] = stats.get("p95")
        flat[f"{key}_max"] = stats.get("max")
    return flat


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [flatten_row(row) for row in rows]
    if not flat_rows:
        return
    fieldnames = list(flat_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def save_overview_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["episode_id"] for row in rows]
    e_p95 = [row["e_q_norm"]["p95"] or 0.0 for row in rows]
    res_p95 = [row["tau_residual_corrected_norm"]["p95"] or 0.0 for row in rows]
    dres_p95 = [row["delta_tau_residual_norm"]["p95"] or 0.0 for row in rows]
    false_alarm = [row.get("legacy_no_contact_false_alarm_fraction") for row in rows]
    false_alarm = [np.nan if value is None else float(value) for value in false_alarm]

    x = np.arange(len(rows), dtype=np.float64)
    width = 0.26
    fig, axes = plt.subplots(2, 1, figsize=(max(11.0, len(rows) * 0.85), 8.0), sharex=True)
    axes[0].bar(x - width, e_p95, width=width, label=r"$||e_q||$ p95")
    axes[0].bar(x, res_p95, width=width, label=r"$||\tau_{res,corr}||$ p95")
    axes[0].bar(x + width, dres_p95, width=width, label=r"$||\Delta\tau_{res,corr}||$ p95")
    axes[0].set_ylabel("Norm p95")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].bar(x, false_alarm, width=0.5, color="tab:red", label="Legacy original_42 false alarm fraction")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("False alarm fraction")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_title("Real no-contact residual diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="contact_detection/config_legacy_gru_mlp.yaml")
    parser.add_argument("--stage", default="randomized_sim")
    parser.add_argument("--csv", action="append", default=[], help="Real log CSV. Can be repeated.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest CSV containing csv_path rows.")
    parser.add_argument(
        "--probability-summary-dir",
        default=None,
        help="Optional directory containing <episode>_summary.json and <episode>_probability.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="contact_detection/outputs_legacy_gru_mlp/randomized_sim/real_inference/no_contact_residual_analysis",
    )
    parser.add_argument(
        "--resample-to-target-dt",
        action="store_true",
        help="Resample real logs to config target_dt. Disabled by default so diagnostics use recorded samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_stage_config(load_config(args.config), args.stage)
    config.setdefault("real_inference", {})["resample_to_target_dt"] = bool(args.resample_to_target_dt)
    csv_paths: list[Path] = []
    for manifest in args.manifest:
        csv_paths.extend(load_manifest_csvs(manifest))
    csv_paths.extend(Path(path).expanduser().resolve() for path in args.csv)
    csv_paths = sorted(dict.fromkeys(csv_paths))
    if not csv_paths:
        raise ValueError("Provide at least one --csv or --manifest.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_summary_dir = (
        Path(args.probability_summary_dir).expanduser().resolve() if args.probability_summary_dir else None
    )
    rows = [summarize_one(path, config, probability_summary_dir) for path in csv_paths]
    payload = {
        "config": str(Path(args.config).expanduser().resolve()),
        "stage": str(args.stage),
        "num_episodes": len(rows),
        "feature_direction": "residual_torque_real_robot",
        "recommended_first_residual_feature_mode": "residual_v2",
        "recommended_first_residual_feature_blocks": [
            "tau_residual_corrected",
            "delta_tau_residual",
            "qdot",
            "delta_qdot",
        ],
        "note": (
            "This is a no-contact diagnostic summary. It should be used to design residual features "
            "and identify hard negative episodes, not as contact detection accuracy."
        ),
        "episodes": rows,
    }
    save_json(output_dir / "real_no_contact_residual_summary.json", payload)
    write_summary_csv(output_dir / "real_no_contact_residual_summary.csv", rows)
    save_overview_plot(output_dir / "real_no_contact_residual_overview.png", rows)
    print(f"Saved residual summary to {output_dir / 'real_no_contact_residual_summary.json'}")
    print(f"Saved residual CSV to {output_dir / 'real_no_contact_residual_summary.csv'}")
    print(f"Saved residual overview figure to {output_dir / 'real_no_contact_residual_overview.png'}")


if __name__ == "__main__":
    main()
