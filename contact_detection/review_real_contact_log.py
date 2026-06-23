"""Review a real robot contact-detection log and produce actionable diagnostics.

This script is meant for the first real-robot transfer checks:

1. record_real_log.py records q/qdot/q_des/tau_cmd.
2. infer_real_log.py writes P(contact).
3. this script summarizes whether P(contact) reacts during annotated contact
   intervals, whether no-contact false alarms are high, and whether the real
   features look out-of-distribution relative to the training scaler.

The model input policy remains unchanged: tau_ext/F/T/external force are not
used. Contact intervals are only weak labels for review/calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from utils import (
    StandardScaler,
    apply_stage_config,
    build_input_features,
    ensure_output_dirs,
    load_config,
    load_real_log_csv,
    output_root,
    save_json,
    save_residual_timeseries_figure,
)


def load_probability_csv(path: str | Path) -> dict[str, np.ndarray]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Probability CSV not found: {csv_path}")
    time_values: list[float] = []
    prob_values: list[float] = []
    pred_values: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        for row in reader:
            time_values.append(float(row["time"]))
            prob_text = row.get("contact_probability", "nan")
            pred_text = row.get("contact_prediction", "-1")
            prob_values.append(float(prob_text) if prob_text else float("nan"))
            pred_values.append(int(float(pred_text)) if pred_text else -1)
    return {
        "time": np.asarray(time_values, dtype=np.float64),
        "probability": np.asarray(prob_values, dtype=np.float64),
        "prediction": np.asarray(pred_values, dtype=np.int64),
    }


def load_intervals(text: str, config: dict, real_data: dict[str, np.ndarray]) -> list[tuple[float, float]]:
    if text.strip():
        raw = json.loads(text)
    else:
        raw = config.get("real_inference", {}).get("contact_intervals", [])
    intervals: list[tuple[float, float]] = []
    for item in raw:
        if len(item) != 2:
            raise ValueError("Each contact interval must be [start_s, end_s]")
        start_s, end_s = float(item[0]), float(item[1])
        if end_s > start_s:
            intervals.append((start_s, end_s))
    if not intervals and "contact_marker" in real_data:
        marker = np.asarray(real_data["contact_marker"], dtype=np.int64).reshape(-1)
        time = np.asarray(real_data["time"], dtype=np.float64).reshape(-1)
        positive = marker >= 1
        starts = np.flatnonzero(positive & np.r_[True, ~positive[:-1]])
        ends = np.flatnonzero(positive & np.r_[~positive[1:], True])
        intervals = [(float(time[start]), float(time[end])) for start, end in zip(starts, ends) if end >= start]
    return intervals


def interval_mask(time: np.ndarray, intervals: list[tuple[float, float]], margin_s: float = 0.0) -> np.ndarray:
    mask = np.zeros(time.shape[0], dtype=bool)
    for start_s, end_s in intervals:
        mask |= (time >= (float(start_s) - float(margin_s))) & (time <= (float(end_s) + float(margin_s)))
    return mask


def finite_stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "max": None, "p95": None}
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def threshold_segments(time: np.ndarray, probability: np.ndarray, threshold: float, mask: np.ndarray) -> list[dict]:
    valid = np.isfinite(probability)
    positive = valid & (probability >= float(threshold)) & mask
    if positive.size == 0:
        return []
    starts = np.flatnonzero(positive & np.r_[True, ~positive[:-1]])
    ends = np.flatnonzero(positive & np.r_[~positive[1:], True])
    segments = []
    for start_idx, end_idx in zip(starts, ends):
        segments.append(
            {
                "start_s": float(time[start_idx]),
                "end_s": float(time[end_idx]),
                "duration_s": float(max(0.0, time[end_idx] - time[start_idx])),
                "peak_probability": float(np.nanmax(probability[start_idx : end_idx + 1])),
            }
        )
    return segments


def interval_review(
    time: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    intervals: list[tuple[float, float]],
) -> list[dict]:
    rows = []
    for idx, (start_s, end_s) in enumerate(intervals):
        mask = (time >= start_s) & (time <= end_s)
        prob = probability[mask]
        local_time = time[mask]
        detected_indices = np.flatnonzero(np.isfinite(prob) & (prob >= float(threshold)))
        if detected_indices.size:
            first_idx = int(detected_indices[0])
            detected = True
            detection_time_s = float(local_time[first_idx])
            delay_s = float(detection_time_s - float(start_s))
        else:
            detected = False
            detection_time_s = None
            delay_s = None
        rows.append(
            {
                "event_index": int(idx),
                "start_s": float(start_s),
                "end_s": float(end_s),
                "duration_s": float(end_s - start_s),
                "detected": detected,
                "detection_time_s": detection_time_s,
                "detection_delay_s": delay_s,
                "max_probability": finite_stats(prob)["max"],
                "mean_probability": finite_stats(prob)["mean"],
                "fraction_above_threshold": None
                if prob.size == 0
                else float(np.mean(np.isfinite(prob) & (prob >= float(threshold)))),
            }
        )
    return rows


def load_decision_threshold(config: dict, out_dirs: dict[str, Path]) -> float:
    model_path = out_dirs["models"] / "gru_detector.pt"
    return load_decision_threshold_from_model_path(model_path)


def load_decision_threshold_from_model_path(model_path: Path) -> float:
    if not model_path.exists():
        return 0.5
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        return float(checkpoint.get("decision_threshold", 0.5))
    except Exception:
        return 0.5


def load_checkpoint_metadata(model_path: Path) -> dict:
    if not model_path.exists():
        return {}
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        return {
            "feature_mode": checkpoint.get("feature_mode", "original_42"),
            "feature_names": checkpoint.get("feature_names", []),
            "use_delta_features": bool(checkpoint.get("use_delta_features", True)),
            "residual_expected_torque_mode": checkpoint.get("residual_expected_torque_mode", "command_total"),
            "residual_offset_policy": checkpoint.get("residual_offset_policy", "episode_initial_no_contact_mean"),
            "tau_ext_input_policy": checkpoint.get("tau_ext_input_policy", "label_only_never_feature"),
        }
    except Exception:
        return {}


def make_feedback(summary: dict) -> list[str]:
    feedback: list[str] = []
    threshold = float(summary["decision_threshold"])
    no_contact = summary["probability_stats"]["no_contact"]
    contact = summary["probability_stats"]["contact"]
    fp_rate = summary["no_contact_false_alarm"]["fraction_above_threshold"]
    detected_events = int(summary["contact_interval_detection"]["detected_events"])
    total_events = int(summary["contact_interval_detection"]["num_intervals"])
    max_scaled = summary["feature_stats"].get("scaled_feature_max_abs", {}).get("max")

    if total_events == 0:
        feedback.append("No contact intervals were provided. Add --contact-intervals-json or contact_marker for event-level review.")
    elif detected_events < total_events:
        feedback.append("Some annotated contact intervals did not cross the decision threshold. Check whether contact is too weak, threshold is too high, or sim-to-real feature patterns differ.")
    else:
        feedback.append("All annotated contact intervals crossed the decision threshold at least once.")

    if fp_rate is not None and fp_rate > 0.05:
        feedback.append("No-contact false alarm fraction is high. First try real validation threshold calibration before fine-tuning.")
    elif fp_rate is not None:
        feedback.append("No-contact false alarm fraction is not high under the current threshold.")

    if contact.get("max") is not None and contact["max"] < threshold:
        feedback.append("Peak contact probability stayed below threshold. If e/qdot/tau_cmd responses are visible, consider weak-label fine-tuning.")
    if no_contact.get("p95") is not None and no_contact["p95"] > 0.5 * threshold:
        feedback.append("No-contact probability baseline is close to the threshold. Inspect tau_cmd scaling, qdot noise, and q_des alignment.")
    if max_scaled is not None and max_scaled > 6.0:
        feedback.append("Real features exceed roughly 6 training standard deviations. This suggests scaler/domain mismatch or an out-of-distribution robot state.")
    return feedback


def save_review_figure(
    path: Path,
    time: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    intervals: list[tuple[float, float]],
    e_norm: np.ndarray,
    qdot_norm: np.ndarray,
    tau_cmd_norm: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax in axes:
        for start_s, end_s in intervals:
            ax.axvspan(start_s, end_s, color="tab:orange", alpha=0.18, linewidth=0)
        ax.grid(True, alpha=0.3)

    axes[0].plot(time, probability, color="tab:blue", linewidth=1.7, label="P(contact)")
    axes[0].axhline(float(threshold), color="black", linestyle="--", linewidth=1.0, label="threshold")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right")

    axes[1].plot(time, e_norm, color="tab:red", linewidth=1.2, label="||e_q||")
    axes[1].plot(time, qdot_norm, color="tab:green", linewidth=1.2, label="||qdot||")
    axes[1].set_ylabel("State response")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, tau_cmd_norm, color="tab:purple", linewidth=1.2, label="||tau_cmd||")
    axes[2].set_ylabel("Command")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--real-csv", required=True, help="CSV recorded by record_real_log.py or a compatible logger.")
    parser.add_argument("--prob-csv", default="", help="CSV from infer_real_log.py. Defaults to outputs/<stage>/real_inference/real_contact_probability.csv.")
    parser.add_argument("--model-path", default="", help="Optional GRU checkpoint path used to read the same decision threshold as inference.")
    parser.add_argument("--scaler-path", default="", help="Optional scaler path used for feature out-of-distribution review.")
    parser.add_argument("--contact-intervals-json", default="", help='Optional intervals like "[[5, 7], [12, 13]]".')
    parser.add_argument("--allow-zero-tau", action="store_true")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold for review only.")
    parser.add_argument("--output-json", default="", help="Optional review JSON path.")
    parser.add_argument("--output-figure", default="", help="Optional review figure path.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    out_dirs = ensure_output_dirs(output_root(config))

    real_data = load_real_log_csv(
        args.real_csv,
        config,
        allow_zero_tau_cmd_override=True if args.allow_zero_tau else None,
    )
    prob_path = Path(args.prob_csv).expanduser().resolve() if args.prob_csv.strip() else out_dirs["real_inference"] / "real_contact_probability.csv"
    prob_data = load_probability_csv(prob_path)
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path.strip() else out_dirs["models"] / "gru_detector.pt"
    scaler_path = Path(args.scaler_path).expanduser().resolve() if args.scaler_path.strip() else out_dirs["models"] / "scaler.pkl"
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else load_decision_threshold_from_model_path(model_path)
    )
    checkpoint_meta = load_checkpoint_metadata(model_path)
    feature_mode = str(checkpoint_meta.get("feature_mode", config.get("dataset", {}).get("feature_mode", "original_42")))
    use_delta_features = bool(checkpoint_meta.get("use_delta_features", config.get("dataset", {}).get("use_delta_features", True)))

    intervals = load_intervals(args.contact_intervals_json, config, real_data)
    prob_time = prob_data["time"]
    probability = prob_data["probability"]
    contact_mask = interval_mask(prob_time, intervals, margin_s=0.0)
    no_contact_mask = ~contact_mask
    finite_mask = np.isfinite(probability)

    episode_id = np.asarray(real_data.get("episode_id", np.zeros(real_data["time"].shape[0])), dtype=np.int64)
    features, feature_names = build_input_features(
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
    e_norm = np.linalg.norm(real_data["q_des"] - real_data["q"], axis=1)
    qdot_norm = np.linalg.norm(real_data["qdot"], axis=1)
    tau_cmd_norm = np.linalg.norm(real_data["tau_cmd"], axis=1)
    tau_residual_norm = (
        np.linalg.norm(real_data["tau_residual"], axis=1) if "tau_residual" in real_data else None
    )
    tau_residual_corrected_norm = (
        np.linalg.norm(real_data["tau_residual_corrected"], axis=1)
        if "tau_residual_corrected" in real_data
        else None
    )

    feature_stats = {
        "e_norm": finite_stats(e_norm),
        "qdot_norm": finite_stats(qdot_norm),
        "tau_cmd_norm": finite_stats(tau_cmd_norm),
        "feature_mode": feature_mode,
    }
    if tau_residual_norm is not None:
        feature_stats["tau_residual_norm"] = finite_stats(tau_residual_norm)
    if tau_residual_corrected_norm is not None:
        feature_stats["tau_residual_corrected_norm"] = finite_stats(tau_residual_corrected_norm)
    if scaler_path.exists():
        scaler = StandardScaler.load(scaler_path)
        scaled = scaler.transform(features)
        abs_scaled = np.max(np.abs(scaled), axis=1)
        top_idx = int(np.argmax(np.max(np.abs(scaled), axis=0))) if scaled.size else -1
        feature_stats["scaled_feature_max_abs"] = finite_stats(abs_scaled)
        feature_stats["largest_scaled_feature"] = {
            "index": top_idx,
            "name": feature_names[top_idx] if 0 <= top_idx < len(feature_names) else None,
            "max_abs_z": None if top_idx < 0 else float(np.max(np.abs(scaled[:, top_idx]))),
        }

    interval_rows = interval_review(prob_time, probability, threshold, intervals)
    detected_events = sum(1 for row in interval_rows if row["detected"])
    false_alarm_segments = threshold_segments(prob_time, probability, threshold, no_contact_mask & finite_mask)
    no_contact_duration_s = float(np.sum(no_contact_mask[:-1] * np.diff(prob_time))) if prob_time.size >= 2 else 0.0

    summary = {
        "stage": config["experiment_stage"],
        "real_csv": str(Path(args.real_csv).expanduser().resolve()),
        "probability_csv": str(prob_path),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "decision_threshold": threshold,
        "input_policy": (
            "model input uses checkpoint feature_mode; tau_ext/F/T/external force are not model inputs"
        ),
        "checkpoint_feature_policy": checkpoint_meta,
        "contact_intervals": [{"start_s": s, "end_s": e} for s, e in intervals],
        "probability_stats": {
            "all": finite_stats(probability),
            "no_contact": finite_stats(probability[no_contact_mask]),
            "contact": finite_stats(probability[contact_mask]),
            "fraction_above_threshold_all": float(np.mean(finite_mask & (probability >= threshold))) if probability.size else None,
        },
        "contact_interval_detection": {
            "num_intervals": int(len(intervals)),
            "detected_events": int(detected_events),
            "missed_events": int(len(intervals) - detected_events),
            "event_detection_rate": None if not intervals else float(detected_events / len(intervals)),
            "intervals": interval_rows,
        },
        "no_contact_false_alarm": {
            "num_segments": int(len(false_alarm_segments)),
            "segments": false_alarm_segments[:20],
            "no_contact_duration_s": no_contact_duration_s,
            "false_alarm_per_second": None if no_contact_duration_s <= 0.0 else float(len(false_alarm_segments) / no_contact_duration_s),
            "fraction_above_threshold": None
            if not np.any(no_contact_mask)
            else float(np.mean(finite_mask[no_contact_mask] & (probability[no_contact_mask] >= threshold))),
        },
        "feature_stats": feature_stats,
    }
    summary["feedback"] = make_feedback(summary)

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json.strip() else out_dirs["real_inference"] / "real_contact_review.json"
    output_figure = Path(args.output_figure).expanduser().resolve() if args.output_figure.strip() else out_dirs["figures"] / "real_contact_review.png"
    save_json(output_json, summary)

    # The probability output and feature log should usually share the same time base.
    # If they do not, interpolate probability only for plotting.
    plot_probability = probability
    if prob_time.shape != real_data["time"].shape or np.max(np.abs(prob_time[: min(prob_time.size, real_data["time"].size)] - real_data["time"][: min(prob_time.size, real_data["time"].size)])) > 1.0e-6:
        finite = np.isfinite(probability)
        plot_probability = np.interp(real_data["time"], prob_time[finite], probability[finite]) if np.any(finite) else np.full(real_data["time"].shape, np.nan)
    save_review_figure(
        output_figure,
        real_data["time"],
        plot_probability,
        threshold,
        intervals,
        e_norm,
        qdot_norm,
        tau_cmd_norm,
    )
    if "tau_residual" in real_data:
        save_residual_timeseries_figure(
            out_dirs["figures"] / "residual_timeseries_example.png",
            real_data["time"],
            real_data["tau_residual"],
            real_data.get("tau_residual_corrected"),
            probability=plot_probability,
            threshold=threshold,
        )

    print(f"Saved real contact review JSON to {output_json}")
    print(f"Saved real contact review figure to {output_figure}")
    for item in summary["feedback"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
