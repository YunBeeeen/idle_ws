"""Compare real-robot contact detectors on the same manually labeled CSV.

This script is for poster/debug comparison, not model selection.  It applies
multiple trained GRU checkpoints to the same real log, then saves:

- probability overlay figure
- nc/pc confusion matrix comparison figure
- JSON summary
- CSV table

Labels come from ``--contact-intervals-json`` when provided; otherwise the
script uses ``contact_label``/``contact_marker`` columns if available.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contact_dataset import ContactWindowDataset
from models import GRUDetector
from utils import (
    StandardScaler,
    apply_stage_config,
    binary_classification_metrics,
    build_input_features,
    load_config,
    load_real_log_csv,
    save_json,
    select_torch_device,
    sigmoid,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_path: Path
    scaler_path: Path


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required for compare_real_models.py") from exc


def parse_model_spec(values: list[str]) -> ModelSpec:
    if len(values) != 3:
        raise ValueError("--model requires NAME MODEL_PATH SCALER_PATH")
    name, model_path, scaler_path = values
    return ModelSpec(
        name=str(name),
        model_path=Path(model_path).expanduser().resolve(),
        scaler_path=Path(scaler_path).expanduser().resolve(),
    )


def load_intervals(text: str) -> list[tuple[float, float]]:
    if not str(text).strip():
        return []
    raw = json.loads(text)
    intervals: list[tuple[float, float]] = []
    for item in raw:
        if len(item) != 2:
            raise ValueError("Each contact interval must be [start_s, end_s]")
        start_s, end_s = float(item[0]), float(item[1])
        if end_s > start_s:
            intervals.append((start_s, end_s))
    return intervals


def labels_from_intervals(time: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    labels = np.zeros(time.shape[0], dtype=np.int64)
    for start_s, end_s in intervals:
        labels[(time >= float(start_s)) & (time <= float(end_s))] = 1
    return labels


def labels_from_real_data(real_data: dict[str, np.ndarray], intervals: list[tuple[float, float]]) -> np.ndarray:
    if intervals:
        return labels_from_intervals(np.asarray(real_data["time"], dtype=np.float64), intervals)
    if "contact_label" in real_data:
        return np.asarray(real_data["contact_label"], dtype=np.int64).reshape(-1)
    if "contact_marker" in real_data:
        return np.asarray(real_data["contact_marker"], dtype=np.int64).reshape(-1)
    raise ValueError(
        "No labels available. Provide --contact-intervals-json or use a CSV with contact_label/contact_marker."
    )


def row_normalized_confusion(metrics: dict[str, float | int]) -> list[list[float]]:
    tn = int(metrics["tn"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    tp = int(metrics["tp"])
    nc_total = max(tn + fp, 1)
    pc_total = max(fn + tp, 1)
    return [
        [float(tn / nc_total), float(fp / nc_total)],
        [float(fn / pc_total), float(tp / pc_total)],
    ]


def run_model(
    *,
    spec: ModelSpec,
    real_data: dict[str, np.ndarray],
    config: dict,
    torch_module: Any,
    DataLoader: Any,
    device: Any,
) -> dict[str, Any]:
    if not spec.model_path.exists():
        raise FileNotFoundError(f"Missing model: {spec.model_path}")
    if not spec.scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {spec.scaler_path}")

    checkpoint = torch_module.load(spec.model_path, map_location="cpu")
    scaler = StandardScaler.load(spec.scaler_path)
    use_delta_features = bool(checkpoint.get("use_delta_features", False))
    feature_mode = str(checkpoint.get("feature_mode", config.get("dataset", {}).get("feature_mode", "original_42")))
    window_length = int(checkpoint.get("window_length", config["dataset"]["window_length"]))
    stride = int(checkpoint.get("stride", config["dataset"]["stride"]))
    decision_threshold = float(checkpoint.get("decision_threshold", 0.5))

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

    dataset = ContactWindowDataset(
        features=features,
        labels=np.zeros(real_data["time"].shape[0], dtype=np.float32),
        episode_id=episode_id,
        window_length=window_length,
        stride=stride,
        scaler=scaler,
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)

    model = GRUDetector(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        bidirectional=bool(checkpoint.get("bidirectional", False)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    with torch_module.no_grad():
        for windows, _labels in loader:
            windows = windows.to(device=device, dtype=torch_module.float32)
            logits = model(windows)
            chunks.append(sigmoid(logits.cpu().numpy()))
    valid_prob = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float64)
    valid_pred = (valid_prob >= decision_threshold).astype(np.int64)

    full_prob = np.full(real_data["time"].shape[0], np.nan, dtype=np.float64)
    full_pred = np.full(real_data["time"].shape[0], -1, dtype=np.int64)
    full_prob[dataset.end_indices] = valid_prob
    full_pred[dataset.end_indices] = valid_pred

    scaled = scaler.transform(features)
    max_abs_z_by_feature = np.max(np.abs(scaled), axis=0) if scaled.size else np.zeros(0)
    top_idx = int(np.argmax(max_abs_z_by_feature)) if max_abs_z_by_feature.size else -1

    return {
        "name": spec.name,
        "model_path": str(spec.model_path),
        "scaler_path": str(spec.scaler_path),
        "feature_mode": feature_mode,
        "feature_names": feature_names,
        "decision_threshold": decision_threshold,
        "time": np.asarray(real_data["time"], dtype=np.float64),
        "probability": full_prob,
        "prediction": full_pred,
        "end_indices": dataset.end_indices,
        "top_scaled_feature": None
        if top_idx < 0
        else {
            "name": feature_names[top_idx],
            "max_abs_z": float(max_abs_z_by_feature[top_idx]),
        },
    }


def save_probability_overlay(
    path: Path,
    time: np.ndarray,
    labels: np.ndarray,
    intervals: list[tuple[float, float]],
    model_results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for start_s, end_s in intervals:
        ax.axvspan(start_s, end_s, color="tab:orange", alpha=0.18, linewidth=0)
    if not intervals:
        ax.fill_between(time, 0.0, 1.0, where=labels.astype(bool), color="tab:orange", alpha=0.12, step="pre")
    for result in model_results:
        ax.plot(
            result["time"],
            result["probability"],
            linewidth=1.7,
            label=f"{result['name']} ({result['feature_mode']})",
        )
        ax.axhline(
            float(result["decision_threshold"]),
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"{result['name']} threshold={float(result['decision_threshold']):.3f}",
        )
    ax.set_title("Real Robot Contact Probability Comparison")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("P(contact)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(rows), figsize=(5.0 * len(rows), 4.4), squeeze=False)
    for ax, row in zip(axes[0], rows):
        counts = np.asarray(row["confusion_matrix"], dtype=np.int64)
        norm = np.asarray(row["confusion_matrix_row_normalized"], dtype=np.float64)
        im = ax.imshow(norm, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_xticks([0, 1], ["nc", "pc"])
        ax.set_yticks([0, 1], ["nc", "pc"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{row['model']} nc/pc confusion")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{counts[i, j]}\n{100.0 * norm[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    color="black" if norm[i, j] < 0.65 else "white",
                    fontsize=10,
                )
        ax.text(
            0.5,
            -0.28,
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} FPR={row['false_positive_rate']:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="row-normalized")
    fig.suptitle("nc = no contact, pc = physical contact", y=1.02)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "feature_mode",
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "tn",
        "fp",
        "fn",
        "tp",
        "contact_fraction_above_threshold",
        "no_contact_fraction_above_threshold",
        "contact_max_probability",
        "no_contact_p95_probability",
        "top_scaled_feature",
        "top_scaled_feature_max_abs_z",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="contact_detection/config_legacy_gru_mlp.yaml")
    parser.add_argument("--stage", default="randomized_sim")
    parser.add_argument("--csv", required=True, help="Real CSV to compare on.")
    parser.add_argument("--contact-intervals-json", default="", help='Example: "[[5.0, 13.0]]"')
    parser.add_argument("--output-dir", default="contact_detection/outputs_real/model_compare_20260619")
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "MODEL_PATH", "SCALER_PATH"),
        help="Model spec. Can be repeated.",
    )
    parser.add_argument("--allow-zero-tau", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    # Real-specific models in this workflow are trained on the recorded 100 Hz
    # CSV samples directly.  Keep the review time base identical unless the
    # caller intentionally edits the script/config for a different experiment.
    config.setdefault("real_inference", {})
    config["real_inference"]["resample_to_target_dt"] = False
    torch_module, DataLoader = _import_torch()
    device = select_torch_device(torch_module, args.device)

    model_specs = [parse_model_spec(values) for values in args.model or []]
    if not model_specs:
        model_specs = [
            ModelSpec(
                "30D_no_eq",
                Path("contact_detection/outputs_real/full_real_no_eq_v1_gru_20260619/models/gru_detector.pt").resolve(),
                Path("contact_detection/outputs_real/full_real_no_eq_v1_gru_20260619/models/scaler.pkl").resolve(),
            ),
            ModelSpec(
                "24D_no_eq_no_dqdot",
                Path("contact_detection/outputs_real/full_real_no_eq_no_dqdot_v1_gru_20260619/models/gru_detector.pt").resolve(),
                Path("contact_detection/outputs_real/full_real_no_eq_no_dqdot_v1_gru_20260619/models/scaler.pkl").resolve(),
            ),
        ]

    real_data = load_real_log_csv(
        args.csv,
        config,
        allow_zero_tau_cmd_override=True if args.allow_zero_tau else None,
    )
    intervals = load_intervals(args.contact_intervals_json)
    labels = labels_from_real_data(real_data, intervals)
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"

    model_results = [
        run_model(
            spec=spec,
            real_data=real_data,
            config=config,
            torch_module=torch_module,
            DataLoader=DataLoader,
            device=device,
        )
        for spec in model_specs
    ]

    rows: list[dict[str, Any]] = []
    time = np.asarray(real_data["time"], dtype=np.float64)
    for result in model_results:
        valid = result["prediction"] >= 0
        y_true = labels[valid]
        y_pred = result["prediction"][valid]
        probability = result["probability"][valid]
        metrics = binary_classification_metrics(y_true, y_pred)
        contact_mask = y_true == 1
        no_contact_mask = y_true == 0
        top_feature = result["top_scaled_feature"] or {}
        row = {
            "model": result["name"],
            "feature_mode": result["feature_mode"],
            "threshold": float(result["decision_threshold"]),
            **metrics,
            "confusion_matrix": [[int(metrics["tn"]), int(metrics["fp"])], [int(metrics["fn"]), int(metrics["tp"])]],
            "confusion_matrix_row_normalized": row_normalized_confusion(metrics),
            "contact_fraction_above_threshold": None
            if not np.any(contact_mask)
            else float(np.mean(y_pred[contact_mask] == 1)),
            "no_contact_fraction_above_threshold": None
            if not np.any(no_contact_mask)
            else float(np.mean(y_pred[no_contact_mask] == 1)),
            "contact_max_probability": None
            if not np.any(contact_mask)
            else float(np.nanmax(probability[contact_mask])),
            "no_contact_p95_probability": None
            if not np.any(no_contact_mask)
            else float(np.nanpercentile(probability[no_contact_mask], 95.0)),
            "top_scaled_feature": top_feature.get("name"),
            "top_scaled_feature_max_abs_z": top_feature.get("max_abs_z"),
            "model_path": result["model_path"],
            "scaler_path": result["scaler_path"],
        }
        rows.append(row)

    summary = {
        "real_csv": str(Path(args.csv).expanduser().resolve()),
        "contact_intervals": [{"start_s": s, "end_s": e} for s, e in intervals],
        "label_basis": "manual_contact_intervals" if intervals else "csv_contact_label_or_marker",
        "used_for_model_selection": False,
        "models": rows,
        "poster_note": (
            "Comparison uses the same real CSV and fixed checkpoint thresholds. "
            "nc=no contact, pc=physical contact. Test/review metrics are not used for model selection."
        ),
    }

    save_json(metrics_dir / "real_model_comparison_summary.json", summary)
    save_table_csv(metrics_dir / "real_model_comparison_table.csv", rows)
    save_probability_overlay(
        figures_dir / "real_model_probability_overlay.png",
        time,
        labels,
        intervals,
        model_results,
    )
    save_confusion_comparison(figures_dir / "real_model_confusion_comparison.png", rows)

    print(f"Saved comparison summary: {metrics_dir / 'real_model_comparison_summary.json'}")
    print(f"Saved comparison table: {metrics_dir / 'real_model_comparison_table.csv'}")
    print(f"Saved probability overlay: {figures_dir / 'real_model_probability_overlay.png'}")
    print(f"Saved confusion figure: {figures_dir / 'real_model_confusion_comparison.png'}")
    for row in rows:
        print(
            f"{row['model']}: F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"FPR={row['false_positive_rate']:.3f} confusion={row['confusion_matrix']}"
        )


if __name__ == "__main__":
    main()
