"""Generate paper-ready figures for the contact detection pipeline.

evaluate_detectors.py와 infer_real_log.py가 만든 중간 결과를 다시 읽어서
2페이지 논문/발표에 넣기 좋은 figure를 한 번에 재생성하는 helper다.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from utils import (
    apply_stage_config,
    ensure_output_dirs,
    load_config,
    load_json,
    output_root,
    save_metric_bar_figure,
    save_overview_pipeline_figure,
    save_precision_recall_curve_figure,
    save_real_probability_figure,
    save_sim_prediction_example,
    save_threshold_tradeoff_figure,
)


def load_real_probability_csv(path) -> tuple[np.ndarray, np.ndarray]:
    """Load time and P(contact) from infer_real_log.py output CSV."""
    time_values: list[float] = []
    prob_values: list[float] = []
    with open(path, "r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            time_values.append(float(row["time"]))
            prob_values.append(float(row["contact_probability"]))
    return np.asarray(time_values, dtype=np.float64), np.asarray(prob_values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage override.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    out_dirs = ensure_output_dirs(output_root(config))

    # overview_pipeline.png는 논문 방법론 그림: sim data -> training -> sim eval -> real inference.
    save_overview_pipeline_figure(out_dirs["figures"] / "overview_pipeline.png")

    metrics_path = out_dirs["metrics"] / "sim_test_metrics.json"
    example_data_path = out_dirs["metrics"] / "sim_prediction_example_data.npz"
    real_csv_path = out_dirs["real_inference"] / "real_contact_probability.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Simulation metrics not found: {metrics_path}. Run evaluate_detectors.py before plot_results.py."
        )
    if not example_data_path.exists():
        raise FileNotFoundError(
            f"Simulation example data not found: {example_data_path}. Run evaluate_detectors.py first."
        )

    # sim metric bar와 prediction example은 evaluate_detectors.py 결과를 재사용한다.
    metrics_payload = load_json(metrics_path)
    save_metric_bar_figure(out_dirs["figures"] / "sim_metric_bar.png", metrics_payload)
    if "gru_threshold_sweep" in metrics_payload:
        save_threshold_tradeoff_figure(
            out_dirs["figures"] / "gru_threshold_tradeoff.png",
            metrics_payload["gru_threshold_sweep"],
            comparison_sweeps={"MLP": metrics_payload.get("mlp_threshold_sweep", [])},
        )
    if "gru_threshold_sweep" in metrics_payload or "mlp_threshold_sweep" in metrics_payload:
        save_precision_recall_curve_figure(
            out_dirs["figures"] / "precision_recall_curve_mlp_gru.png",
            {
                "MLP": metrics_payload.get("mlp_threshold_sweep", []),
                "GRU": metrics_payload.get("gru_threshold_sweep", []),
            },
        )

    example_data = np.load(example_data_path, allow_pickle=False)
    save_sim_prediction_example(
        out_dirs["figures"] / "sim_prediction_example.png",
        example_data["time"],
        example_data["label"],
        example_data["threshold_prediction"],
        example_data["gru_probability"],
        mlp_probability=example_data["mlp_probability"] if "mlp_probability" in example_data.files else None,
        e_norm=example_data["e_norm"] if "e_norm" in example_data.files else None,
    )

    # real figure는 optional이다. real inference를 아직 안 돌렸다면 simulation figure만 재생성한다.
    if real_csv_path.exists():
        real_time, real_prob = load_real_probability_csv(real_csv_path)
        save_real_probability_figure(
            out_dirs["figures"] / "real_contact_probability.png",
            real_time,
            real_prob,
            intervals=config.get("real_inference", {}).get("contact_intervals", []),
        )
    print(f"Saved figures to {out_dirs['figures']}")


if __name__ == "__main__":
    main()
