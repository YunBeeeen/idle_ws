"""Run binary GRU contact detection on a real robot CSV log.

논문 흐름에서 이 파일은 "학습된 GRU를 실제 로봇 로그에 적용해
P(contact)가 의도적 접촉 순간에 올라가는지 확인"하는 단계다.

실제 로봇에는 외력 정답 ground truth가 없으므로 기본 출력은 정량 정확도
평가가 아니다. 결과는 preliminary real-robot feasibility check로 해석한다.

입력 feature는 시뮬레이션/학습과 동일해야 한다:
    [q, qdot, e_q, tau_cmd, optional delta]

qdot이 없으면 q와 time으로 미분해 추정하고, real log dt가 학습 dt와 다르면
config에 따라 resampling한다. tau_ext/external force/measured torque는 사용하지 않는다.
"""

from __future__ import annotations

# argparse/csv: 실제 로봇 CSV 경로와 옵션을 받고, inference 결과 CSV를 저장한다.
import argparse
import csv
from pathlib import Path

# NumPy는 real log column을 [N, 6] array로 다루고 probability를 full time축에 맞추는 데 사용한다.
import numpy as np

from contact_dataset import ContactWindowDataset
from utils import (
    StandardScaler,
    apply_stage_config,
    binary_classification_metrics,
    build_input_features,
    ensure_output_dirs,
    load_config,
    load_real_log_csv,
    output_root,
    save_json,
    save_residual_timeseries_figure,
    save_real_probability_figure,
    select_torch_device,
    sigmoid,
)


def _import_torch():
    try:
        # PyTorch는 저장된 GRU checkpoint를 load해서 real log sliding window에 forward pass를 수행한다.
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for real-log inference. Install the dependencies from "
            "contact_detection/requirements.txt before running infer_real_log.py."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage/model directory to use.")
    parser.add_argument("--csv", required=True, help="Path to the real robot log CSV")
    parser.add_argument("--model-path", default="", help="Optional GRU checkpoint path. Defaults to outputs/<stage>/models/gru_detector.pt.")
    parser.add_argument("--scaler-path", default="", help="Optional scaler path. Defaults to outputs/<stage>/models/scaler.pkl.")
    parser.add_argument("--output-csv", default="", help="Optional output probability CSV path.")
    parser.add_argument("--output-figure", default="", help="Optional output probability figure path.")
    parser.add_argument("--summary-json", default="", help="Optional output summary JSON path.")
    parser.add_argument(
        "--allow-zero-tau",
        action="store_true",
        help="Allow missing tau_cmd columns by filling commanded torque with zeros.",
    )
    parser.add_argument(
        "--use-marker-metrics",
        action="store_true",
        help="Compute real precision/recall/F1 only if the CSV contains contact_marker.",
    )
    args = parser.parse_args()

    # 1) 어떤 stage 모델을 쓸지 선택한다. 보통 논문/실험은 randomized_sim을 사용한다.
    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    out_dirs = ensure_output_dirs(output_root(config))

    scaler_path = Path(args.scaler_path).expanduser().resolve() if args.scaler_path.strip() else out_dirs["models"] / "scaler.pkl"
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path.strip() else out_dirs["models"] / "gru_detector.pt"
    for path in (scaler_path, model_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact is missing: {path}. Run train_detectors.py first."
            )

    # 2) scaler.pkl과 gru_detector.pt가 있어야 real log inference가 가능하다.
    torch, DataLoader = _import_torch()
    from models import GRUDetector

    scaler = StandardScaler.load(scaler_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    if str(checkpoint.get("model_type", "binary")) != "binary":
        raise ValueError(
            f"{model_path} is not a binary checkpoint. "
            "Use config.yaml/staged binary training for real-log inference."
        )
    use_delta_features = bool(checkpoint.get("use_delta_features", False))
    feature_mode = str(checkpoint.get("feature_mode", config.get("dataset", {}).get("feature_mode", "original_42")))
    window_length = int(checkpoint.get("window_length", config["dataset"]["window_length"]))
    stride = int(checkpoint.get("stride", config["dataset"]["stride"]))

    # 3) CSV loader는 joint order 재정렬, qdot 미존재 시 numerical differentiation,
    #    target_dt resampling을 처리한다.
    real_data = load_real_log_csv(
        args.csv,
        config,
        allow_zero_tau_cmd_override=True if args.allow_zero_tau else None,
    )
    episode_id = np.asarray(real_data.get("episode_id", np.zeros(real_data["time"].shape[0])), dtype=np.int64)
    # 4) 학습과 동일한 feature를 만든다. tau_ext/state.tau/external force는 여기로 들어올 경로가 없다.
    features, _feature_names = build_input_features(
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
    # 5) 실제 로그에는 label이 없으므로 dummy zero label을 넣고 window만 만든다.
    inference_dataset = ContactWindowDataset(
        features=features,
        labels=np.zeros(real_data["time"].shape[0], dtype=np.float32),
        episode_id=episode_id,
        window_length=window_length,
        stride=stride,
        scaler=scaler,
    )

    # 6) checkpoint metadata로 같은 GRU 구조를 복원한다.
    model = GRUDetector(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        bidirectional=bool(checkpoint.get("bidirectional", False)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = select_torch_device(torch, config["training"].get("device", "auto"))
    model.to(device)
    model.eval()
    print(f"Running real-log inference on device={device}")

    # 7) DataLoader로 real log window를 batch inference한다. shuffle=False로 시간 순서를 유지한다.
    loader = DataLoader(
        inference_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    prob_chunks: list[np.ndarray] = []
    # 8) torch.no_grad(): 실제 inference에서는 gradient가 필요 없어서 메모리/속도를 아낀다.
    with torch.no_grad():
        for windows, _ in loader:
            windows = windows.to(device=device, dtype=torch.float32)
            logits = model(windows)
            prob_chunks.append(sigmoid(logits.cpu().numpy()))
    valid_prob = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.zeros(0, dtype=np.float64)
    decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
    valid_pred = (valid_prob >= decision_threshold).astype(np.int64)

    # 9) window가 완전히 차기 전 구간은 예측이 없으므로 NaN으로 남긴다.
    full_prob = np.full(real_data["time"].shape[0], np.nan, dtype=np.float64)
    full_pred = np.full(real_data["time"].shape[0], -1, dtype=np.int64)
    full_prob[inference_dataset.end_indices] = valid_prob
    full_pred[inference_dataset.end_indices] = valid_pred

    csv_output_path = Path(args.output_csv).expanduser().resolve() if args.output_csv.strip() else out_dirs["real_inference"] / "real_contact_probability.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "contact_probability", "contact_prediction"])
        for time_value, prob_value, pred_value in zip(real_data["time"], full_prob, full_pred):
            writer.writerow([f"{float(time_value):.9f}", f"{float(prob_value):.9f}", int(pred_value)])

    figure_output_path = Path(args.output_figure).expanduser().resolve() if args.output_figure.strip() else out_dirs["figures"] / "real_contact_probability.png"
    save_real_probability_figure(
        figure_output_path,
        real_data["time"],
        full_prob,
        intervals=config.get("real_inference", {}).get("contact_intervals", []),
    )

    finite_prob = valid_prob[np.isfinite(valid_prob)]
    full_finite_prob = full_prob[np.isfinite(full_prob)]
    labels_for_sanity = None
    if "contact_label" in real_data:
        labels_for_sanity = np.asarray(real_data["contact_label"], dtype=np.int64).reshape(-1)
    elif "contact_marker" in real_data:
        labels_for_sanity = np.asarray(real_data["contact_marker"], dtype=np.int64).reshape(-1)
    if labels_for_sanity is not None and labels_for_sanity.shape[0] == full_pred.shape[0]:
        valid_mask = full_pred >= 0
        sanity_label = labels_for_sanity[valid_mask]
        sanity_pred = full_pred[valid_mask]
        sanity_prob = full_prob[valid_mask]
    else:
        valid_mask = full_pred >= 0
        sanity_label = np.zeros(int(np.sum(valid_mask)), dtype=np.int64)
        sanity_pred = full_pred[valid_mask]
        sanity_prob = full_prob[valid_mask]
    summary = {
        "stage": config["experiment_stage"],
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "decision_threshold": decision_threshold,
        "feature_mode": feature_mode,
        "feature_names": list(checkpoint.get("feature_names", _feature_names)),
        "residual_expected_torque_mode": checkpoint.get("residual_expected_torque_mode", "command_total"),
        "residual_offset_policy": checkpoint.get("residual_offset_policy", "episode_initial_no_contact_mean"),
        "tau_ext_input_policy": checkpoint.get("tau_ext_input_policy", "label_only_never_feature"),
        "estimated_log_dt": float(real_data["estimated_dt"][0]),
        "num_valid_windows": int(valid_prob.shape[0]),
        "mean_contact_probability": float(np.mean(finite_prob)) if finite_prob.size else None,
        "max_contact_probability": float(np.max(finite_prob)) if finite_prob.size else None,
        "p95_contact_probability": float(np.percentile(finite_prob, 95.0)) if finite_prob.size else None,
        "p99_contact_probability": float(np.percentile(finite_prob, 99.0)) if finite_prob.size else None,
        "note": "preliminary real-robot feasibility check, not quantitative accuracy evaluation",
    }
    # 10) 실제 로봇 정량 metric은 사용자가 명시적으로 marker metrics를 켠 경우에만 계산한다.
    if args.use_marker_metrics:
        if "contact_marker" not in real_data:
            raise ValueError("--use-marker-metrics was set, but CSV has no contact_marker column.")
        marker = np.asarray(real_data["contact_marker"], dtype=np.int64)[inference_dataset.end_indices]
        summary["marker_metrics"] = binary_classification_metrics(marker, valid_pred)
    summary_output_path = Path(args.summary_json).expanduser().resolve() if args.summary_json.strip() else out_dirs["real_inference"] / "real_contact_probability_summary.json"
    save_json(summary_output_path, summary)

    real_no_contact_sanity = {
        "stage": config["experiment_stage"],
        "real_csv": str(Path(args.csv).expanduser().resolve()),
        "probability_csv": str(csv_output_path),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_mode": feature_mode,
        "decision_threshold": decision_threshold,
        "label_source": "contact_label_or_contact_marker_if_available_else_assume_no_contact",
        "num_valid_windows": int(np.sum(valid_mask)),
        "num_positive_labels": int(np.sum(sanity_label == 1)),
        "false_positive_fraction": (
            None
            if sanity_pred.size == 0 or not np.any(sanity_label == 0)
            else float(np.mean(sanity_pred[sanity_label == 0] == 1))
        ),
        "probability_mean": float(np.nanmean(sanity_prob)) if sanity_prob.size else None,
        "probability_max": float(np.nanmax(sanity_prob)) if sanity_prob.size else None,
        "probability_p95": float(np.nanpercentile(sanity_prob, 95.0)) if sanity_prob.size else None,
        "probability_p99": float(np.nanpercentile(sanity_prob, 99.0)) if sanity_prob.size else None,
        "used_for_model_selection": False,
        "tau_ext_input_policy": "tau_ext/external force is not used as model input",
    }
    save_json(out_dirs["metrics"] / "real_no_contact_sanity.json", real_no_contact_sanity)
    if "tau_residual" in real_data:
        residual_figure_path = out_dirs["figures"] / "residual_timeseries_example.png"
        save_residual_timeseries_figure(
            residual_figure_path,
            real_data["time"],
            real_data["tau_residual"],
            real_data.get("tau_residual_corrected"),
            probability=full_prob,
            threshold=decision_threshold,
        )

    print(
        "Real-log qualitative summary: "
        f"mean P(contact)={summary['mean_contact_probability']}, "
        f"max P(contact)={summary['max_contact_probability']}, "
        f"threshold={decision_threshold:.3f}"
    )
    if "marker_metrics" in summary:
        print(f"Marker metrics were explicitly enabled: {summary['marker_metrics']}")
    else:
        print("No real precision/recall/F1 computed: this is a preliminary feasibility check.")
    print(f"Saved real-log contact probabilities to {csv_output_path}")
    print(f"Saved real-log probability figure to {figure_output_path}")
    print(f"Saved real-log summary to {summary_output_path}")
    print(f"Saved real no-contact sanity summary to {out_dirs['metrics'] / 'real_no_contact_sanity.json'}")


if __name__ == "__main__":
    main()
