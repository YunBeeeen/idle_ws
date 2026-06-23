"""Create detailed simulation visualizations for the trained contact detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contact_dataset import ContactWindowDataset
from models import GRUDetector
from utils import (
    StandardScaler,
    apply_stage_config,
    binary_classification_metrics,
    ensure_output_dirs,
    load_config,
    load_json,
    load_npz_dataset,
    output_root,
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
        raise ImportError("PyTorch is required for visualize_sim_detection.py.") from exc


def run_gru_probability(config: dict, split_path: Path) -> tuple[np.ndarray, ContactWindowDataset, dict]:
    out_dirs = ensure_output_dirs(output_root(config))
    scaler = StandardScaler.load(out_dirs["models"] / "scaler.pkl")
    checkpoint_path = out_dirs["models"] / "gru_detector.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained GRU model: {checkpoint_path}")

    torch, DataLoader = _import_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    bundle = ContactWindowDataset.from_npz(
        split_path,
        window_length=int(config["dataset"]["window_length"]),
        stride=int(config["dataset"]["stride"]),
        use_delta_features=bool(checkpoint.get("use_delta_features", config["dataset"]["use_delta_features"])),
        scaler=scaler,
    )

    if str(checkpoint.get("model_type", "binary")).strip().lower() != "binary":
        raise ValueError(f"{checkpoint_path} is not a binary checkpoint.")
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

    loader = DataLoader(
        bundle.dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for windows, _labels in loader:
            windows = windows.to(device=device, dtype=torch.float32)
            logits = model(windows)
            chunks.append(sigmoid(logits.cpu().numpy()))
    probability = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float64)
    print(f"Computed GRU probabilities on device={device}")
    return probability, bundle.dataset, checkpoint


def episode_window_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    episode_id: np.ndarray,
    threshold: float,
) -> list[tuple[float, int, dict[str, float | int]]]:
    rows: list[tuple[float, int, dict[str, float | int]]] = []
    prediction = (probability >= float(threshold)).astype(np.int64)
    for episode in np.unique(episode_id):
        mask = episode_id == episode
        if int(np.sum(labels[mask])) == 0:
            continue
        metrics = binary_classification_metrics(labels[mask], prediction[mask])
        rows.append((float(metrics["f1"]), int(episode), metrics))
    rows.sort(key=lambda item: item[0])
    return rows


def choose_episode(rows: list[tuple[float, int, dict[str, float | int]]], choice: str) -> int:
    if not rows:
        raise ValueError("No positive-contact episodes were found in this split.")
    if choice == "challenging":
        return rows[0][1]
    if choice == "median":
        return rows[len(rows) // 2][1]
    if choice == "clean":
        return rows[-1][1]
    try:
        return int(choice)
    except ValueError as exc:
        raise ValueError("--episode must be 'challenging', 'median', 'clean', or an integer episode id") from exc


def full_window_signal(length: int, end_indices: np.ndarray, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = np.full(length, fill, dtype=np.float64)
    out[end_indices] = np.asarray(values, dtype=np.float64)
    return out


def save_dashboard(
    path: Path,
    data: dict[str, np.ndarray],
    episode: int,
    probability_full: np.ndarray,
    gru_pred_full: np.ndarray,
    threshold_pred_full: np.ndarray,
    gru_threshold: float,
    threshold_gamma: float,
    metrics: dict[str, float | int] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = data["episode_id"] == int(episode)
    if not np.any(mask):
        raise ValueError(f"Episode {episode} not found")

    time = data["time"][mask]
    label = data["label"][mask]
    tau_ext_norm = np.linalg.norm(data["tau_ext"][mask], axis=1)
    prob = probability_full[mask]
    gru_pred = gru_pred_full[mask]
    threshold_pred = threshold_pred_full[mask]
    q = data["q"][mask]
    qdot = data["qdot"][mask]
    tau_cmd = data["tau_cmd"][mask]

    if metrics is None:
        title_metrics = ""
    else:
        title_metrics = (
            f" | GRU P={float(metrics['precision']):.2f}, "
            f"R={float(metrics['recall']):.2f}, F1={float(metrics['f1']):.2f}"
        )

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"Simulation Contact Detection Dashboard - episode {episode}{title_metrics}", fontsize=15)

    axes[0].step(time, label, where="post", color="black", linewidth=2.0, label="contact label")
    axes[0].plot(time, tau_ext_norm, color="tab:orange", linewidth=1.5, label=r"$||\tau_{ext}||$")
    axes[0].set_ylabel("Label / Nm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time, prob, color="tab:blue", linewidth=2.0, label="GRU P(contact)")
    axes[1].step(time, gru_pred, where="post", color="tab:blue", alpha=0.35, linewidth=1.4, label="GRU hard pred")
    axes[1].step(
        time,
        threshold_pred,
        where="post",
        color="tab:red",
        alpha=0.75,
        linewidth=1.4,
        label=f"baseline pred, gamma={threshold_gamma:.3f}",
    )
    axes[1].axhline(gru_threshold, color="tab:blue", linestyle="--", linewidth=1.2, label=f"GRU threshold={gru_threshold:.2f}")
    axes[1].set_ylabel("Detection")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    for idx in range(q.shape[1]):
        axes[2].plot(time, q[:, idx], linewidth=1.0, label=f"q{idx + 1}")
    axes[2].set_ylabel("q [rad]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=6, loc="upper right", fontsize=8)

    for idx in range(qdot.shape[1]):
        axes[3].plot(time, qdot[:, idx], linewidth=1.0, label=f"qd{idx + 1}")
    axes[3].set_ylabel("qdot [rad/s]")
    axes[3].grid(True, alpha=0.3)

    for idx in range(tau_cmd.shape[1]):
        axes[4].plot(time, tau_cmd[:, idx], linewidth=1.0, label=f"tau{idx + 1}")
    axes[4].set_xlabel("Time [s]")
    axes[4].set_ylabel("tau_cmd [Nm]")
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage override.")
    parser.add_argument("--split", default="sim_test", choices=["sim_train", "sim_val", "sim_test"])
    parser.add_argument(
        "--episode",
        default="median",
        help="Episode id, or one of: challenging, median, clean",
    )
    parser.add_argument("--output", default="", help="Optional output PNG path.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    out_dirs = ensure_output_dirs(output_root(config))
    split_path = out_dirs["datasets"] / f"{args.split}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    data = load_npz_dataset(split_path)
    probability, dataset, checkpoint = run_gru_probability(config, split_path)
    labels_for_windows = dataset.labels_for_windows().astype(np.int64)
    episodes_for_windows = dataset.episodes_for_windows()

    configured_threshold = config["training"].get("gru_decision_threshold")
    gru_threshold = (
        float(configured_threshold) if configured_threshold is not None else float(checkpoint.get("decision_threshold", 0.5))
    )
    rows = episode_window_metrics(labels_for_windows, probability, episodes_for_windows, gru_threshold)
    episode = choose_episode(rows, str(args.episode))
    metrics_by_episode = {episode_id: metrics for _score, episode_id, metrics in rows}

    threshold_payload = load_json(out_dirs["models"] / "threshold.json")
    threshold_gamma = float(threshold_payload["gamma"])
    threshold_score_full, _ = threshold_score_from_data(data, np.arange(data["time"].shape[0]), config)
    threshold_pred_full = (threshold_score_full >= threshold_gamma).astype(np.float64)

    probability_full = full_window_signal(data["time"].shape[0], dataset.end_indices, probability)
    gru_pred_full = full_window_signal(
        data["time"].shape[0],
        dataset.end_indices,
        (probability >= gru_threshold).astype(np.float64),
        fill=0.0,
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else out_dirs["figures"] / f"{args.split}_episode_{episode}_dashboard.png"
    )
    save_dashboard(
        output_path,
        data,
        episode,
        probability_full,
        gru_pred_full,
        threshold_pred_full,
        gru_threshold,
        threshold_gamma,
        metrics_by_episode.get(episode),
    )
    print(f"Saved simulation dashboard to {output_path}")


if __name__ == "__main__":
    main()
