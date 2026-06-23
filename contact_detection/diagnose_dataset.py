"""Diagnose whether the simulated contact dataset has learnable contact signatures.

논문 결과를 믿기 전에 확인해야 하는 sanity-check 단계다. 성능보다 더 위험한
문제는 label leakage나 "feature에서 contact가 관측 불가능한 데이터"를 학습하는
것이다.

이 스크립트는 contact/no-contact 구간의 ||e_q||, ||delta_e_q||, ||qdot|| 분포를
비교하고, label=1이 실제로 nonzero ||tau_ext||와 동기화되는지 확인한다.
"""

from __future__ import annotations

# argparse/warnings: stage별 dataset을 CLI로 선택하고, observability 문제가 있으면 명확히 경고한다.
import argparse
import warnings
from pathlib import Path

# matplotlib은 dataset diagnosis figure를 저장하는 데 사용한다. 서버/터미널 환경에서도 저장되도록 Agg backend를 쓴다.
import matplotlib
# NumPy는 label, tau_ext, e_q norm 같은 통계량 계산에 사용한다.
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    apply_stage_config,
    compute_episodewise_delta,
    ensure_output_dirs,
    iter_episode_slices,
    load_config,
    load_npz_dataset,
    output_root,
    positive_segments,
    save_json,
)


OBSERVABILITY_WARNING = (
    "WARNING: contact labels may not be observable from current features. "
    "Increase disturbance magnitude/duration or reduce controller stiffness."
)


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    """Return simple descriptive statistics for one scalar signal."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _concat_splits(split_data: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate train/val/test only for diagnosis plots/statistics, not for training."""
    keys = split_data[next(iter(split_data))].keys()
    combined: dict[str, np.ndarray] = {}
    for key in keys:
        values = [payload[key] for payload in split_data.values() if key in payload]
        if not values:
            continue
        first = values[0]
        if first.dtype.kind in ("U", "S", "O"):
            combined[key] = first
        elif first.ndim == 0:
            combined[key] = np.asarray(values)
        else:
            combined[key] = np.concatenate(values, axis=0)
    return combined


def _assert_episode_splits_disjoint(split_data: dict[str, dict[str, np.ndarray]]) -> dict[str, list[int]]:
    """Fail fast if train/val/test episode_id sets overlap."""
    episode_sets = {
        split: set(np.asarray(payload["episode_id"]).astype(np.int64).tolist())
        for split, payload in split_data.items()
    }
    names = list(episode_sets)
    for idx, lhs in enumerate(names):
        for rhs in names[idx + 1 :]:
            overlap = episode_sets[lhs] & episode_sets[rhs]
            if overlap:
                raise ValueError(f"Episode leakage between {lhs} and {rhs}: {sorted(overlap)[:10]}")
    return {name: sorted(values) for name, values in episode_sets.items()}


def _event_counts_and_duration(data: dict[str, np.ndarray]) -> tuple[dict[str, int], float | None]:
    """Count positive label segments per episode and estimate contact duration."""
    event_counts: dict[str, int] = {}
    durations: list[float] = []
    time = np.asarray(data["time"], dtype=np.float64)
    label = np.asarray(data["label"]).astype(np.int64)
    episode_id = np.asarray(data["episode_id"]).astype(np.int64)
    for start, end in iter_episode_slices(episode_id):
        episode = int(episode_id[start])
        local_segments = positive_segments(label[start:end])
        event_counts[str(episode)] = len(local_segments)
        for seg_start, seg_end in local_segments:
            global_start = start + seg_start
            global_end = start + seg_end
            durations.append(float(time[global_end] - time[global_start]))
    return event_counts, float(np.mean(durations)) if durations else None


def _save_diagnosis_figure(path: Path, data: dict[str, np.ndarray]) -> None:
    """Save one episode plot showing whether label=1 aligns with observable signals."""
    time = np.asarray(data["time"], dtype=np.float64)
    label = np.asarray(data["label"], dtype=np.float64)
    episode_id = np.asarray(data["episode_id"]).astype(np.int64)
    tau_norm = np.linalg.norm(data["tau_ext"], axis=1)
    e_q = data["q_des"] - data["q"]
    e_norm = np.linalg.norm(e_q, axis=1)
    delta_e_norm = np.linalg.norm(compute_episodewise_delta(e_q, episode_id), axis=1)
    qdot_norm = np.linalg.norm(data["qdot"], axis=1)

    # contact가 있는 episode를 우선 선택해서 diagnosis figure가 실제 event를 보여주게 한다.
    chosen_start, chosen_end = iter_episode_slices(episode_id)[0]
    for start, end in iter_episode_slices(episode_id):
        if int(np.sum(label[start:end])) > 0:
            chosen_start, chosen_end = start, end
            break

    sl = slice(chosen_start, chosen_end)
    local_time = time[sl] - time[chosen_start]
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    series = [
        (tau_norm[sl], r"$||\tau_{ext}||$ [Nm]"),
        (label[sl], "label"),
        (e_norm[sl], r"$||e_q||$"),
        (delta_e_norm[sl], r"$||\Delta e_q||$"),
        (qdot_norm[sl], r"$||\dot{q}||$"),
    ]
    for ax, (values, ylabel) in zip(axes, series):
        ax.plot(local_time, values, linewidth=1.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time in episode [s]")
    fig.suptitle(f"Dataset diagnosis example, episode {int(episode_id[chosen_start])}")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def diagnose(split_data: dict[str, dict[str, np.ndarray]], eps: float) -> dict:
    """Compute dataset-level sanity checks before trusting training results."""
    episode_sets = _assert_episode_splits_disjoint(split_data)
    data = _concat_splits(split_data)
    label = np.asarray(data["label"]).astype(np.int64)
    episode_id = np.asarray(data["episode_id"]).astype(np.int64)
    tau_norm = np.linalg.norm(data["tau_ext"], axis=1)
    e_q = data["q_des"] - data["q"]
    e_norm = np.linalg.norm(e_q, axis=1)
    delta_e_norm = np.linalg.norm(compute_episodewise_delta(e_q, episode_id), axis=1)
    qdot_norm = np.linalg.norm(data["qdot"], axis=1)
    sat = np.asarray(data["is_saturated"]).astype(bool)

    contact_mask = label == 1
    no_contact_mask = label == 0
    event_counts, mean_duration = _event_counts_and_duration(data)
    tau_label = (tau_norm > float(eps)).astype(np.int64)
    # label leakage를 막는 것과 별개로, label 자체가 tau_ext와 동기화되어 있는지 확인한다.
    label_tau_sync_ok = bool(np.array_equal(tau_label, label))

    payload = {
        "total_sample_count": int(label.size),
        "contact_sample_count": int(np.sum(contact_mask)),
        "no_contact_sample_count": int(np.sum(no_contact_mask)),
        "contact_ratio": float(np.mean(contact_mask)) if label.size else 0.0,
        "episode_id_sets": episode_sets,
        "contact_event_count_per_episode": event_counts,
        "mean_contact_duration_s": mean_duration,
        "tau_ext_norm": _stats(tau_norm),
        "e_q_norm": _stats(e_norm),
        "delta_e_q_norm": _stats(delta_e_norm),
        "qdot_norm": _stats(qdot_norm),
        "contact_vs_no_contact": {
            "e_q_norm_contact": _stats(e_norm[contact_mask]),
            "e_q_norm_no_contact": _stats(e_norm[no_contact_mask]),
            "delta_e_q_norm_contact": _stats(delta_e_norm[contact_mask]),
            "delta_e_q_norm_no_contact": _stats(delta_e_norm[no_contact_mask]),
            "qdot_norm_contact": _stats(qdot_norm[contact_mask]),
            "qdot_norm_no_contact": _stats(qdot_norm[no_contact_mask]),
        },
        "saturation_ratio": float(np.mean(sat)) if sat.size else 0.0,
        "label_tau_ext_sync": {
            "eps": float(eps),
            "ok": label_tau_sync_ok,
            "mismatch_count": int(np.sum(tau_label != label)),
        },
        "warnings": [],
    }

    contact_delta = float(payload["contact_vs_no_contact"]["delta_e_q_norm_contact"]["mean"] or 0.0)
    no_contact_delta = float(payload["contact_vs_no_contact"]["delta_e_q_norm_no_contact"]["mean"] or 0.0)
    contact_error = float(payload["contact_vs_no_contact"]["e_q_norm_contact"]["mean"] or 0.0)
    no_contact_error = float(payload["contact_vs_no_contact"]["e_q_norm_no_contact"]["mean"] or 0.0)
    delta_gap = abs(contact_delta - no_contact_delta)
    error_gap = abs(contact_error - no_contact_error)
    reference = max(contact_delta, no_contact_delta, contact_error, no_contact_error, 1.0e-9)
    # contact/no-contact feature 분포가 너무 비슷하면 GRU가 배울 신호가 약하다는 뜻이다.
    if max(delta_gap, error_gap) < 0.05 * reference:
        warnings.warn(OBSERVABILITY_WARNING)
        payload["warnings"].append(OBSERVABILITY_WARNING)
    if not label_tau_sync_ok:
        warning = "WARNING: label and tau_ext norm are not synchronized. Regenerate the dataset."
        warnings.warn(warning)
        payload["warnings"].append(warning)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage override.")
    args = parser.parse_args()

    # 1) stage output directory에서 sim_train/sim_val/sim_test를 모두 읽는다.
    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    out_dirs = ensure_output_dirs(output_root(config))
    split_paths = {
        "train": out_dirs["datasets"] / "sim_train.npz",
        "val": out_dirs["datasets"] / "sim_val.npz",
        "test": out_dirs["datasets"] / "sim_test.npz",
    }
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files. Run generate_sim_dataset.py first: {missing}")

    # 2) diagnosis는 모든 split을 검사하지만, 학습처럼 섞어서 scaler를 fit하지는 않는다.
    split_data = {name: load_npz_dataset(path) for name, path in split_paths.items()}
    eps = float(config.get("simulation", {}).get("disturbance_label_eps", 1.0e-6))
    payload = diagnose(split_data, eps)
    payload["stage"] = config["experiment_stage"]
    save_json(out_dirs["metrics"] / "dataset_diagnosis.json", payload)
    _save_diagnosis_figure(out_dirs["figures"] / "dataset_diagnosis_example.png", _concat_splits(split_data))
    print(f"Saved dataset diagnosis to {out_dirs['metrics'] / 'dataset_diagnosis.json'}")


if __name__ == "__main__":
    main()
