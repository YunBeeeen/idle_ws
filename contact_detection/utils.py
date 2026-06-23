"""Shared utilities for the contact detection pipeline."""

from __future__ import annotations

# 이 파일은 pipeline 전체에서 공유하는 "공구함"이다.
# config loading, path handling, feature construction, scaler, metrics, plotting helper가 들어 있다.
import csv
import json
import pickle
import random
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# matplotlib은 모든 figure 저장에 사용한다. GUI가 없는 ROS/서버 터미널에서도 저장되도록 Agg backend를 강제한다.
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REQUIRED_DATASET_KEYS = (
    "time",
    "q",
    "qdot",
    "q_des",
    "qdot_des",
    "tau_cmd_raw",
    "tau_cmd",
    "tau_ext",
    "label",
    "episode_id",
    "is_saturated",
    "joint_names",
)


STAGE_ORDER = ("easy_hold", "sine_no_randomization", "randomized_sim", "low_torque_analysis", "small_disturbance")


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and attach path metadata."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    inherited = config.pop("inherits", None)
    if inherited is not None:
        inherited_path = Path(str(inherited)).expanduser()
        if not inherited_path.is_absolute():
            inherited_path = path.parent / inherited_path
        base_config = load_config(inherited_path)
        config = deep_update(base_config, config)
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    config.setdefault("seed", 42)
    config.setdefault("experiment_stage", "easy_hold")
    return config


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively update a config dict in place."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_stage_config(config: dict, stage: str | None = None) -> dict:
    """Apply a curriculum-stage override and record the active stage."""

    stage_name = str(stage or config.get("experiment_stage", "easy_hold"))
    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Unsupported experiment stage: {stage_name}. Use one of {list(STAGE_ORDER)}")
    stage_overrides = config.get("stages", {}).get(stage_name, {})
    if stage_overrides:
        deep_update(config, stage_overrides)
    config["experiment_stage"] = stage_name
    config["_stage"] = stage_name
    return config


def config_dir(config: dict) -> Path:
    return Path(config["_config_dir"]).resolve()


def project_root(config: dict) -> Path:
    return config_dir(config)


def output_root(config: dict) -> Path:
    output_dir = config.get("output_dir", "outputs")
    root = resolve_path(str(output_dir), project_root(config))
    if bool(config.get("stage_output_dirs", True)):
        stage = config.get("_stage", config.get("experiment_stage"))
        if stage:
            root = root / str(stage)
    return root


def ensure_output_dirs(root: str | Path) -> dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    dirs = {
        "root": root_path,
        "datasets": root_path / "datasets",
        "models": root_path / "models",
        "figures": root_path / "figures",
        "metrics": root_path / "metrics",
        "real_inference": root_path / "real_inference",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_path(path_text: str, base_dir: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def save_json(path: str | Path, payload: dict) -> None:
    path_obj = Path(path).expanduser().resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict:
    path_obj = Path(path).expanduser().resolve()
    with path_obj.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _strip_private_config_fields(config: dict) -> dict:
    cleaned: dict = {}
    for key, value in config.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_private_config_fields(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_private_config_fields(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def save_config_yaml(path: str | Path, config: dict) -> None:
    path_obj = Path(path).expanduser().resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = _strip_private_config_fields(config)
    with path_obj.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def select_torch_device(torch_module, requested: str | None = None):
    """Return a torch device from config, falling back clearly when CUDA is unavailable."""

    request = (requested or "auto").strip().lower()
    if request in ("", "auto"):
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        return torch_module.device("cpu")
    if request.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                f"training.device='{requested}' was requested, but torch.cuda.is_available() is false. "
                "Check nvidia-smi, driver installation, and whether this shell was opened after driver setup."
            )
        return torch_module.device(request)
    if request == "cpu":
        return torch_module.device("cpu")
    raise ValueError(f"Unsupported torch device setting: {requested!r}. Use 'auto', 'cpu', or 'cuda'.")


def ensure_vector_length(name: str, values: Iterable[float], dof: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.shape != (dof,):
        raise ValueError(f"{name} must have length {dof}, got shape {arr.shape}")
    return arr


def robot_joint_names(config: dict) -> list[str]:
    dof = int(config["robot"]["dof"])
    names = [str(name) for name in config["robot"].get("joint_names", [f"j{idx + 1}" for idx in range(dof)])]
    if len(names) != dof:
        raise ValueError(f"robot.joint_names must have length {dof}, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError(f"robot.joint_names contains duplicates: {names}")
    return names


def real_csv_joint_order(config: dict) -> list[str]:
    dof = int(config["robot"]["dof"])
    order = [str(name) for name in config["robot"].get("real_csv_joint_order", robot_joint_names(config))]
    if len(order) != dof:
        raise ValueError(f"robot.real_csv_joint_order must have length {dof}, got {len(order)}")
    if sorted(order) != sorted(robot_joint_names(config)):
        raise ValueError(
            "robot.real_csv_joint_order must contain the same names as robot.joint_names: "
            f"{order} vs {robot_joint_names(config)}"
        )
    return order


def reorder_matrix_by_joint_order(matrix: np.ndarray, source_order: list[str], target_order: list[str], name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(source_order):
        raise ValueError(f"{name} must be [N, {len(source_order)}], got {arr.shape}")
    index_by_source = {joint: idx for idx, joint in enumerate(source_order)}
    missing = [joint for joint in target_order if joint not in index_by_source]
    if missing:
        raise ValueError(f"{name} source order missing joints: {missing}")
    indices = [index_by_source[joint] for joint in target_order]
    if source_order != target_order:
        warnings.warn(f"Reordered {name} from CSV joint order {source_order} to model order {target_order}.")
    return arr[:, indices]


def validate_motor_joint_map(config: dict) -> None:
    joint_names = robot_joint_names(config)
    mapping = config["robot"].get("motor_joint_map", {str(idx + 1): joint for idx, joint in enumerate(joint_names)})
    ordered = [str(mapping.get(str(idx + 1), "")) for idx in range(len(joint_names))]
    if ordered != joint_names:
        raise ValueError(
            "robot.motor_joint_map sorted by motor id must match robot.joint_names. "
            f"motor map gives {ordered}, expected {joint_names}"
        )


FEATURE_MODE_ORIGINAL = "original_42"
RESIDUAL_FEATURE_MODES = {
    "residual_v1",
    "residual_v2",
    "residual_v3",
    "residual_cmd_v1",
    "real_cmd_error_v1",
}
REAL_FEATURE_MODES = {
    # 실물 로봇에서는 steady-state tracking error가 contact처럼 보일 수 있어서
    # e_q / delta_e_q를 뺀 ablation을 별도 feature mode로 둔다.
    "real_no_eq_v1",
    # delta_qdot까지 과민하게 튀는지 확인하기 위한 더 보수적인 ablation.
    "real_no_eq_no_dqdot_v1",
}
SUPPORTED_FEATURE_MODES = {FEATURE_MODE_ORIGINAL, *RESIDUAL_FEATURE_MODES, *REAL_FEATURE_MODES}


FEATURE_MODE_BLOCKS: dict[str, tuple[str, ...]] = {
    "residual_v1": ("qdot", "tau_cmd", "tau_residual_corrected", "delta_tau_residual", "delta_e_q", "delta_qdot"),
    "residual_v2": ("tau_residual_corrected", "delta_tau_residual", "qdot", "delta_qdot"),
    "residual_v3": ("q", "qdot", "tau_cmd", "tau_residual_corrected", "delta_tau_residual", "delta_e_q", "delta_qdot"),
    "residual_cmd_v1": (
        "tau_cmd",
        "delta_tau_cmd",
        "tau_residual_corrected",
        "delta_tau_residual",
        "qdot",
        "delta_qdot",
    ),
    "real_cmd_error_v1": (
        "tau_cmd",
        "delta_tau_cmd",
        "tau_residual_corrected",
        "delta_tau_residual",
        "qdot",
        "delta_qdot",
        "e_q",
        "delta_e_q",
    ),
    "real_no_eq_v1": (
        "q",
        "qdot",
        "tau_cmd",
        "delta_qdot",
        "delta_tau_cmd",
    ),
    "real_no_eq_no_dqdot_v1": (
        "q",
        "qdot",
        "tau_cmd",
        "delta_tau_cmd",
    ),
}


def normalize_feature_mode(feature_mode: str | None) -> str:
    mode = str(feature_mode or FEATURE_MODE_ORIGINAL).strip()
    if mode in {"", "original", "legacy", "original_42d"}:
        mode = FEATURE_MODE_ORIGINAL
    if mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(
            f"Unsupported dataset.feature_mode={feature_mode!r}. "
            f"Use one of {sorted(SUPPORTED_FEATURE_MODES)}."
        )
    return mode


def feature_names(dof: int, use_delta_features: bool = False, feature_mode: str = FEATURE_MODE_ORIGINAL) -> list[str]:
    mode = normalize_feature_mode(feature_mode)
    names: list[str] = []
    if mode == FEATURE_MODE_ORIGINAL:
        for prefix in ("q", "qdot", "e_q", "tau_cmd"):
            names.extend([f"{prefix}{idx + 1}" for idx in range(dof)])
        if use_delta_features:
            for prefix in ("delta_e_q", "delta_qdot", "delta_tau_cmd"):
                names.extend([f"{prefix}{idx + 1}" for idx in range(dof)])
        return names

    for prefix in FEATURE_MODE_BLOCKS[mode]:
        names.extend([f"{prefix}{idx + 1}" for idx in range(dof)])
    return names


def compute_episodewise_delta(values: np.ndarray, episode_id: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    delta = np.zeros_like(arr)
    if arr.shape[0] <= 1:
        return delta
    if episode_id is None:
        delta[1:] = arr[1:] - arr[:-1]
        return delta
    ep = np.asarray(episode_id)
    same_prev = ep[1:] == ep[:-1]
    delta[1:][same_prev] = arr[1:][same_prev] - arr[:-1][same_prev]
    return delta


def build_input_features(
    q: np.ndarray,
    qdot: np.ndarray,
    q_des: np.ndarray,
    tau_cmd: np.ndarray,
    use_delta_features: bool = False,
    episode_id: np.ndarray | None = None,
    tau_residual: np.ndarray | None = None,
    tau_residual_corrected: np.ndarray | None = None,
    feature_mode: str = FEATURE_MODE_ORIGINAL,
) -> tuple[np.ndarray, list[str]]:
    """Build the model input matrix.

    기본 ``original_42``는 기존 x_t = [q, qdot, e_q, tau_cmd, ...] 흐름이다.
    residual feature mode는 실제 로봇에서 얻을 수 있는 ``tau_meas - tau_cmd``
    기반 residual만 받는다.  이 함수는 여전히 tau_ext를 인자로 받지 않는다.
    """

    mode = normalize_feature_mode(feature_mode)
    q_arr = np.asarray(q, dtype=np.float64)
    qdot_arr = np.asarray(qdot, dtype=np.float64)
    q_des_arr = np.asarray(q_des, dtype=np.float64)
    tau_cmd_arr = np.asarray(tau_cmd, dtype=np.float64)
    if q_arr.ndim != 2:
        raise ValueError(f"q must be [N, dof], got shape {q_arr.shape}")
    expected_shape = q_arr.shape
    for name, arr in (
        ("qdot", qdot_arr),
        ("q_des", q_des_arr),
        ("tau_cmd", tau_cmd_arr),
    ):
        if arr.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {arr.shape}")

    e_q = q_des_arr - q_arr
    if mode == FEATURE_MODE_ORIGINAL:
        feature_blocks = [q_arr, qdot_arr, e_q, tau_cmd_arr]
        if use_delta_features:
            # Delta feature는 episode_id가 같은 연속 sample에 대해서만 계산한다.
            # episode가 바뀌는 첫 sample은 zero delta가 되므로 GRU window가
            # 서로 다른 episode를 이어붙여 보는 leakage를 방지한다.
            feature_blocks.extend(
                [
                    compute_episodewise_delta(e_q, episode_id),
                    compute_episodewise_delta(qdot_arr, episode_id),
                    compute_episodewise_delta(tau_cmd_arr, episode_id),
                ]
            )
        features = np.concatenate(feature_blocks, axis=1)
        return features, feature_names(q_arr.shape[1], use_delta_features, mode)

    delta_e_q = compute_episodewise_delta(e_q, episode_id)
    delta_qdot = compute_episodewise_delta(qdot_arr, episode_id)
    delta_tau_cmd = compute_episodewise_delta(tau_cmd_arr, episode_id)
    blocks_by_name = {
        "q": q_arr,
        "qdot": qdot_arr,
        "tau_cmd": tau_cmd_arr,
        "delta_tau_cmd": delta_tau_cmd,
        "e_q": e_q,
        "delta_e_q": delta_e_q,
        "delta_qdot": delta_qdot,
    }

    mode_blocks = FEATURE_MODE_BLOCKS[mode]
    needs_residual = "tau_residual_corrected" in mode_blocks or "delta_tau_residual" in mode_blocks
    if needs_residual:
        tau_res_arr = tau_residual_corrected if tau_residual_corrected is not None else tau_residual
        if tau_res_arr is None:
            raise ValueError(
                f"dataset.feature_mode={mode!r} requires tau_residual_corrected or tau_residual. "
                "For real robot logs this is computed from tau_meas - tau_cmd. tau_ext is not a valid substitute."
            )
        tau_res_arr = np.asarray(tau_res_arr, dtype=np.float64)
        if tau_res_arr.shape != expected_shape:
            raise ValueError(f"tau_residual shape mismatch: expected {expected_shape}, got {tau_res_arr.shape}")
        blocks_by_name["tau_residual_corrected"] = tau_res_arr
        blocks_by_name["delta_tau_residual"] = compute_episodewise_delta(tau_res_arr, episode_id)

    features = np.concatenate([blocks_by_name[name] for name in mode_blocks], axis=1)
    names = feature_names(q_arr.shape[1], use_delta_features, mode)
    if any("tau_ext" in name for name in names):
        raise RuntimeError("tau_ext must never appear in model feature_names.")
    return features, names


def compute_residual_offset_corrected(
    time: np.ndarray,
    tau_residual: np.ndarray,
    episode_id: np.ndarray | None = None,
    contact_label: np.ndarray | None = None,
    offset_duration_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract per-episode initial no-contact residual offset.

    ``tau_residual`` is expected to be ``tau_meas - tau_cmd``.  The offset is
    estimated from the first ``offset_duration_s`` seconds of each episode,
    restricted to label 0 when contact labels are available.
    """

    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    residual_arr = np.asarray(tau_residual, dtype=np.float64)
    if residual_arr.ndim != 2 or residual_arr.shape[0] != time_arr.shape[0]:
        raise ValueError(f"tau_residual must be [N, dof], got {residual_arr.shape} for time {time_arr.shape}")
    if episode_id is None:
        episode_arr = np.zeros(time_arr.shape[0], dtype=np.int64)
    else:
        episode_arr = np.asarray(episode_id).reshape(-1)
        if episode_arr.shape[0] != time_arr.shape[0]:
            raise ValueError("episode_id length must match time for residual offset correction")
    if contact_label is None:
        label_arr = np.zeros(time_arr.shape[0], dtype=np.int64)
    else:
        label_arr = np.asarray(contact_label).astype(np.int64).reshape(-1)
        if label_arr.shape[0] != time_arr.shape[0]:
            raise ValueError("contact_label length must match time for residual offset correction")

    corrected = residual_arr.copy()
    offsets = np.zeros((len(iter_episode_slices(episode_arr)), residual_arr.shape[1]), dtype=np.float64)
    for row_idx, (start, end) in enumerate(iter_episode_slices(episode_arr)):
        local_time = time_arr[start:end]
        local_time = local_time - float(local_time[0])
        offset_mask = local_time <= float(offset_duration_s)
        offset_mask &= label_arr[start:end] == 0
        if not np.any(offset_mask):
            offset_mask = np.ones(end - start, dtype=bool)
        offset = np.mean(residual_arr[start:end][offset_mask], axis=0)
        offsets[row_idx] = offset
        corrected[start:end] = residual_arr[start:end] - offset
    return corrected, offsets


@dataclass
class StandardScaler:
    """Lightweight NumPy standard scaler."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Scaler expects a 2D array, got shape {arr.shape}")
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.maximum(std, 1.0e-8)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return (arr - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return arr * self.std + self.mean

    def save(self, path: str | Path) -> None:
        path_obj = Path(path).expanduser().resolve()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as stream:
            pickle.dump({"mean": self.mean, "std": self.std}, stream)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler":
        path_obj = Path(path).expanduser().resolve()
        with path_obj.open("rb") as stream:
            payload = pickle.load(stream)
        return cls(mean=np.asarray(payload["mean"]), std=np.asarray(payload["std"]))


def load_npz_dataset(path: str | Path) -> dict[str, np.ndarray]:
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path_obj}")
    payload = np.load(path_obj, allow_pickle=False)
    data = {key: payload[key] for key in payload.files}
    missing = [key for key in REQUIRED_DATASET_KEYS if key not in data]
    if missing:
        raise KeyError(f"Dataset {path_obj} is missing required keys: {missing}")
    return data


def contact_region_names(config: dict) -> list[str]:
    region_cfg = config.get("contact_regions", {})
    names = list(region_cfg.get("names", ["no_contact"]))
    if not names or str(names[0]) != "no_contact":
        raise ValueError("contact_regions.names must start with 'no_contact'")
    return [str(name) for name in names]


def frame_to_region_id(config: dict) -> dict[str, int]:
    names = contact_region_names(config)
    name_to_id = {name: idx for idx, name in enumerate(names)}
    mapping = config.get("contact_regions", {}).get("frame_to_region", {})
    if not isinstance(mapping, dict):
        raise ValueError("contact_regions.frame_to_region must be a mapping")
    result: dict[str, int] = {}
    for frame_name, region_name in mapping.items():
        if str(region_name) not in name_to_id:
            raise ValueError(f"Unknown contact region '{region_name}' for frame '{frame_name}'")
        result[str(frame_name)] = int(name_to_id[str(region_name)])
    return result


def region_confusion_metrics(
    true_regions: np.ndarray,
    pred_regions: np.ndarray,
    num_regions: int,
) -> dict[str, object]:
    true_arr = np.asarray(true_regions).astype(np.int64).reshape(-1)
    pred_arr = np.asarray(pred_regions).astype(np.int64).reshape(-1)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"Region shape mismatch: {true_arr.shape} vs {pred_arr.shape}")
    confusion = np.zeros((int(num_regions), int(num_regions)), dtype=np.int64)
    valid = (true_arr >= 0) & (true_arr < int(num_regions)) & (pred_arr >= 0) & (pred_arr < int(num_regions))
    for true_value, pred_value in zip(true_arr[valid], pred_arr[valid]):
        confusion[int(true_value), int(pred_value)] += 1
    total = int(np.sum(confusion))
    accuracy = float(np.trace(confusion) / max(total, 1))
    per_region_recall: dict[str, float | None] = {}
    for region_idx in range(int(num_regions)):
        support = int(np.sum(confusion[region_idx, :]))
        per_region_recall[str(region_idx)] = None if support == 0 else float(confusion[region_idx, region_idx] / support)
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "support": confusion.sum(axis=1).astype(int).tolist(),
        "per_region_recall": per_region_recall,
    }


def iter_episode_slices(episode_id: np.ndarray) -> list[tuple[int, int]]:
    episode_arr = np.asarray(episode_id)
    if episode_arr.ndim != 1:
        raise ValueError(f"episode_id must be 1D, got shape {episode_arr.shape}")
    if episode_arr.size == 0:
        return []
    slices: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, episode_arr.size):
        if episode_arr[idx] != episode_arr[idx - 1]:
            slices.append((start, idx))
            start = idx
    slices.append((start, episode_arr.size))
    return slices


def sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    clipped = np.clip(arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(labels).astype(np.int64).reshape(-1)
    y_pred = np.asarray(predictions).astype(np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Label/prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    false_negative_rate = fn / max(fn + tp, 1)
    f1 = 0.0
    if (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
    }


def search_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    num_points: int = 200,
) -> dict[str, float | dict]:
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels).astype(np.int64).reshape(-1)
    if score_arr.shape != label_arr.shape:
        raise ValueError(f"Score/label shape mismatch: {score_arr.shape} vs {label_arr.shape}")
    if score_arr.size == 0:
        raise ValueError("Cannot search threshold on empty arrays")

    min_score = float(np.min(score_arr))
    max_score = float(np.max(score_arr))
    if np.isclose(min_score, max_score):
        candidates = np.asarray([min_score], dtype=np.float64)
    else:
        unique = np.unique(score_arr)
        if unique.size <= num_points:
            candidates = unique
        else:
            candidates = np.linspace(min_score, max_score, num_points, dtype=np.float64)

    best_threshold = float(candidates[0])
    best_metrics = binary_classification_metrics(label_arr, score_arr >= best_threshold)
    best_key = (
        best_metrics["f1"],
        best_metrics["precision"],
        best_metrics["recall"],
        -best_threshold,
    )
    for threshold in candidates[1:]:
        metrics = binary_classification_metrics(label_arr, score_arr >= threshold)
        key = (metrics["f1"], metrics["precision"], metrics["recall"], -float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return {"threshold": best_threshold, "metrics": best_metrics}


def fbeta_from_metrics(metrics: dict[str, float | int], beta: float = 2.0) -> float:
    """Return F-beta score from a metrics dictionary.

    F1 treats precision and recall equally.  For this project, missed contacts
    can be more important than false alarms, so F2 is useful because it gives
    recall more weight while still penalizing a detector that always predicts
    contact.
    """

    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    beta_sq = float(beta) ** 2
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denom)


def search_threshold_with_policy(
    scores: np.ndarray,
    labels: np.ndarray,
    num_points: int = 200,
    selection_policy: str = "f1",
    target_recall: float = 0.85,
    fbeta_beta: float = 2.0,
) -> dict[str, float | bool | dict | str]:
    """Select a binary operating threshold using a validation-set policy.

    Supported policies:
    - ``f1``: maximize F1-score, the original balanced operating point.
    - ``f2``: maximize F2-score, a recall-weighted operating point.
    - ``recall_constrained_f1``: require recall >= target_recall, then
      maximize F1-score.  This is the recommended setting when the detector
      should miss as few contacts as possible while avoiding the trivial
      "always contact" solution.
    """

    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels).astype(np.int64).reshape(-1)
    if score_arr.shape != label_arr.shape:
        raise ValueError(f"Score/label shape mismatch: {score_arr.shape} vs {label_arr.shape}")
    if score_arr.size == 0:
        raise ValueError("Cannot search threshold on empty arrays")

    min_score = float(np.min(score_arr))
    max_score = float(np.max(score_arr))
    if np.isclose(min_score, max_score):
        candidates = np.asarray([min_score], dtype=np.float64)
    else:
        unique = np.unique(score_arr)
        if unique.size <= num_points:
            candidates = unique
        else:
            candidates = np.linspace(min_score, max_score, num_points, dtype=np.float64)

    policy = str(selection_policy)
    if policy not in {"f1", "f2", "recall_constrained_f1"}:
        raise ValueError(
            "selection_policy must be one of: f1, f2, recall_constrained_f1; "
            f"got {selection_policy!r}"
        )

    best_threshold = float(candidates[0])
    best_metrics = binary_classification_metrics(label_arr, score_arr >= best_threshold)
    best_score = -np.inf
    best_key: tuple[float, ...] | None = None
    target_satisfied = False
    fallback_threshold = best_threshold
    fallback_metrics = best_metrics
    fallback_key: tuple[float, ...] | None = None

    for threshold in candidates:
        metrics = binary_classification_metrics(label_arr, score_arr >= float(threshold))
        f2 = fbeta_from_metrics(metrics, beta=fbeta_beta)
        if policy == "f1":
            selection_score = float(metrics["f1"])
            key = (selection_score, float(metrics["recall"]), float(metrics["precision"]), -float(threshold))
            valid = True
        elif policy == "f2":
            selection_score = f2
            key = (selection_score, float(metrics["f1"]), float(metrics["recall"]), -float(threshold))
            valid = True
        else:
            # Primary goal: satisfy the minimum recall.  Among thresholds that
            # do so, keep F1 as high as possible to avoid a useless always-on
            # detector.
            selection_score = float(metrics["f1"])
            valid = float(metrics["recall"]) + 1.0e-12 >= float(target_recall)
            key = (selection_score, float(metrics["recall"]), float(metrics["precision"]), -float(threshold))
            fallback_candidate_key = (
                float(metrics["recall"]),
                float(metrics["f1"]),
                float(metrics["precision"]),
                -float(threshold),
            )
            if fallback_key is None or fallback_candidate_key > fallback_key:
                fallback_key = fallback_candidate_key
                fallback_threshold = float(threshold)
                fallback_metrics = metrics

        if not valid:
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = float(selection_score)
            target_satisfied = True

    if policy == "recall_constrained_f1" and not target_satisfied:
        best_threshold = float(fallback_threshold)
        best_metrics = fallback_metrics
        best_score = float(best_metrics["f1"])

    best_metrics = dict(best_metrics)
    best_metrics["f2"] = fbeta_from_metrics(best_metrics, beta=fbeta_beta)
    return {
        "threshold": float(best_threshold),
        "metrics": best_metrics,
        "selection_policy": policy,
        "selection_score": float(best_score),
        "target_recall": float(target_recall),
        "target_recall_satisfied": bool(target_satisfied or policy != "recall_constrained_f1"),
        "fbeta_beta": float(fbeta_beta),
    }


def threshold_score_from_data(data: dict[str, np.ndarray], indices: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    """Compute a scalar threshold-baseline score at selected sample indices."""

    metric_cfg = config.get("threshold", {})
    metric = str(metric_cfg.get("metric", "error_norm"))
    episode_id = data.get("episode_id")
    e_q = np.asarray(data["q_des"], dtype=np.float64) - np.asarray(data["q"], dtype=np.float64)
    delta_e_q = compute_episodewise_delta(e_q, episode_id)
    e_norm = np.linalg.norm(e_q, axis=1)
    delta_norm = np.linalg.norm(delta_e_q, axis=1)
    if metric == "error_norm":
        score = e_norm
    elif metric == "delta_error_norm":
        score = delta_norm
    elif metric == "combined":
        alpha = float(metric_cfg.get("alpha", 1.0))
        beta = float(metric_cfg.get("beta", 1.0))
        score = alpha * e_norm + beta * delta_norm
    else:
        raise ValueError("threshold.metric must be one of: error_norm, delta_error_norm, combined")
    meta = {
        "threshold_metric": metric,
        "alpha": float(metric_cfg.get("alpha", 1.0)),
        "beta": float(metric_cfg.get("beta", 1.0)),
    }
    return score[np.asarray(indices, dtype=np.int64)], meta


def positive_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    y = np.asarray(labels).astype(np.int64).reshape(-1)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(y):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, y.size - 1))
    return segments


def compute_detection_delay(
    time: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    episode_id: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels).astype(np.int64).reshape(-1)
    pred_arr = np.asarray(predictions).astype(np.int64).reshape(-1)
    if time_arr.shape != label_arr.shape or time_arr.shape != pred_arr.shape:
        raise ValueError(
            f"time/labels/predictions shape mismatch: {time_arr.shape}, {label_arr.shape}, {pred_arr.shape}"
        )

    if episode_id is None:
        slices = [(0, label_arr.size)]
    else:
        slices = iter_episode_slices(np.asarray(episode_id).reshape(-1))

    delays: list[float] = []
    missed = 0
    num_events = 0
    for start, end in slices:
        local_segments = positive_segments(label_arr[start:end])
        num_events += len(local_segments)
        for seg_start, seg_end in local_segments:
            global_start = start + seg_start
            global_end = start + seg_end
            hits = np.flatnonzero(pred_arr[global_start : global_end + 1] == 1)
            if hits.size == 0:
                missed += 1
                continue
            detected_idx = global_start + int(hits[0])
            delays.append(float(time_arr[detected_idx] - time_arr[global_start]))

    if num_events == 0:
        return {"num_events": 0, "detected_events": 0, "missed_events": 0, "mean_delay_s": None}
    mean_delay = float(np.mean(delays)) if delays else None
    return {
        "num_events": int(num_events),
        "detected_events": int(len(delays)),
        "missed_events": int(missed),
        "mean_delay_s": mean_delay,
    }


def infer_qdot_from_q(q: np.ndarray, time: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64)
    time_arr = np.asarray(time, dtype=np.float64)
    if q_arr.ndim != 2:
        raise ValueError(f"q must be [N, dof], got shape {q_arr.shape}")
    if time_arr.ndim != 1 or time_arr.shape[0] != q_arr.shape[0]:
        raise ValueError(f"time shape mismatch for q differentiation: {time_arr.shape} vs {q_arr.shape}")
    if q_arr.shape[0] <= 1:
        return np.zeros_like(q_arr)
    qdot = np.zeros_like(q_arr)
    for axis in range(q_arr.shape[1]):
        qdot[:, axis] = np.gradient(q_arr[:, axis], time_arr, edge_order=1)
    return qdot


def estimate_dt(time: np.ndarray) -> float:
    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    if time_arr.size < 2:
        raise ValueError("Need at least two time samples to estimate dt")
    diffs = np.diff(time_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        raise ValueError("Could not estimate positive dt from time column")
    return float(np.median(diffs))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    win = int(window)
    if win <= 1:
        return arr
    kernel = np.ones(win, dtype=np.float64) / float(win)
    out = np.zeros_like(arr)
    for col in range(arr.shape[1]):
        out[:, col] = np.convolve(arr[:, col], kernel, mode="same")
    return out


def resample_series(time: np.ndarray, arrays: dict[str, np.ndarray], target_dt: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    if time_arr.size < 2:
        return time_arr, arrays
    target = float(target_dt)
    if target <= 0.0:
        raise ValueError(f"real_inference.target_dt must be positive, got {target}")
    new_time = np.arange(time_arr[0], time_arr[-1] + 0.5 * target, target, dtype=np.float64)
    new_arrays: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        arr = np.asarray(values)
        if arr.ndim == 1:
            new_arrays[name] = np.interp(new_time, time_arr, arr.astype(np.float64))
        elif arr.ndim == 2:
            cols = [np.interp(new_time, time_arr, arr[:, idx].astype(np.float64)) for idx in range(arr.shape[1])]
            new_arrays[name] = np.stack(cols, axis=1)
        else:
            raise ValueError(f"Cannot resample {name} with shape {arr.shape}")
    return new_time, new_arrays


def _extract_required_matrix(rows: list[dict[str, str]], columns: list[str], csv_path: Path) -> np.ndarray:
    values: list[list[float]] = []
    for row_idx, row in enumerate(rows):
        row_values: list[float] = []
        for column in columns:
            if column not in row or row[column] == "":
                raise ValueError(f"Missing column '{column}' at row {row_idx + 2} in {csv_path}")
            row_values.append(float(row[column]))
        values.append(row_values)
    return np.asarray(values, dtype=np.float64)


def _fixed_target_from_config(config: dict, dof: int) -> np.ndarray | None:
    real_cfg = config.get("real_inference", {})
    fixed = real_cfg.get("fixed_target_q_des")
    if fixed is None:
        return None
    return ensure_vector_length("real_inference.fixed_target_q_des", fixed, dof)


def _matrix_from_wide_columns(
    rows: list[dict[str, str]],
    headers: list[str],
    prefix: str,
    joint_order: list[str],
    csv_path: Path,
) -> np.ndarray | None:
    numbered = [f"{prefix}{idx + 1}" for idx in range(len(joint_order))]
    if all(column in headers for column in numbered):
        return _extract_required_matrix(rows, numbered, csv_path)
    named = [f"{prefix}_{joint}" for joint in joint_order]
    if all(column in headers for column in named):
        return _extract_required_matrix(rows, named, csv_path)
    return None


def load_real_log_csv(
    csv_path: str | Path,
    config: dict,
    allow_zero_tau_cmd_override: bool | None = None,
) -> dict[str, np.ndarray]:
    """Load a real robot log in either wide or long format."""

    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Real log CSV not found: {path}")
    dof = int(config["robot"]["dof"])
    joint_names = robot_joint_names(config)
    csv_joint_order = real_csv_joint_order(config)
    fixed_target = _fixed_target_from_config(config, dof)
    real_cfg = config.get("real_inference", {})
    allow_zero_tau_cmd = (
        bool(real_cfg.get("allow_zero_tau_cmd", False))
        if allow_zero_tau_cmd_override is None
        else bool(allow_zero_tau_cmd_override)
    )

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        headers = list(reader.fieldnames)
        rows = list(reader)

    time_column = "time" if "time" in headers else "timestamp" if "timestamp" in headers else None
    if time_column is None:
        raise ValueError(f"CSV must contain either 'time' or 'timestamp': {path}")

    q_wide = _matrix_from_wide_columns(rows, headers, "q", csv_joint_order, path)
    qdot_des = None
    tau_meas = None
    tau_ff = None
    kp = None
    kd = None
    tau_residual = None
    tau_residual_corrected = None
    contact_label = None
    episode_id = None
    motion_type = None
    if q_wide is not None:
        time = np.asarray([float(row[time_column]) for row in rows], dtype=np.float64)
        q = reorder_matrix_by_joint_order(q_wide, csv_joint_order, joint_names, "q")
        qdot = _matrix_from_wide_columns(rows, headers, "qdot", csv_joint_order, path)
        if qdot is None:
            qdot = _matrix_from_wide_columns(rows, headers, "qd", csv_joint_order, path)
        if qdot is None:
            warnings.warn("qdot columns missing in real log, using numerical differentiation from q.")
        else:
            qdot = reorder_matrix_by_joint_order(qdot, csv_joint_order, joint_names, "qdot")

        q_des_wide = _matrix_from_wide_columns(rows, headers, "q_des", csv_joint_order, path)
        if q_des_wide is not None:
            q_des = reorder_matrix_by_joint_order(q_des_wide, csv_joint_order, joint_names, "q_des")
        elif fixed_target is not None:
            q_des = np.repeat(fixed_target[None, :], q.shape[0], axis=0)
        else:
            raise ValueError(
                "q_des columns are missing. Provide q_des1..q_des6 in the CSV or set "
                "'real_inference.fixed_target_q_des' in the config."
            )

        qdot_des_wide = _matrix_from_wide_columns(rows, headers, "qdot_des", csv_joint_order, path)
        if qdot_des_wide is None:
            qdot_des_wide = _matrix_from_wide_columns(rows, headers, "qd_des", csv_joint_order, path)
        if qdot_des_wide is not None:
            qdot_des = reorder_matrix_by_joint_order(qdot_des_wide, csv_joint_order, joint_names, "qdot_des")
        else:
            qdot_des = np.zeros_like(q)

        tau_wide = _matrix_from_wide_columns(rows, headers, "tau_cmd", csv_joint_order, path)
        if tau_wide is not None:
            tau_cmd = reorder_matrix_by_joint_order(tau_wide, csv_joint_order, joint_names, "tau_cmd")
        elif allow_zero_tau_cmd:
            warnings.warn("tau_cmd columns missing in real log, filling tau_cmd with zeros.")
            tau_cmd = np.zeros_like(q)
        else:
            raise ValueError(
                "tau_cmd columns are missing. Provide tau_cmd1..tau_cmd6 or set "
                "'real_inference.allow_zero_tau_cmd: true' to force zero fill."
            )
        tau_meas_wide = _matrix_from_wide_columns(rows, headers, "tau_meas", csv_joint_order, path)
        if tau_meas_wide is not None:
            tau_meas = reorder_matrix_by_joint_order(tau_meas_wide, csv_joint_order, joint_names, "tau_meas")
        tau_ff_wide = _matrix_from_wide_columns(rows, headers, "tau_ff", csv_joint_order, path)
        if tau_ff_wide is not None:
            tau_ff = reorder_matrix_by_joint_order(tau_ff_wide, csv_joint_order, joint_names, "tau_ff")
        kp_wide = _matrix_from_wide_columns(rows, headers, "kp", csv_joint_order, path)
        if kp_wide is not None:
            kp = reorder_matrix_by_joint_order(kp_wide, csv_joint_order, joint_names, "kp")
        kd_wide = _matrix_from_wide_columns(rows, headers, "kd", csv_joint_order, path)
        if kd_wide is not None:
            kd = reorder_matrix_by_joint_order(kd_wide, csv_joint_order, joint_names, "kd")
        tau_res_wide = _matrix_from_wide_columns(rows, headers, "tau_residual", csv_joint_order, path)
        if tau_res_wide is not None:
            tau_residual = reorder_matrix_by_joint_order(tau_res_wide, csv_joint_order, joint_names, "tau_residual")
        tau_res_corr_wide = _matrix_from_wide_columns(rows, headers, "tau_residual_corrected", csv_joint_order, path)
        if tau_res_corr_wide is not None:
            tau_residual_corrected = reorder_matrix_by_joint_order(
                tau_res_corr_wide,
                csv_joint_order,
                joint_names,
                "tau_residual_corrected",
            )
        label_column = "contact_label" if "contact_label" in headers else "label" if "label" in headers else None
        if label_column is not None:
            contact_label = np.asarray([float(row[label_column]) for row in rows], dtype=np.float64)
        if "episode_id" in headers:
            episode_id = np.asarray([int(float(row["episode_id"])) for row in rows], dtype=np.int64)
        if "motion_type" in headers:
            motion_type = np.asarray([str(row["motion_type"]) for row in rows], dtype="<U64")
    elif {"motor_id", "q"}.issubset(headers):
        qdot_column = "qd" if "qd" in headers else "qdot" if "qdot" in headers else None
        if qdot_column is None:
            warnings.warn("Long-format log has no qd/qdot column, using numerical differentiation after pivot.")
        tau_column = "tau_cmd" if "tau_cmd" in headers else None
        grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        for row in rows:
            grouped.setdefault(row[time_column], []).append(row)

        time_values: list[float] = []
        q_list: list[list[float]] = []
        qdot_list: list[list[float]] = []
        tau_list: list[list[float]] = []
        dropped_groups = 0

        for raw_time, group_rows in grouped.items():
            by_motor: dict[int, dict[str, str]] = {}
            for row in group_rows:
                motor_id = int(float(row["motor_id"]))
                if 1 <= motor_id <= dof:
                    by_motor[motor_id] = row
            if len(by_motor) != dof:
                dropped_groups += 1
                continue

            time_values.append(float(raw_time))
            motor_map = config["robot"].get("motor_joint_map", {str(idx + 1): f"j{idx + 1}" for idx in range(dof)})
            source_order = [str(motor_map[str(idx + 1)]) for idx in range(dof)]
            q_list.append([float(by_motor[idx + 1]["q"]) for idx in range(dof)])

            if qdot_column is not None:
                qdot_list.append([float(by_motor[idx + 1][qdot_column]) for idx in range(dof)])
            if tau_column is not None:
                tau_list.append([float(by_motor[idx + 1][tau_column]) for idx in range(dof)])

        if dropped_groups > 0:
            warnings.warn(
                f"Dropped {dropped_groups} incomplete timestamps while pivoting long-format CSV: {path.name}"
            )
        if not time_values:
            raise ValueError(f"No complete {dof}-motor samples found in long-format CSV: {path}")

        time = np.asarray(time_values, dtype=np.float64)
        q = reorder_matrix_by_joint_order(np.asarray(q_list, dtype=np.float64), source_order, joint_names, "q")
        qdot = (
            reorder_matrix_by_joint_order(np.asarray(qdot_list, dtype=np.float64), source_order, joint_names, "qdot")
            if qdot_column is not None
            else None
        )

        if fixed_target is not None:
            q_des = np.repeat(fixed_target[None, :], q.shape[0], axis=0)
        else:
            raise ValueError(
                "Long-format real log does not contain q_des. Set 'real_inference.fixed_target_q_des' "
                "in the config for inference on this log format."
            )

        if tau_column is not None:
            tau_cmd = reorder_matrix_by_joint_order(np.asarray(tau_list, dtype=np.float64), source_order, joint_names, "tau_cmd")
        elif allow_zero_tau_cmd:
            warnings.warn("tau_cmd missing in long-format real log, filling tau_cmd with zeros.")
            tau_cmd = np.zeros_like(q)
        else:
            raise ValueError(
                "Long-format real log does not contain tau_cmd. Set "
                "'real_inference.allow_zero_tau_cmd: true' to force zero fill."
            )
    else:
        raise ValueError(
            "Unsupported CSV format. Expected wide columns like q1..q6 or long-format columns "
            "including motor_id, q, and timestamp/time."
        )
    if qdot_des is None:
        qdot_des = np.zeros_like(q)

    order = np.argsort(time)
    time = time[order]
    q = q[order]
    q_des = q_des[order]
    qdot_des = qdot_des[order]
    tau_cmd = tau_cmd[order]
    if tau_meas is not None:
        tau_meas = tau_meas[order]
    if tau_ff is not None:
        tau_ff = tau_ff[order]
    if kp is not None:
        kp = kp[order]
    if kd is not None:
        kd = kd[order]
    if tau_residual is not None:
        tau_residual = tau_residual[order]
    if tau_residual_corrected is not None:
        tau_residual_corrected = tau_residual_corrected[order]
    if contact_label is not None:
        contact_label = contact_label[order]
    if episode_id is not None:
        episode_id = episode_id[order]
    else:
        episode_id = np.zeros(time.shape[0], dtype=np.int64)
    if motion_type is not None:
        motion_type = motion_type[order]
    contact_marker = None
    if "contact_marker" in headers and q_wide is not None:
        contact_marker = np.asarray([float(row["contact_marker"]) for row in rows], dtype=np.float64)[order]
    if qdot is None:
        qdot = infer_qdot_from_q(q, time)
    else:
        qdot = qdot[order]
    smoothing_window = int(real_cfg.get("qdot_smoothing_window", 1))
    qdot = moving_average(qdot, smoothing_window)

    original_dt = estimate_dt(time) if time.shape[0] >= 2 else float("nan")
    original_time = time.copy()
    target_dt = float(real_cfg.get("target_dt", config.get("simulation", {}).get("dt", original_dt)))
    resample = bool(real_cfg.get("resample_to_target_dt", True))
    if np.isfinite(original_dt) and np.isfinite(target_dt) and abs(original_dt - target_dt) > max(1.0e-9, 0.05 * target_dt):
        if resample:
            warnings.warn(f"Resampling real log from dt~{original_dt:.6f}s to target_dt={target_dt:.6f}s.")
            resample_inputs = {"q": q, "qdot": qdot, "q_des": q_des, "qdot_des": qdot_des, "tau_cmd": tau_cmd}
            if tau_meas is not None:
                resample_inputs["tau_meas"] = tau_meas
            if tau_ff is not None:
                resample_inputs["tau_ff"] = tau_ff
            if kp is not None:
                resample_inputs["kp"] = kp
            if kd is not None:
                resample_inputs["kd"] = kd
            if tau_residual is not None:
                resample_inputs["tau_residual"] = tau_residual
            if tau_residual_corrected is not None:
                resample_inputs["tau_residual_corrected"] = tau_residual_corrected
            if contact_label is not None:
                resample_inputs["contact_label"] = contact_label[:, None]
            time, arrays = resample_series(
                time,
                resample_inputs,
                target_dt,
            )
            q = arrays["q"]
            qdot = arrays["qdot"]
            q_des = arrays["q_des"]
            qdot_des = arrays["qdot_des"]
            tau_cmd = arrays["tau_cmd"]
            tau_meas = arrays.get("tau_meas")
            tau_ff = arrays.get("tau_ff")
            kp = arrays.get("kp")
            kd = arrays.get("kd")
            tau_residual = arrays.get("tau_residual")
            tau_residual_corrected = arrays.get("tau_residual_corrected")
            if "contact_label" in arrays:
                contact_label = (arrays["contact_label"].reshape(-1) >= 0.5).astype(np.int64)
            episode_id = np.zeros(time.shape[0], dtype=np.int64)
            motion_type = None
        else:
            warnings.warn(
                f"Real log dt~{original_dt:.6f}s differs from target_dt={target_dt:.6f}s and resampling is disabled."
            )

    if tau_residual is None and tau_meas is not None:
        tau_residual = tau_meas - tau_cmd
    if tau_residual_corrected is None and tau_residual is not None:
        offset_duration_s = float(real_cfg.get("residual_offset_duration_s", 2.0))
        tau_residual_corrected, residual_offsets = compute_residual_offset_corrected(
            time,
            tau_residual,
            episode_id=episode_id,
            contact_label=contact_label,
            offset_duration_s=offset_duration_s,
        )
    else:
        residual_offsets = None

    result = {
        "time": time,
        "q": q,
        "qdot": qdot,
        "q_des": q_des,
        "qdot_des": qdot_des,
        "tau_cmd": tau_cmd,
        "episode_id": episode_id,
        "estimated_dt": np.asarray([original_dt], dtype=np.float64),
    }
    if tau_meas is not None:
        result["tau_meas"] = tau_meas
    if tau_ff is not None:
        result["tau_ff"] = tau_ff
    if kp is not None:
        result["kp"] = kp
    if kd is not None:
        result["kd"] = kd
    if tau_residual is not None:
        result["tau_residual"] = tau_residual
        result["tau_residual_source"] = np.asarray(["tau_meas_minus_tau_cmd"], dtype="<U32")
    if tau_residual_corrected is not None:
        result["tau_residual_corrected"] = tau_residual_corrected
        result["residual_offset_policy"] = np.asarray(["episode_initial_no_contact_mean"], dtype="<U64")
    if residual_offsets is not None:
        result["tau_residual_offsets"] = residual_offsets
    if contact_label is not None:
        result["contact_label"] = np.asarray(contact_label).astype(np.int64).reshape(-1)
    if motion_type is not None and motion_type.shape[0] == time.shape[0]:
        result["motion_type"] = motion_type
    if contact_marker is not None:
        if contact_marker.shape[0] == result["time"].shape[0]:
            result["contact_marker"] = contact_marker
        elif resample:
            resampled_time, arrays = resample_series(original_time, {"contact_marker": contact_marker}, target_dt)
            if resampled_time.shape[0] == result["time"].shape[0]:
                result["contact_marker"] = (arrays["contact_marker"] >= 0.5).astype(np.int64)
    return result


def save_training_curve(path: str | Path, history: list[dict]) -> None:
    if not history:
        raise ValueError("Training history is empty")
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    val_f1 = [item["val_f1"] for item in history]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(epochs, train_loss, label="train_loss", linewidth=2.0)
    axes[0].plot(epochs, val_loss, label="val_loss", linewidth=2.0)
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="val_f1", color="tab:green", linewidth=2.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_bar_figure(path: str | Path, metrics_payload: dict) -> None:
    labels = ["Precision", "Recall", "F1", "Accuracy"]
    threshold_metrics = metrics_payload["threshold"]
    mlp_metrics = metrics_payload.get("mlp")
    gru_metrics = metrics_payload["gru"]
    threshold_values = [
        threshold_metrics["precision"],
        threshold_metrics["recall"],
        threshold_metrics["f1"],
        threshold_metrics["accuracy"],
    ]
    series = [("Threshold", threshold_values, "#c95f5f")]
    if mlp_metrics is not None:
        series.append(
            (
                "MLP",
                [
                    mlp_metrics["precision"],
                    mlp_metrics["recall"],
                    mlp_metrics["f1"],
                    mlp_metrics["accuracy"],
                ],
                "#7a8f54",
            )
        )
    series.append(
        (
            "GRU",
            [
                gru_metrics["precision"],
                gru_metrics["recall"],
                gru_metrics["f1"],
                gru_metrics["accuracy"],
            ],
            "#4f81bd",
        )
    )

    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    offsets = np.linspace(-(len(series) - 1) / 2.0, (len(series) - 1) / 2.0, len(series))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset, (name, values, color) in zip(offsets, series):
        ax.bar(x + offset * width, values, width=width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(path: str | Path, confusion: np.ndarray, title: str) -> None:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Confusion matrix must be 2x2, got {matrix.shape}")
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def row_normalized_confusion(confusion: np.ndarray) -> np.ndarray:
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Confusion matrix must be 2x2, got {matrix.shape}")
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0.0)


def save_binary_nc_pc_confusion_matrix_figure(
    path: str | Path,
    confusion: np.ndarray,
    title: str,
    metrics: dict | None = None,
) -> None:
    """Save a poster-style nc/pc confusion matrix.

    Layout is rows=True label and columns=Predicted label:
        [[TN, FP],
         [FN, TP]]
    """

    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Confusion matrix must be 2x2, got {matrix.shape}")
    normalized = row_normalized_confusion(matrix)

    fig, ax = plt.subplots(figsize=(4.9, 4.55))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["nc\n(no contact)", "pc\n(physical contact)"])
    ax.set_yticklabels(["nc\n(no contact)", "pc\n(physical contact)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            percent = float(normalized[row, col]) * 100.0
            color = "white" if normalized[row, col] > 0.55 else "black"
            ax.text(
                col,
                row,
                f"{value:,}\n{percent:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold" if row == col else "normal",
            )

    metric_text = ""
    if metrics is not None:
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")
        if precision is not None and recall is not None and f1 is not None:
            metric_text = f"Precision {precision:.3f}   Recall {recall:.3f}   F1 {f1:.3f}"
    if metric_text:
        ax.text(
            0.5,
            -0.23,
            metric_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized ratio")
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_nc_pc_confusion_comparison_figure(
    path: str | Path,
    confusion_by_model: dict[str, np.ndarray],
    metrics_by_model: dict[str, dict] | None = None,
    title: str = "nc/pc confusion matrix comparison",
) -> None:
    """Save side-by-side poster confusion matrices for detector comparison."""

    items = [(name, np.asarray(matrix, dtype=np.int64)) for name, matrix in confusion_by_model.items()]
    if not items:
        raise ValueError("confusion_by_model must contain at least one matrix")
    for _name, matrix in items:
        if matrix.shape != (2, 2):
            raise ValueError(f"Confusion matrix must be 2x2, got {matrix.shape}")

    fig_width = max(4.0 * len(items), 5.0)
    fig, axes = plt.subplots(1, len(items), figsize=(fig_width, 4.25), squeeze=False)
    last_im = None
    for ax, (name, matrix) in zip(axes[0], items):
        normalized = row_normalized_confusion(matrix)
        last_im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["nc", "pc"])
        ax.set_yticklabels(["nc", "pc"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        metric_text = ""
        if metrics_by_model is not None and name in metrics_by_model:
            metrics = metrics_by_model[name]
            precision = metrics.get("precision")
            recall = metrics.get("recall")
            f1 = metrics.get("f1")
            if precision is not None and recall is not None and f1 is not None:
                metric_text = f"\nP {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}"
        ax.set_title(f"{name}{metric_text}", fontsize=10)

        for row in range(2):
            for col in range(2):
                value = int(matrix[row, col])
                percent = float(normalized[row, col]) * 100.0
                color = "white" if normalized[row, col] > 0.55 else "black"
                ax.text(
                    col,
                    row,
                    f"{value:,}\n{percent:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    fontweight="bold" if row == col else "normal",
                )

    fig.suptitle(f"{title}\nnc: no contact, pc: physical contact", y=1.03, fontsize=12)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025)
        cbar.set_label("Row-normalized ratio")
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_region_confusion_matrix_figure(
    path: str | Path,
    confusion: np.ndarray,
    region_names: list[str],
    title: str = "Contact region confusion matrix",
) -> None:
    matrix = np.asarray(confusion, dtype=np.int64)
    names = list(region_names)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Region confusion matrix must be square, got {matrix.shape}")
    if len(names) != matrix.shape[0]:
        names = [str(idx) for idx in range(matrix.shape[0])]
    fig_size = max(5.0, 1.0 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="Blues")
    ticks = np.arange(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted region")
    ax.set_ylabel("True region")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sim_prediction_example(
    path: str | Path,
    time: np.ndarray,
    label: np.ndarray,
    threshold_prediction: np.ndarray,
    gru_probability: np.ndarray,
    mlp_probability: np.ndarray | None = None,
    mlp_decision_threshold: float | None = None,
    gru_decision_threshold: float | None = None,
    e_norm: np.ndarray | None = None,
) -> None:
    time_arr = np.asarray(time, dtype=np.float64)
    label_arr = np.asarray(label, dtype=np.float64)
    threshold_arr = np.asarray(threshold_prediction, dtype=np.float64)
    mlp_arr = None if mlp_probability is None else np.asarray(mlp_probability, dtype=np.float64)
    prob_arr = np.asarray(gru_probability, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(time_arr, label_arr, where="post", label="Ground truth contact", linewidth=2.0, color="black")
    if e_norm is not None:
        e_arr = np.asarray(e_norm, dtype=np.float64)
        scale = max(float(np.nanmax(e_arr)), 1.0e-9)
        ax.plot(time_arr, e_arr / scale, label=r"$||e_q||$ normalized", linewidth=1.2, color="tab:orange", alpha=0.75)
    ax.step(
        time_arr,
        threshold_arr,
        where="post",
        label="Threshold prediction",
        linewidth=1.8,
        color="tab:red",
        alpha=0.85,
    )
    if mlp_arr is not None:
        ax.plot(time_arr, mlp_arr, label="MLP P(contact)", linewidth=1.7, color="tab:green")
        if mlp_decision_threshold is not None:
            ax.axhline(
                float(mlp_decision_threshold),
                linestyle="--",
                linewidth=1.0,
                color="tab:green",
                alpha=0.65,
                label=f"MLP threshold={float(mlp_decision_threshold):.2f}",
            )
    ax.plot(time_arr, prob_arr, label="GRU P(contact)", linewidth=2.0, color="tab:blue")
    if gru_decision_threshold is not None:
        ax.axhline(
            float(gru_decision_threshold),
            linestyle="--",
            linewidth=1.2,
            color="tab:blue",
            alpha=0.65,
            label=f"GRU threshold={float(gru_decision_threshold):.2f}",
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Contact / Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sim_prediction_examples(
    path: str | Path,
    examples: list[dict[str, np.ndarray | str | float]],
    mlp_decision_threshold: float | None = None,
    gru_decision_threshold: float | None = None,
) -> None:
    if not examples:
        raise ValueError("examples must contain at least one episode payload")

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3.0 * len(examples)), sharex=False)
    if len(examples) == 1:
        axes = [axes]

    for ax, example in zip(axes, examples):
        time_arr = np.asarray(example["time"], dtype=np.float64)
        label_arr = np.asarray(example["label"], dtype=np.float64)
        threshold_arr = np.asarray(example["threshold_prediction"], dtype=np.float64)
        mlp_arr = (
            np.asarray(example["mlp_probability"], dtype=np.float64)
            if "mlp_probability" in example
            else None
        )
        prob_arr = np.asarray(example["gru_probability"], dtype=np.float64)
        title = str(example.get("title", "Episode"))

        ax.step(time_arr, label_arr, where="post", label="Ground truth contact", linewidth=1.8, color="black")
        ax.step(
            time_arr,
            threshold_arr,
            where="post",
            label="Threshold prediction",
            linewidth=1.5,
            color="tab:red",
            alpha=0.8,
        )
        if mlp_arr is not None:
            ax.plot(time_arr, mlp_arr, label="MLP P(contact)", linewidth=1.5, color="tab:green")
            if mlp_decision_threshold is not None:
                ax.axhline(float(mlp_decision_threshold), linestyle="--", linewidth=1.0, color="tab:green", alpha=0.55)
        ax.plot(time_arr, prob_arr, label="GRU P(contact)", linewidth=1.8, color="tab:blue")
        if gru_decision_threshold is not None:
            ax.axhline(float(gru_decision_threshold), linestyle="--", linewidth=1.0, color="tab:blue", alpha=0.65)
        ax.set_title(title)
        ax.set_ylabel("Contact / P")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_threshold_tradeoff_figure(
    path: str | Path,
    sweep_rows: list[dict[str, float]],
    comparison_sweeps: dict[str, list[dict[str, float]]] | None = None,
) -> None:
    if not sweep_rows:
        raise ValueError("sweep_rows is empty")
    curves = {"GRU": sweep_rows}
    if comparison_sweeps:
        for name, rows in comparison_sweeps.items():
            if rows:
                curves[str(name)] = rows

    fig, ax = plt.subplots(figsize=(8, 4.5))
    color_map = {"GRU": "tab:blue", "MLP": "tab:green"}
    for name, rows in curves.items():
        thresholds = np.asarray([row["threshold"] for row in rows], dtype=np.float64)
        precision = np.asarray([row["precision"] for row in rows], dtype=np.float64)
        recall = np.asarray([row["recall"] for row in rows], dtype=np.float64)
        f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)
        color = color_map.get(str(name), None)
        ax.plot(thresholds, precision, label=f"{name} Precision", linewidth=1.8, color=color, alpha=0.55)
        ax.plot(thresholds, recall, label=f"{name} Recall", linewidth=1.8, linestyle="--", color=color, alpha=0.8)
        ax.plot(thresholds, f1, label=f"{name} F1", linewidth=2.0, linestyle="-.", color=color, alpha=0.95)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve_figure(
    path: str | Path,
    sweeps: dict[str, list[dict[str, float]]],
) -> None:
    if not sweeps:
        raise ValueError("sweeps is empty")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    color_map = {"GRU": "tab:blue", "MLP": "tab:green", "Threshold": "tab:red"}
    for name, rows in sweeps.items():
        if not rows:
            continue
        precision = np.asarray([row["precision"] for row in rows], dtype=np.float64)
        recall = np.asarray([row["recall"] for row in rows], dtype=np.float64)
        order = np.argsort(recall)
        ax.plot(
            recall[order],
            precision[order],
            linewidth=2.0,
            label=str(name),
            color=color_map.get(str(name), None),
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_real_probability_figure(
    path: str | Path,
    time: np.ndarray,
    probability: np.ndarray,
    intervals: list[list[float]] | None = None,
) -> None:
    time_arr = np.asarray(time, dtype=np.float64)
    prob_arr = np.asarray(probability, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4))
    if intervals:
        for start, end in intervals:
            ax.axvspan(float(start), float(end), color="0.85", alpha=0.8)
    ax.plot(time_arr, prob_arr, linewidth=2.0, color="tab:blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("P(contact)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Preliminary feasibility check, not quantitative accuracy evaluation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_residual_timeseries_figure(
    path: str | Path,
    time: np.ndarray,
    tau_residual: np.ndarray,
    tau_residual_corrected: np.ndarray | None = None,
    probability: np.ndarray | None = None,
    threshold: float | None = None,
) -> None:
    """Save real-log residual norm diagnostics."""

    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    raw = np.asarray(tau_residual, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != time_arr.shape[0]:
        raise ValueError(f"tau_residual must be [N, dof], got {raw.shape}")
    raw_norm = np.linalg.norm(raw, axis=1)
    corrected_norm = None
    if tau_residual_corrected is not None:
        corrected = np.asarray(tau_residual_corrected, dtype=np.float64)
        if corrected.shape == raw.shape:
            corrected_norm = np.linalg.norm(corrected, axis=1)

    axes_count = 2 if probability is not None else 1
    fig, axes = plt.subplots(axes_count, 1, figsize=(10.5, 3.8 * axes_count), sharex=True)
    if axes_count == 1:
        axes = [axes]
    axes[0].plot(time_arr, raw_norm, linewidth=1.4, label=r"$||\tau_{residual}||$")
    if corrected_norm is not None:
        axes[0].plot(time_arr, corrected_norm, linewidth=1.4, label=r"$||\tau_{residual,corr}||$")
    axes[0].set_ylabel("Residual torque norm [Nm]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    if probability is not None:
        prob = np.asarray(probability, dtype=np.float64).reshape(-1)
        axes[1].plot(time_arr, prob, linewidth=1.7, color="tab:blue", label="P(contact)")
        if threshold is not None:
            axes[1].axhline(float(threshold), color="black", linestyle="--", linewidth=1.0, label="threshold")
        axes[1].set_ylabel("P(contact)")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_overview_pipeline_figure(path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    x_positions = [0.02, 0.29, 0.56, 0.81]
    width = 0.16
    height = 0.36
    labels = [
        "1. Sim dataset generation\nURDF nominal dynamics\n+ disturbance torque labels",
        "2. Detector training\nThreshold baseline\nand GRU detector",
        "3. Simulation evaluation\nPrecision / Recall / F1\nand qualitative traces",
        "4. Real log inference\nP(contact) trend only\n(preliminary check)",
    ]
    colors = ["#dde9f7", "#e7f4dd", "#fce6d5", "#efe1f5"]

    for xpos, label_text, color in zip(x_positions, labels, colors):
        box = FancyBboxPatch(
            (xpos, 0.32),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(xpos + width / 2.0, 0.5, label_text, ha="center", va="center", fontsize=11)

    for idx in range(len(x_positions) - 1):
        arrow = FancyArrowPatch(
            (x_positions[idx] + width, 0.5),
            (x_positions[idx + 1], 0.5),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8,
            color="#4d4d4d",
        )
        ax.add_patch(arrow)

    fig.tight_layout()
    fig.savefig(Path(path).expanduser().resolve(), dpi=300, bbox_inches="tight")
    plt.close(fig)
