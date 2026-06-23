"""Sliding-window dataset utilities for GRU contact detection.

논문 흐름에서 이 파일은 "관절 상태 시계열을 GRU 입력 window로 변환"하는
부분이다. 한 sample의 기본 feature는 다음과 같다.

    x_t = [q_t, qdot_t, e_q_t, tau_cmd_t]
    e_q_t = q_des_t - q_t

config에서 use_delta_features=true이면 episode 내부에서만 delta feature를
추가한다. episode 첫 sample의 delta는 zero이고, window는 episode 경계를
넘지 않는다. 이 조건은 데이터 누수(leakage)를 막기 위해 중요하다.

주의: tau_ext는 여기에서 feature로 전달되지 않는다. tau_ext는 시뮬레이션
라벨 검증용으로만 dataset npz에 남아 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import StandardScaler, build_input_features, iter_episode_slices, load_npz_dataset


def build_window_end_indices(episode_id: np.ndarray, window_length: int, stride: int) -> np.ndarray:
    """Return valid window end indices without crossing episode boundaries."""
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    end_indices: list[int] = []
    for start, end in iter_episode_slices(episode_id):
        # episode slice 내부에서만 [end-window_length+1, end] window를 만든다.
        # 이 한 줄이 train/val/test 안의 episode boundary leakage를 막는 핵심이다.
        if (end - start) < window_length:
            continue
        first_end = start + window_length - 1
        end_indices.extend(range(first_end, end, stride))
    return np.asarray(end_indices, dtype=np.int64)


def shift_labels_episodewise(labels: np.ndarray, episode_id: np.ndarray, delay_steps: int) -> np.ndarray:
    """Shift contact labels later within each episode.

    A positive delay means the training label reacts after the command label.
    Values shifted beyond an episode boundary are dropped and leading samples
    are filled with no-contact.
    """

    label_arr = np.asarray(labels).astype(np.int64).reshape(-1)
    episode_arr = np.asarray(episode_id).reshape(-1)
    if label_arr.shape[0] != episode_arr.shape[0]:
        raise ValueError(f"labels and episode_id length mismatch: {label_arr.shape[0]} vs {episode_arr.shape[0]}")
    delay = int(delay_steps)
    if delay <= 0:
        return label_arr.copy()
    shifted = np.zeros_like(label_arr, dtype=np.int64)
    for start, end in iter_episode_slices(episode_arr):
        if (end - start) <= delay:
            continue
        shifted[start + delay : end] = label_arr[start : end - delay]
    return shifted


def transition_exclusion_sample_mask(
    labels: np.ndarray,
    episode_id: np.ndarray,
    exclusion_steps: int,
) -> np.ndarray:
    """Return a boolean mask that is False around label transitions."""

    label_arr = np.asarray(labels).astype(np.int64).reshape(-1)
    episode_arr = np.asarray(episode_id).reshape(-1)
    if label_arr.shape[0] != episode_arr.shape[0]:
        raise ValueError(f"labels and episode_id length mismatch: {label_arr.shape[0]} vs {episode_arr.shape[0]}")
    radius = int(exclusion_steps)
    mask = np.ones(label_arr.shape[0], dtype=bool)
    if radius <= 0:
        return mask
    for start, end in iter_episode_slices(episode_arr):
        local = label_arr[start:end]
        if local.size <= 1:
            continue
        transition_offsets = np.flatnonzero(local[1:] != local[:-1]) + 1
        for offset in transition_offsets:
            lo = max(start, start + int(offset) - radius)
            hi = min(end, start + int(offset) + radius + 1)
            mask[lo:hi] = False
    return mask


@dataclass
class DatasetBundle:
    dataset: "ContactWindowDataset"
    scaler: StandardScaler
    feature_names: list[str]


class ContactSingleStepDataset:
    """Aligned single-timestep dataset for MLP-style baselines.

    이 dataset은 GRU window dataset과 같은 end_indices를 그대로 사용한다.
    따라서 MLP도 "현재 시점 x_t"만 보지만, 평가/학습 sample 집합 자체는
    GRU와 정확히 맞춰 공정 비교가 가능하다.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        selected_indices: np.ndarray,
        episode_id: np.ndarray | None = None,
        original_labels: np.ndarray | None = None,
    ) -> None:
        feature_arr = np.asarray(features, dtype=np.float32)
        label_arr = np.asarray(labels, dtype=np.float32).reshape(-1)
        index_arr = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        original_label_arr = label_arr if original_labels is None else np.asarray(original_labels, dtype=np.float32).reshape(-1)
        if feature_arr.ndim != 2:
            raise ValueError(f"features must be [N, D], got {feature_arr.shape}")
        if feature_arr.shape[0] != label_arr.shape[0]:
            raise ValueError(
                f"features and labels must share the same first dimension: {feature_arr.shape[0]} vs {label_arr.shape[0]}"
            )
        if original_label_arr.shape[0] != feature_arr.shape[0]:
            raise ValueError(
                "features and original_labels must share the same first dimension: "
                f"{feature_arr.shape[0]} vs {original_label_arr.shape[0]}"
            )
        if index_arr.size == 0:
            raise ValueError("selected_indices is empty")
        if int(np.min(index_arr)) < 0 or int(np.max(index_arr)) >= feature_arr.shape[0]:
            raise ValueError("selected_indices contain values outside the feature range")

        self.features = feature_arr
        self.labels = label_arr
        self.original_labels = original_label_arr
        self.selected_indices = index_arr
        self.input_dim = int(feature_arr.shape[1])
        self.episode_id = None if episode_id is None else np.asarray(episode_id).reshape(-1)
        if self.episode_id is not None and self.episode_id.shape[0] != feature_arr.shape[0]:
            raise ValueError(
                f"episode_id must share the feature length: {self.episode_id.shape[0]} vs {feature_arr.shape[0]}"
            )

    @classmethod
    def from_window_dataset(cls, dataset: "ContactWindowDataset") -> "ContactSingleStepDataset":
        return cls(
            features=dataset.features,
            labels=dataset.labels,
            selected_indices=dataset.end_indices,
            episode_id=dataset.episode_id,
            original_labels=dataset.original_labels,
        )

    def __len__(self) -> int:
        return int(self.selected_indices.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        selected_idx = int(self.selected_indices[index])
        return self.features[selected_idx], np.float32(self.labels[selected_idx])

    def labels_for_steps(self) -> np.ndarray:
        return self.labels[self.selected_indices].astype(np.int64)

    def original_labels_for_steps(self) -> np.ndarray:
        return self.original_labels[self.selected_indices].astype(np.int64)

    def episodes_for_steps(self) -> np.ndarray:
        if self.episode_id is None:
            raise ValueError("episode_id is not available for this dataset")
        return self.episode_id[self.selected_indices]

    def feature_rows(self) -> np.ndarray:
        return self.features[self.selected_indices].astype(np.float32)


class ContactWindowDataset:
    """NumPy-backed sliding-window dataset for GRU training and inference."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        episode_id: np.ndarray,
        window_length: int,
        stride: int = 1,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
        contact_region: np.ndarray | None = None,
        return_region: bool = False,
        label_delay_steps: int = 0,
        transition_exclusion_steps: int = 0,
        exclude_transition_windows: bool = False,
    ) -> None:
        # features는 이미 build_input_features()로 만든 [N, D] matrix다.
        # label과 episode_id 길이가 다르면 window label이 틀어지므로 즉시 에러를 낸다.
        feature_arr = np.asarray(features, dtype=np.float64)
        label_arr = np.asarray(labels, dtype=np.float32).reshape(-1)
        episode_arr = np.asarray(episode_id).reshape(-1)

        if feature_arr.ndim != 2:
            raise ValueError(f"features must be [N, D], got {feature_arr.shape}")
        if feature_arr.shape[0] != label_arr.shape[0] or feature_arr.shape[0] != episode_arr.shape[0]:
            raise ValueError(
                "features, labels, and episode_id must share the same first dimension: "
                f"{feature_arr.shape[0]}, {label_arr.shape[0]}, {episode_arr.shape[0]}"
            )
        if not np.isfinite(feature_arr).all():
            raise ValueError("features contain non-finite values. Regenerate the dataset or inspect preprocessing.")
        if not np.isfinite(label_arr).all():
            raise ValueError("labels contain non-finite values. Regenerate the dataset.")

        self.window_length = int(window_length)
        self.stride = int(stride)
        self.raw_features = feature_arr
        self.original_labels = label_arr.astype(np.float32)
        self.labels = shift_labels_episodewise(label_arr, episode_arr, int(label_delay_steps)).astype(np.float32)
        self.episode_id = episode_arr
        self.return_region = bool(return_region)
        if contact_region is None:
            self.contact_region = np.zeros_like(label_arr, dtype=np.int64)
        else:
            region_arr = np.asarray(contact_region).astype(np.int64).reshape(-1)
            if region_arr.shape[0] != feature_arr.shape[0]:
                raise ValueError(
                    f"contact_region must share the feature length: {region_arr.shape[0]} vs {feature_arr.shape[0]}"
                )
            self.contact_region = region_arr
        self.label_delay_steps = int(label_delay_steps)
        self.transition_exclusion_steps = int(transition_exclusion_steps)
        self.exclude_transition_windows = bool(exclude_transition_windows)
        self.all_end_indices = build_window_end_indices(self.episode_id, self.window_length, self.stride)
        self.transition_sample_mask = transition_exclusion_sample_mask(
            self.original_labels,
            self.episode_id,
            self.transition_exclusion_steps,
        )
        if self.exclude_transition_windows and self.transition_exclusion_steps > 0:
            self.end_indices = self.all_end_indices[self.transition_sample_mask[self.all_end_indices]]
        else:
            self.end_indices = self.all_end_indices
        self.excluded_window_count = int(self.all_end_indices.size - self.end_indices.size)
        if self.end_indices.size == 0:
            raise ValueError(
                "No valid windows were produced. Check episode lengths, window_length, and stride."
            )

        # scaler는 train set에서만 fit한다. val/test/real inference는 저장된 scaler를 재사용한다.
        if fit_scaler:
            self.scaler = StandardScaler.fit(self.raw_features)
        elif scaler is not None:
            self.scaler = scaler
        else:
            raise ValueError("Provide a scaler or set fit_scaler=True")

        self.features = self.scaler.transform(self.raw_features).astype(np.float32)
        self.input_dim = int(self.features.shape[1])

    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        window_length: int,
        stride: int,
        use_delta_features: bool,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
        return_region: bool = False,
        label_delay_steps: int = 0,
        transition_exclusion_steps: int = 0,
        exclude_transition_windows: bool = False,
        feature_mode: str = "original_42",
    ) -> DatasetBundle:
        # npz에는 tau_ext도 들어 있지만 build_input_features()에 전달하지 않는다.
        # 따라서 tau_ext가 feature/scaler에 들어가는 leakage 경로가 없다.
        data = load_npz_dataset(npz_path)
        feature_matrix, feature_names = build_input_features(
            q=data["q"],
            qdot=data["qdot"],
            q_des=data["q_des"],
            tau_cmd=data["tau_cmd"],
            use_delta_features=use_delta_features,
            episode_id=data["episode_id"],
            tau_residual=data.get("tau_residual"),
            tau_residual_corrected=data.get("tau_residual_corrected"),
            feature_mode=feature_mode,
        )
        dataset = cls(
            features=feature_matrix,
            labels=data["label"],
            episode_id=data["episode_id"],
            window_length=window_length,
            stride=stride,
            scaler=scaler,
            fit_scaler=fit_scaler,
            contact_region=data.get("contact_region"),
            return_region=return_region,
            label_delay_steps=int(label_delay_steps),
            transition_exclusion_steps=int(transition_exclusion_steps),
            exclude_transition_windows=bool(exclude_transition_windows),
        )
        return DatasetBundle(dataset=dataset, scaler=dataset.scaler, feature_names=feature_names)

    def __len__(self) -> int:
        return int(self.end_indices.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        end_idx = int(self.end_indices[index])
        start_idx = end_idx - self.window_length + 1
        window = self.features[start_idx : end_idx + 1]
        label = np.float32(self.labels[end_idx])
        if self.return_region:
            return window, label, np.int64(self.contact_region[end_idx])
        return window, label

    def labels_for_windows(self) -> np.ndarray:
        return self.labels[self.end_indices].astype(np.int64)

    def original_labels_for_windows(self) -> np.ndarray:
        return self.original_labels[self.end_indices].astype(np.int64)

    def contact_regions_for_windows(self) -> np.ndarray:
        return self.contact_region[self.end_indices].astype(np.int64)

    def time_for_windows(self, time: np.ndarray) -> np.ndarray:
        time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
        return time_arr[self.end_indices]

    def episodes_for_windows(self) -> np.ndarray:
        return self.episode_id[self.end_indices]

    def raw_feature_rows_for_windows(self) -> np.ndarray:
        return self.raw_features[self.end_indices]

    def feature_windows(self) -> np.ndarray:
        windows = np.stack([self[idx][0] for idx in range(len(self))], axis=0)
        return windows.astype(np.float32)

    def aligned_single_step_dataset(self) -> ContactSingleStepDataset:
        return ContactSingleStepDataset.from_window_dataset(self)
