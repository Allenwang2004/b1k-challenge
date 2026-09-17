"""BEHAVIOR-1K 2026 challenge demos as a training dataset (LeRobot v3.0 format).

Replaces ``omnigibson.learning.datas.lerobot_dataset.BehaviorLeRobotDataset`` (2025, LeRobot v2.1,
removed from BEHAVIOR-1K v3.9.0). Only the behaviour the training pipeline relied on is kept:

- task filtering by name,
- modality selection (``rgb`` only by default -> depth videos are neither required on disk nor decoded),
- local-only loading of a *partial* download (only the episodes whose data + video files are present),
- ``delta_timestamps`` action chunks with ``action_is_pad``,
- items keyed like the 2025 dataset (``observation.images.rgb.{head,left_wrist,right_wrist}``) so
  ``LeRobotB1KDataConfig``'s repack mapping is unchanged,
- ``dataset.meta.episodes[episode_index]["length"]`` for ``ComputeSubtaskStateFromMeta``.

Expected layout under ``root`` (download only what you train on, the full set is 3.27 TB)::

    meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/**   (all of meta/episodes: the
                                                                             episode table is positional)
    data/chunk-<task>/file-*.parquet
    videos/observation.rgb.*/chunk-<task>/file-*.mp4
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)

# 2026 video key -> key expected by the 2025-era transforms in b1k.training.config.
CAMERA_KEY_MAP = {
    "observation.rgb.zed_link_camera_0": "observation.images.rgb.head",
    "observation.rgb.left_realsense_link_camera_0": "observation.images.rgb.left_wrist",
    "observation.rgb.right_realsense_link_camera_0": "observation.images.rgb.right_wrist",
}


def _modality_of(video_key: str) -> str:
    # "observation.rgb.zed_link_camera_0" -> "rgb"; "observation.depth_linear.<cam>" -> "depth_linear"
    return video_key.split(".")[1]


class BehaviorEpisodeMeta:
    """Minimal metadata view used by the transforms (``meta.episodes[ep]["length"]``, ``meta.tasks``)."""

    def __init__(self, episodes: dict[int, dict], tasks: dict[int, str], fps: int):
        self.episodes = episodes
        self.tasks = tasks
        self.fps = fps


class BehaviorLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        tasks: Iterable[str] | None = None,
        modalities: Sequence[str] = ("rgb",),
        episodes: Sequence[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 5e-4,
        video_backend: str | None = None,
        return_uint8: bool = True,
        local_only: bool = True,
    ):
        if not local_only:
            raise NotImplementedError("Only local datasets are supported; download the task folders first.")
        self.repo_id = repo_id
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"No LeRobot dataset at {self.root} (missing meta/info.json)")

        meta = LeRobotDatasetMetadata(repo_id, root=self.root)

        # --- modality selection: drop video features we don't train on (depth) ---------------------
        modalities = set(modalities)
        kept = {k: ft for k, ft in meta.features.items() if ft["dtype"] != "video" or _modality_of(k) in modalities}
        dropped = sorted(set(meta.features) - set(kept))
        if dropped:
            logger.info(f"Ignoring video features {dropped} (modalities={sorted(modalities)})")
        meta.info.features = kept
        unknown = [k for k in meta.video_keys if k not in CAMERA_KEY_MAP]
        if unknown:
            raise ValueError(f"No camera-name mapping for video keys {unknown}; extend CAMERA_KEY_MAP")

        # --- task selection -------------------------------------------------------------------------
        task_table = meta.tasks  # DataFrame: index = task name, column task_index
        name_to_index = {str(name): int(row["task_index"]) for name, row in task_table.iterrows()}
        if tasks is None:
            selected_task_indices = set(name_to_index.values())
        else:
            missing = sorted(set(tasks) - set(name_to_index))
            if missing:
                logger.warning(f"{len(missing)} requested tasks are not in {self.root}/meta/tasks.parquet: {missing[:5]}...")
            selected_task_indices = {name_to_index[t] for t in tasks if t in name_to_index}

        # --- episode selection: requested tasks, optional explicit list, and present on disk ---------
        ep_table = meta.episodes  # HF Dataset, positional == episode_index
        ep_index = np.asarray(ep_table["episode_index"])
        ep_task = np.asarray(ep_table["task_index"])
        ep_length = np.asarray(ep_table["length"])
        if not np.array_equal(ep_index, np.arange(len(ep_index))):
            raise ValueError("meta/episodes must contain every episode in order; download all of meta/episodes/**")

        candidates = [int(e) for e in ep_index[np.isin(ep_task, list(selected_task_indices))]]
        if episodes is not None:
            wanted = set(int(e) for e in episodes)
            candidates = [e for e in candidates if e in wanted]

        available = [e for e in candidates if self._episode_files_present(meta, e)]
        if len(available) < len(candidates):
            logger.warning(
                f"{len(candidates) - len(available)} of {len(candidates)} selected episodes are not fully "
                f"downloaded under {self.root} and will be skipped"
            )
        if not available:
            raise FileNotFoundError(
                f"No episodes for tasks {sorted(selected_task_indices)} found under {self.root}/data and /videos"
            )
        self.episodes = available

        # --- reader ---------------------------------------------------------------------------------
        self.reader = DatasetReader(
            meta=meta,
            root=self.root,
            episodes=self.episodes,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
            delta_timestamps=delta_timestamps,
            image_transforms=None,
            return_uint8=return_uint8,
        )
        self.reader.load_and_activate()
        self.lerobot_meta = meta
        self.meta = BehaviorEpisodeMeta(
            episodes={int(e): {"length": int(ep_length[e]), "task_index": int(ep_task[e])} for e in self.episodes},
            tasks={idx: name for name, idx in name_to_index.items()},
            fps=meta.fps,
        )
        self.video_keys = list(meta.video_keys)
        logger.info(
            f"BehaviorLeRobotDataset: {len(self.episodes)} episodes / {len(self)} frames, "
            f"tasks={sorted({self.meta.episodes[e]['task_index'] for e in self.episodes})}, cameras={self.video_keys}"
        )

    @staticmethod
    def _episode_files_present(meta: LeRobotDatasetMetadata, ep_idx: int) -> bool:
        if not (meta.root / meta.get_data_file_path(ep_idx)).exists():
            return False
        return all((meta.root / meta.get_video_file_path(ep_idx, k)).exists() for k in meta.video_keys)

    def __len__(self) -> int:
        return self.reader.num_frames

    def __getitem__(self, idx: int) -> dict:
        item = self.reader.get_item(idx)
        for src, dst in CAMERA_KEY_MAP.items():
            if src in item:
                item[dst] = item.pop(src)
        return item


# ---------------------------------------------------------------------------------------------------
# Episode enumeration for the offline scripts (compute_norm_stats.py, train_fast_tokenizer.py).
# They only need observation.state / action per episode, so they read parquet directly.
# ---------------------------------------------------------------------------------------------------

_EPISODE_COLUMNS = ["episode_index", "task_index", "observation.state", "action"]


def list_episode_frames(data_root: str | Path) -> list[tuple[Path, int | None]]:
    """Return ``(parquet_path, episode_index)`` for every locally available episode.

    - 2025 layout (one file per episode, ``data/task-*/episode_*.parquet``): episode_index is ``None``.
    - 2026 / LeRobot v3 layout (``data/chunk-*/file-*.parquet``, many episodes per file): one entry per
      episode found in the file.
    """
    data_root = Path(data_root).expanduser()
    per_episode = sorted(data_root.glob("data/task-*/episode_*.parquet"))
    if per_episode:
        return [(p, None) for p in per_episode]

    import pyarrow.parquet as pq

    out: list[tuple[Path, int | None]] = []
    for path in sorted(data_root.glob("data/chunk-*/file-*.parquet")):
        ep = pq.read_table(path, columns=["episode_index"]).column("episode_index").unique().to_pylist()
        out.extend((path, int(e)) for e in sorted(ep))
    return out


def load_episode_frames(path: str | Path, episode_index: int | None):
    """Read one episode's ``observation.state`` / ``action`` rows (pandas DataFrame, ordered by frame)."""
    import pandas as pd

    if episode_index is None:
        return pd.read_parquet(path)
    df = pd.read_parquet(path, columns=_EPISODE_COLUMNS, filters=[("episode_index", "==", int(episode_index))])
    return df.reset_index(drop=True)
