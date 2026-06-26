import argparse
import collections
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

from videosaur import configuration, data, models
from videosaur.train import _setup_trainer_config


CACHE_FORMAT = "sharded_npy"
FEATURE_SHARD_DIR = "features"
INDEX_FRAGMENT_DIR = "_index_fragments"
FEATURE_SHARD_PREFIX = "rank="
INDEX_FRAGMENT_PREFIX = "rank="
INDEX_FILENAME = "index.parquet"
METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 2
PROVENANCE_KEYS = (
    "cache_row",
    "dataset_index",
    "split_index",
    "episode_index",
    "frame_index",
    "absolute_index",
    "timestamp",
    "task_index",
    "context_absolute_indices",
    "context_frame_indices",
    "context_is_pad",
)
INDEX_FIELD_NAMES = (
    "row_id",
    "split",
    "dataset_index",
    "repo_id",
    "root",
    "camera_key",
    "episode_index",
    "frame_index",
    "absolute_index",
    "timestamp",
    "task_index",
    "context_absolute_indices",
    "context_frame_indices",
    "context_is_pad",
    "shard_path",
    "shard_offset",
    "shard_num_rows",
)


def _parse_splits(value: str) -> List[str]:
    splits = [split.strip() for split in value.split(",") if split.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("Expected at least one split.")
    invalid = [split for split in splits if split not in ("train", "val", "test")]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown splits: {invalid}")
    return splits


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported cache dtype `{name}`.")


def _numpy_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype("float16")
    if name == "float32":
        return np.dtype("float32")
    raise ValueError(f"Unsupported cache dtype `{name}`.")


def _rank_from_env() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _world_size_from_env() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _distributed_rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return _rank_from_env(), _world_size_from_env()


def _distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _clear_output_dir(output_dir: pathlib.Path):
    for path in output_dir.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _overwrite_output_dir(output_dir: pathlib.Path):
    if output_dir.is_symlink():
        target = output_dir.resolve(strict=False)
        if target.exists():
            if target.is_dir():
                _clear_output_dir(target)
            else:
                target.unlink()
                target.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
        return

    if output_dir.exists():
        if output_dir.is_dir():
            _clear_output_dir(output_dir)
        else:
            output_dir.unlink()
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def _prepare_output_dir(output_dir: pathlib.Path, overwrite: bool, rank: int):
    if rank == 0:
        if output_dir.exists() or output_dir.is_symlink():
            if not overwrite and any(output_dir.iterdir()):
                raise FileExistsError(
                    f"Output directory `{output_dir}` is not empty. Pass --overwrite to replace it."
                )
            if overwrite:
                _overwrite_output_dir(output_dir)
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        deadline = time.time() + 120
        while not output_dir.exists():
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for output directory `{output_dir}`.")
            time.sleep(0.5)


def _sanitize_dataset_info(dataset_info: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized = []
    for info in dataset_info:
        item = dict(info)
        if item.get("root") is not None:
            item["root"] = os.fspath(item["root"])
        sanitized.append(item)
    return sanitized


def _git_commit(cwd: pathlib.Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


class SlotFeaturePredictor(pl.LightningModule):
    """Predict wrapper extracting the current-frame slot state from VideoSAUR outputs."""

    def __init__(self, model: pl.LightningModule, output_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.model = model
        self.output_dtype = output_dtype

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model(batch)
        slots = outputs["processor"]["state"]
        if slots.ndim != 4:
            raise ValueError(
                "Expected video slot state with shape [B, T, slots, dim], "
                f"but got {tuple(slots.shape)}."
            )

        features = slots[:, -1].detach().to(device="cpu", dtype=self.output_dtype)
        provenance = {
            key: batch[key].detach().cpu()
            for key in PROVENANCE_KEYS
            if key in batch and torch.is_tensor(batch[key])
        }
        missing = [key for key in PROVENANCE_KEYS if key not in provenance]
        if missing:
            raise KeyError(f"Missing cache provenance keys in batch: {missing}")

        return {"features": features, "provenance": provenance}


class SlotFeaturePredictionWriter(BasePredictionWriter):
    """Write final feature shards and index fragments during prediction."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        *,
        split_names: List[str],
        dataset_info: List[Dict[str, Any]],
        dtype: str,
        target_shard_mb: float,
    ):
        super().__init__(write_interval="batch_and_epoch")
        if target_shard_mb <= 0:
            raise ValueError("`target_shard_mb` must be positive.")
        self.output_dir = pathlib.Path(output_dir)
        self.split_names = split_names
        self.dataset_info = dataset_info
        self.dtype = dtype
        self.np_dtype = _numpy_dtype(dtype)
        self.target_shard_bytes = int(target_shard_mb * 1024 * 1024)
        self.target_shard_mb = target_shard_mb
        self._feature_shape: Optional[Tuple[int, ...]] = None
        self._rows_per_shard: Optional[int] = None
        self._part_counter = 0
        self._buffer_rows = 0
        self._feature_buffer: List[np.ndarray] = []
        self._provenance_buffer: Dict[str, List[np.ndarray]] = {
            key: [] for key in PROVENANCE_KEYS
        }

    def setup(self, trainer, pl_module, stage=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / FEATURE_SHARD_DIR).mkdir(parents=True, exist_ok=True)
        (self.output_dir / INDEX_FRAGMENT_DIR).mkdir(parents=True, exist_ok=True)

    def _append_prediction(self, prediction: Dict[str, Any]):
        features = _array_to_numpy(prediction["features"]).astype(self.np_dtype, copy=True)
        if features.ndim != 3:
            raise ValueError(
                "Expected cached features with shape [B, slots, dim], "
                f"but got {tuple(features.shape)}."
            )
        if self._feature_shape is None:
            self._feature_shape = tuple(features.shape[1:])
            row_bytes = int(np.prod(self._feature_shape)) * self.np_dtype.itemsize
            self._rows_per_shard = max(1, self.target_shard_bytes // max(row_bytes, 1))
        elif tuple(features.shape[1:]) != self._feature_shape:
            raise ValueError(
                f"Feature shape mismatch: expected {self._feature_shape}, "
                f"got {tuple(features.shape[1:])}."
            )

        missing = [key for key in PROVENANCE_KEYS if key not in prediction["provenance"]]
        if missing:
            raise KeyError(f"Missing cache provenance keys in prediction: {missing}")
        provenance = {
            key: _array_to_numpy(prediction["provenance"][key])
            for key in PROVENANCE_KEYS
        }
        rows = provenance["cache_row"]
        if features.shape[0] != rows.shape[0]:
            raise ValueError("Feature/provenance batch size mismatch.")

        self._feature_buffer.append(features)
        for key, value in provenance.items():
            self._provenance_buffer[key].append(np.asarray(value).copy())
        self._buffer_rows += features.shape[0]

    def _flush(self, rank: int):
        if self._buffer_rows == 0:
            return

        features = np.concatenate(self._feature_buffer, axis=0).astype(
            self.np_dtype, copy=False
        )
        provenance = {
            key: np.concatenate(values, axis=0)
            for key, values in self._provenance_buffer.items()
        }
        rows = provenance["cache_row"].astype(np.int64, copy=False)
        if len(rows) != len(features):
            raise ValueError("Feature/provenance shard size mismatch.")

        relative_shard_path = (
            f"{FEATURE_SHARD_DIR}/"
            f"{FEATURE_SHARD_PREFIX}{rank:05d}-part={self._part_counter:08d}.npy"
        )
        feature_path = self.output_dir / relative_shard_path
        np.save(feature_path, features)

        fragment_path = (
            self.output_dir
            / INDEX_FRAGMENT_DIR
            / f"{INDEX_FRAGMENT_PREFIX}{rank:05d}-part={self._part_counter:08d}.parquet"
        )
        _write_index_fragment(
            fragment_path,
            row_id=rows,
            split_names=self.split_names,
            dataset_info=self.dataset_info,
            provenance=provenance,
            shard_path=relative_shard_path,
        )

        self._part_counter += 1
        self._buffer_rows = 0
        self._feature_buffer.clear()
        self._provenance_buffer = {key: [] for key in PROVENANCE_KEYS}

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._append_prediction(prediction)
        if self._rows_per_shard is not None and self._buffer_rows >= self._rows_per_shard:
            self._flush(rank=int(trainer.global_rank))

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        self._flush(rank=int(trainer.global_rank))


class SlotCachePredictDataModule(pl.LightningDataModule):
    """Build cache dataloaders after Lightning has initialized distributed ranks."""

    def __init__(self, datamodule: pl.LightningDataModule, splits: List[str]):
        super().__init__()
        self.datamodule = datamodule
        self.splits = list(splits)

    def predict_dataloader(self):
        rank, world_size = _distributed_rank_world()
        return self.datamodule.cache_dataloaders(
            self.splits,
            distributed=world_size > 1,
            rank=rank,
            world_size=world_size,
        )


def _array_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_index_table(
    *,
    row_id: np.ndarray,
    split_names: List[str],
    dataset_info: List[Dict[str, Any]],
    provenance: Dict[str, np.ndarray],
    shard_path: str,
) -> pa.Table:
    row_id = row_id.astype(np.int64, copy=False)
    dataset_index = provenance["dataset_index"].astype(np.int64, copy=False)
    split_index = provenance["split_index"].astype(np.int64, copy=False)
    episode_index = provenance["episode_index"].astype(np.int64, copy=False)
    frame_index = provenance["frame_index"].astype(np.int64, copy=False)
    absolute_index = provenance["absolute_index"].astype(np.int64, copy=False)
    timestamp = provenance["timestamp"].astype(np.float64, copy=False)
    task_index = provenance["task_index"].astype(np.int64, copy=False)
    context_absolute_indices = provenance["context_absolute_indices"].astype(np.int64, copy=False)
    context_frame_indices = provenance["context_frame_indices"].astype(np.int64, copy=False)
    context_is_pad = provenance["context_is_pad"].astype(bool, copy=False)

    repo_ids = []
    roots = []
    camera_keys = []
    splits = []
    for split_idx, dataset_idx in zip(split_index, dataset_index):
        splits.append(split_names[int(split_idx)])
        info = dataset_info[int(dataset_idx)]
        repo_ids.append(info.get("repo_id"))
        roots.append(info.get("root"))
        camera_keys.append(info.get("camera_key"))

    table = pa.Table.from_pydict(
        {
            "row_id": row_id,
            "split": splits,
            "dataset_index": dataset_index,
            "repo_id": repo_ids,
            "root": roots,
            "camera_key": camera_keys,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "absolute_index": absolute_index,
            "timestamp": timestamp,
            "task_index": task_index,
            "context_absolute_indices": pa.array(
                context_absolute_indices.tolist(), type=pa.list_(pa.int64())
            ),
            "context_frame_indices": pa.array(
                context_frame_indices.tolist(), type=pa.list_(pa.int64())
            ),
            "context_is_pad": pa.array(context_is_pad.tolist(), type=pa.list_(pa.bool_())),
            "shard_path": [shard_path] * len(row_id),
            "shard_offset": np.arange(len(row_id), dtype=np.int64),
            "shard_num_rows": np.full(len(row_id), len(row_id), dtype=np.int64),
        }
    )
    return table


def _write_index_fragment(
    path: pathlib.Path,
    *,
    row_id: np.ndarray,
    split_names: List[str],
    dataset_info: List[Dict[str, Any]],
    provenance: Dict[str, np.ndarray],
    shard_path: str,
):
    table = _build_index_table(
        row_id=row_id,
        split_names=split_names,
        dataset_info=dataset_info,
        provenance=provenance,
        shard_path=shard_path,
    )
    pq.write_table(table, path)


def _read_npy_header(path: pathlib.Path) -> Tuple[Tuple[int, ...], np.dtype]:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported npy format version {version} in `{path}`.")
    if fortran_order:
        raise ValueError(f"Expected C-contiguous npy shard, but `{path}` is Fortran-order.")
    return tuple(shape), np.dtype(dtype)


def _finalize_index(
    output_dir: pathlib.Path,
    *,
    expected_rows: int,
    dtype: str,
    metadata: Dict[str, Any],
):
    output_dir = pathlib.Path(output_dir)
    fragments = sorted(
        (output_dir / INDEX_FRAGMENT_DIR).glob(f"{INDEX_FRAGMENT_PREFIX}*.parquet")
    )
    if not fragments:
        raise RuntimeError(f"No cache index fragments found in `{output_dir}`.")

    table = pa.concat_tables([pq.read_table(fragment) for fragment in fragments])
    missing_fields = [name for name in INDEX_FIELD_NAMES if name not in table.column_names]
    if missing_fields:
        raise ValueError(f"Index fragments are missing fields: {missing_fields}")

    rows = np.asarray(table["row_id"].to_pylist(), dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= expected_rows):
        bad_rows = rows[(rows < 0) | (rows >= expected_rows)][:10].tolist()
        raise ValueError(f"Cache row id out of bounds: {bad_rows}")

    seen = np.zeros(expected_rows, dtype=bool)
    duplicated = []
    for row in rows:
        if seen[row]:
            duplicated.append(int(row))
            if len(duplicated) >= 10:
                break
        seen[row] = True
    if duplicated:
        raise ValueError(f"Duplicate cache rows found: {duplicated}")
    if not seen.all():
        missing = np.flatnonzero(~seen)[:10].tolist()
        raise ValueError(f"Missing cache rows: {missing}")

    np_dtype = _numpy_dtype(dtype)
    shard_paths = table["shard_path"].to_pylist()
    shard_num_rows = table["shard_num_rows"].to_pylist()
    shard_offset = np.asarray(table["shard_offset"].to_pylist(), dtype=np.int64)
    feature_shape = None
    shard_shapes = {}
    for shard_path in sorted(set(shard_paths)):
        full_path = output_dir / shard_path
        if not full_path.exists():
            raise FileNotFoundError(f"Feature shard `{full_path}` is missing.")
        shape, shard_dtype = _read_npy_header(full_path)
        if len(shape) != 3:
            raise ValueError(f"Expected shard `{full_path}` with shape [N, slots, dim].")
        if shard_dtype != np_dtype:
            raise ValueError(
                f"Expected shard `{full_path}` dtype {np_dtype}, but found {shard_dtype}."
            )
        if feature_shape is None:
            feature_shape = tuple(shape[1:])
        elif tuple(shape[1:]) != feature_shape:
            raise ValueError(f"Feature shape mismatch in `{full_path}`.")
        shard_shapes[shard_path] = shape

    for shard_path, declared_rows in zip(shard_paths, shard_num_rows):
        actual_rows = shard_shapes[shard_path][0]
        if int(declared_rows) != actual_rows:
            raise ValueError(
                f"Index declares {declared_rows} rows for `{shard_path}`, "
                f"but shard contains {actual_rows}."
            )
    if np.any(shard_offset < 0):
        raise ValueError("Negative shard offsets found in index fragments.")
    for shard_path in sorted(set(shard_paths)):
        offsets = shard_offset[np.asarray(shard_paths) == shard_path]
        if np.any(offsets >= shard_shapes[shard_path][0]):
            raise ValueError(f"Shard offsets out of bounds for `{shard_path}`.")

    order = np.argsort(rows, kind="stable")
    table = table.take(pa.array(order, type=pa.int64()))
    pq.write_table(table, output_dir / INDEX_FILENAME)

    metadata = dict(metadata)
    metadata["complete"] = True
    metadata["feature_shape"] = [expected_rows, *feature_shape]
    metadata["num_rows"] = expected_rows
    metadata["num_shards"] = len(shard_shapes)
    with open(output_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def finalize_sharded_cache(
    output_dir: pathlib.Path,
    *,
    expected_rows: int,
    dtype: str,
    metadata: Dict[str, Any],
):
    _finalize_index(
        output_dir,
        expected_rows=expected_rows,
        dtype=dtype,
        metadata=metadata,
    )


class SlotFeatureCache:
    """Read slot feature cache files produced by this module."""

    def __init__(self, cache_dir: str, shard_cache_size: int = 2):
        self.cache_dir = pathlib.Path(cache_dir)
        with open(self.cache_dir / METADATA_FILENAME, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        if self.metadata.get("cache_format") != CACHE_FORMAT:
            raise ValueError(
                f"Unsupported cache format `{self.metadata.get('cache_format')}`. "
                f"Expected `{CACHE_FORMAT}`."
            )
        if not self.metadata.get("complete", False):
            raise ValueError(f"Slot feature cache `{self.cache_dir}` is incomplete.")

        self.index = pq.read_table(self.cache_dir / INDEX_FILENAME)
        self.shard_cache_size = int(shard_cache_size)
        self._shard_cache: "collections.OrderedDict[str, np.ndarray]" = (
            collections.OrderedDict()
        )
        self._row_by_dataset_key = {}
        self._row_by_repo_key = {}
        self._feature_by_row = {}
        self._build_lookup()

    @staticmethod
    def _insert_lookup(mapping, key, row_id):
        if key in mapping:
            mapping[key] = None
        else:
            mapping[key] = row_id

    def _build_lookup(self):
        columns = {
            name: self.index[name].to_pylist()
            for name in (
                "row_id",
                "dataset_index",
                "repo_id",
                "episode_index",
                "frame_index",
                "shard_path",
                "shard_offset",
            )
        }
        for row_id, dataset_index, repo_id, episode_index, frame_index, shard_path, offset in zip(
            columns["row_id"],
            columns["dataset_index"],
            columns["repo_id"],
            columns["episode_index"],
            columns["frame_index"],
            columns["shard_path"],
            columns["shard_offset"],
        ):
            row_id = int(row_id)
            self._feature_by_row[row_id] = (str(shard_path), int(offset))
            self._insert_lookup(
                self._row_by_dataset_key,
                (int(dataset_index), int(episode_index), int(frame_index)),
                row_id,
            )
            self._insert_lookup(
                self._row_by_repo_key,
                (str(repo_id), int(episode_index), int(frame_index)),
                row_id,
            )

    def lookup(
        self,
        *,
        episode_index: int,
        frame_index: int,
        dataset_index: Optional[int] = None,
        repo_id: Optional[str] = None,
    ) -> int:
        if dataset_index is None and repo_id is None:
            raise ValueError("Pass either `dataset_index` or `repo_id`.")
        if dataset_index is not None:
            key = (int(dataset_index), int(episode_index), int(frame_index))
            row_id = self._row_by_dataset_key.get(key)
        else:
            key = (str(repo_id), int(episode_index), int(frame_index))
            row_id = self._row_by_repo_key.get(key)
        if row_id is None:
            raise KeyError(f"Ambiguous or missing slot feature cache key: {key}")
        return int(row_id)

    def _load_shard(self, shard_path: str) -> np.ndarray:
        if self.shard_cache_size > 0 and shard_path in self._shard_cache:
            shard = self._shard_cache.pop(shard_path)
            self._shard_cache[shard_path] = shard
            return shard

        shard = np.load(self.cache_dir / shard_path, mmap_mode=None)
        if self.shard_cache_size > 0:
            self._shard_cache[shard_path] = shard
            while len(self._shard_cache) > self.shard_cache_size:
                self._shard_cache.popitem(last=False)
        return shard

    def get(self, row_id: int) -> np.ndarray:
        return self.get_many([row_id])[0]

    def get_many(self, row_ids: Sequence[int]) -> np.ndarray:
        row_ids = [int(row_id) for row_id in row_ids]
        feature_shape = tuple(self.metadata["feature_shape"][1:])
        dtype = _numpy_dtype(self.metadata["dtype"])
        if not row_ids:
            return np.empty((0, *feature_shape), dtype=dtype)

        grouped: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        for output_index, row_id in enumerate(row_ids):
            try:
                shard_path, offset = self._feature_by_row[row_id]
            except KeyError as exc:
                raise KeyError(f"Missing slot feature cache row: {row_id}") from exc
            grouped[shard_path].append((output_index, offset))

        output = np.empty((len(row_ids), *feature_shape), dtype=dtype)
        for shard_path, items in grouped.items():
            shard = self._load_shard(shard_path)
            offsets = [offset for _, offset in items]
            values = shard[offsets]
            for value_index, (output_index, _) in enumerate(items):
                output[output_index] = values[value_index]
        return output

    def get_by_key(self, **kwargs) -> np.ndarray:
        return self.get(self.lookup(**kwargs))


def _load_model(config, checkpoint_path: pathlib.Path):
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _count_rows(datasets_or_loaders) -> int:
    return sum(len(getattr(item, "dataset", item)) for item in datasets_or_loaders)


def _build_metadata(
    *,
    args,
    splits: List[str],
    dataset_info: List[Dict[str, Any]],
    expected_rows: int,
    cache_seed: int,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cache_format": CACHE_FORMAT,
        "complete": False,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_path": os.fspath(pathlib.Path(args.config).resolve()),
        "checkpoint_path": os.fspath(pathlib.Path(args.checkpoint).resolve()),
        "git_commit": _git_commit(pathlib.Path.cwd()),
        "slot_source": "outputs.processor.state[:, -1]",
        "cache_seed": int(cache_seed),
        "dtype": args.dtype,
        "frame_offsets": list(dataset_info[0].get("frame_offsets", []))
        if dataset_info
        else [],
        "splits": splits,
        "dataset_info": dataset_info,
        "expected_rows": expected_rows,
        "target_shard_mb": args.target_shard_mb,
        "feature_layout": {
            "layout": "nested",
            "feature_dir": FEATURE_SHARD_DIR,
            "index_fragment_dir": INDEX_FRAGMENT_DIR,
            "shard_pattern": (
                f"{FEATURE_SHARD_DIR}/"
                f"{FEATURE_SHARD_PREFIX}{{rank:05d}}-part={{part:08d}}.npy"
            ),
            "index_fragment_pattern": (
                f"{INDEX_FRAGMENT_DIR}/"
                f"{INDEX_FRAGMENT_PREFIX}{{rank:05d}}-part={{part:08d}}.parquet"
            ),
        },
        "config_overrides": list(args.config_overrides),
    }


def run(args) -> int:
    rank = _rank_from_env()
    world_size = _world_size_from_env()
    output_dir = pathlib.Path(args.output_dir)
    _prepare_output_dir(output_dir, overwrite=args.overwrite, rank=rank)

    if args.use_optimizations:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = configuration.load_config(args.config, args.config_overrides)
    if args.config_overrides_file is not None:
        config = configuration.override_config(
            config,
            override_config_path=args.config_overrides_file,
            additional_overrides=args.config_overrides,
        )

    cache_seed = int(args.seed)
    pl.seed_everything(cache_seed, workers=True)

    datamodule = data.build(config.dataset, data_dir=args.data_dir)
    splits = args.splits
    cache_datasets = datamodule.cache_datasets(splits)
    expected_rows = _count_rows(cache_datasets)
    dataset_info = _sanitize_dataset_info(datamodule.dataset_info)
    frame_offsets = datamodule._get_frame_offsets()
    for info in dataset_info:
        info["frame_offsets"] = frame_offsets

    model = _load_model(config, pathlib.Path(args.checkpoint))
    predictor = SlotFeaturePredictor(model, output_dtype=_torch_dtype(args.dtype))
    writer = SlotFeaturePredictionWriter(
        output_dir,
        split_names=splits,
        dataset_info=dataset_info,
        dtype=args.dtype,
        target_shard_mb=args.target_shard_mb,
    )

    trainer_config = _setup_trainer_config(config.setdefault("trainer", {}))
    trainer_config["replace_sampler_ddp"] = False
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        callbacks=[writer],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=not args.quiet and not args.no_interactive,
        enable_model_summary=not args.quiet,
        **trainer_config,
    )
    trainer.predict(
        model=predictor,
        datamodule=SlotCachePredictDataModule(datamodule, splits),
        return_predictions=False,
    )
    _distributed_barrier()

    if rank == 0:
        metadata = _build_metadata(
            args=args,
            splits=splits,
            dataset_info=dataset_info,
            expected_rows=expected_rows,
            cache_seed=cache_seed,
        )
        finalize_sharded_cache(
            output_dir,
            expected_rows=expected_rows,
            dtype=args.dtype,
            metadata=metadata,
        )
        for fragment in (output_dir / INDEX_FRAGMENT_DIR).glob(
            f"{INDEX_FRAGMENT_PREFIX}*.parquet"
        ):
            fragment.unlink()
    _distributed_barrier()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache VideoSAUR slot features for LeRobot data.")
    parser.add_argument("config", help="VideoSAUR configuration to use.")
    parser.add_argument(
        "config_overrides", nargs="*", help="Additional OmegaConf dotlist overrides."
    )
    parser.add_argument("--checkpoint", required=True, help="VideoSAUR checkpoint to load.")
    parser.add_argument("--output-dir", required=True, help="Directory for the slot feature cache.")
    parser.add_argument("--splits", type=_parse_splits, default=_parse_splits("train,val,test"))
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible slot initialization during cache generation.",
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dir.")
    parser.add_argument(
        "--target-shard-mb",
        type=float,
        default=256.0,
        help="Target feature shard size in MiB.",
    )
    parser.add_argument("--data-dir", help="Path to data directory.")
    parser.add_argument("--config_overrides_file", help="Configuration to override.")
    parser.add_argument(
        "--no-interactive", action="store_true", help="Disable interactive progress bars."
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress most output.")
    parser.add_argument(
        "--use-optimizations", action="store_true", help="Enable PyTorch performance optimizations."
    )
    return parser


def main():
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
