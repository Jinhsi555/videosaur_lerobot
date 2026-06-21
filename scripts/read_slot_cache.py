#!/usr/bin/env python
"""Inspect and read VideoSAUR LeRobot slot feature caches."""

import argparse
import collections
import json
import pathlib
from typing import Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

CACHE_FORMAT = "sharded_npy"
INDEX_FILENAME = "index.parquet"
METADATA_FILENAME = "metadata.json"


class SlotFeatureCacheReader:
    """Lightweight cache reader for sharded npy slot feature caches."""

    def __init__(self, cache_dir: pathlib.Path, shard_cache_size: int = 2):
        self.cache_dir = pathlib.Path(cache_dir)
        self.index = pq.read_table(self.cache_dir / INDEX_FILENAME)
        self.metadata = _load_metadata(self.cache_dir)
        if self.metadata.get("cache_format") != CACHE_FORMAT:
            raise ValueError(
                f"Unsupported cache format `{self.metadata.get('cache_format')}`. "
                f"Expected `{CACHE_FORMAT}`."
            )
        if not self.metadata.get("complete", False):
            raise ValueError(f"Slot feature cache `{self.cache_dir}` is incomplete.")
        self.shard_cache_size = int(shard_cache_size)
        self._shard_cache = collections.OrderedDict()
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
        dtype = np.dtype(self.metadata["dtype"])
        if not row_ids:
            return np.empty((0, *feature_shape), dtype=dtype)

        grouped = collections.defaultdict(list)
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


def _format_row(row: dict) -> str:
    return (
        f"row_id={row['row_id']} split={row['split']} "
        f"dataset_index={row['dataset_index']} repo_id={row['repo_id']} "
        f"episode={row['episode_index']} frame={row['frame_index']} "
        f"absolute_index={row['absolute_index']} timestamp={row['timestamp']}"
    )


def _load_metadata(cache_dir: pathlib.Path) -> dict:
    metadata_path = cache_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_summary(cache_dir: pathlib.Path, head: int):
    index = pq.read_table(cache_dir / INDEX_FILENAME)
    metadata = _load_metadata(cache_dir)

    print(f"cache_dir: {cache_dir}")
    print(f"cache_format: {metadata.get('cache_format')}")
    print(f"complete: {metadata.get('complete')}")
    print(f"features: shape={metadata.get('feature_shape')} dtype={metadata.get('dtype')}")
    print(f"num_shards: {metadata.get('num_shards')}")
    print(f"index rows: {index.num_rows}")
    if metadata:
        print(f"slot_source: {metadata.get('slot_source')}")
        print(f"splits: {metadata.get('splits')}")
        print(f"frame_offsets: {metadata.get('frame_offsets')}")
        print("datasets:")
        for idx, dataset in enumerate(metadata.get("dataset_info", [])):
            print(
                f"  {idx}: repo_id={dataset.get('repo_id')} "
                f"camera_key={dataset.get('camera_key')} root={dataset.get('root')}"
            )

    if head > 0:
        print(f"index head ({min(head, index.num_rows)} rows):")
        for row in index.slice(0, head).to_pylist():
            print(f"  {_format_row(row)}")


def _lookup_row(args, cache: SlotFeatureCacheReader) -> Optional[int]:
    if args.row is not None:
        return int(args.row)

    has_key_lookup = args.episode_index is not None and args.frame_index is not None
    if not has_key_lookup:
        return None

    if args.dataset_index is None and args.repo_id is None:
        raise SystemExit(
            "Pass --dataset-index or --repo-id together with --episode-index and --frame-index."
        )

    return cache.lookup(
        dataset_index=args.dataset_index,
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        frame_index=args.frame_index,
    )


def _print_feature(
    cache: SlotFeatureCacheReader,
    row_id: int,
    max_values: int,
    save_path: Optional[str],
):
    feature = cache.get(row_id)
    print(f"selected row: {row_id}")
    print(f"feature: shape={feature.shape} dtype={feature.dtype}")
    print(
        "feature stats: "
        f"min={float(np.min(feature)):.6g} "
        f"max={float(np.max(feature)):.6g} "
        f"mean={float(np.mean(feature)):.6g} "
        f"std={float(np.std(feature)):.6g}"
    )

    if max_values > 0:
        values = feature.reshape(-1)[:max_values]
        print(f"first {len(values)} values: {values.tolist()}")

    if save_path is not None:
        output_path = pathlib.Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, np.asarray(feature))
        print(f"saved feature to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read a VideoSAUR slot feature cache.")
    parser.add_argument("cache_dir", help="Directory containing sharded features and index.parquet.")
    parser.add_argument("--head", type=int, default=5, help="Number of index rows to print.")
    parser.add_argument("--row", type=int, help="Read feature by row_id.")
    parser.add_argument("--dataset-index", type=int, help="Lookup by dataset index.")
    parser.add_argument("--repo-id", help="Lookup by repo_id.")
    parser.add_argument("--episode-index", type=int, help="Lookup episode index.")
    parser.add_argument("--frame-index", type=int, help="Lookup frame index.")
    parser.add_argument(
        "--max-values",
        type=int,
        default=12,
        help="Number of flattened feature values to print for the selected row.",
    )
    parser.add_argument(
        "--shard-cache-size",
        type=int,
        default=2,
        help="Number of feature shards to keep in the local LRU cache.",
    )
    parser.add_argument("--save-feature", help="Optional .npy path for the selected feature.")
    return parser


def main():
    args = build_parser().parse_args()
    cache_dir = pathlib.Path(args.cache_dir)
    _print_summary(cache_dir, head=args.head)

    cache = SlotFeatureCacheReader(cache_dir, shard_cache_size=args.shard_cache_size)
    row_id = _lookup_row(args, cache)
    if row_id is not None:
        _print_feature(cache, row_id, args.max_values, args.save_feature)


if __name__ == "__main__":
    main()
