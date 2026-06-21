import json
import math
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from omegaconf import ListConfig
from torch.utils.data._utils import collate as torch_collate

from videosaur.data import pipelines, transforms
from videosaur.data.utils import get_data_root_dir, worker_init_function
from videosaur.utils import config_as_kwargs


def build(config, name: Optional[str] = "WebdatasetDataModule", data_dir: Optional[str] = None):
    name = config.get("name") or name
    if name == "WebdatasetDataModule":
        train_pipeline = None
        if config.train_pipeline:
            train_pipeline = pipelines.build(config.train_pipeline, shuffle=True)

        val_pipeline = None
        if config.val_pipeline:
            val_pipeline = pipelines.build(config.val_pipeline, shuffle=False)

        return WebdatasetDataModule(
            data_dir=data_dir,
            train_pipeline=train_pipeline,
            val_pipeline=val_pipeline,
            **config_as_kwargs(config, to_filter=("train_pipeline", "val_pipeline")),
        )
    elif name == "DummyDataModule":
        return DummyDataModule(
            train_transforms=transforms.build(config.train_transforms),
            val_transforms=transforms.build(config.val_transforms),
            **config_as_kwargs(
                config,
                to_filter=(
                    "train_transforms",
                    "val_transforms",
                ),
            ),
        )
    elif name == "LeRobotDataModule":
        return LeRobotDataModule(
            data_dir=data_dir,
            **config_as_kwargs(config),
        )
    elif name == "MixedLeRobotDataModule":
        return MixedLeRobotDataModule(
            data_dir=data_dir,
            **config_as_kwargs(config),
        )
    else:
        raise ValueError(f"Unknown dataset module `{name}`")


class WebdatasetDataModule(pl.LightningDataModule):
    """DatasetModule for webdataset datasets.

    We primarily rely on iteration-based instead of epoch-based training. Epoch-based training is
    difficult to realize with distributed training (i.e. multi-GPU), because it is hard to ensure
    that each sample is sampled exactly once per epoch. Instead, iteration-based training instead
    just samples a random stream of data. That said, this module does also support epoch-based
    training by setting the `samples_per_epoch` argument. In this case, the dataloader stops after
    `samples_per_epoch // (batch_size * num_nodes)` samples.

    For validation/testing, we need to make sure that each sample is seen exactly once. To do so,
    this module adds padding entries that should be ignored using the "batch_padding_mask" key.
    With distributed training, it is required to specify the number of samples the dataset contains
    using the `val_size`, `test_size` arguments.

    The arguments `val_size` and `test_size` refer to the total number of input samples contained
    in all shards of the split. As the number of samples can be changed by the data pipeline, it is
    the responsibility of the data pipeline to correctly specify how many samples it will output
    using the `get_num_samples` method.
    """

    BATCH_PADDING_MASK_KEY = "batch_padding_mask"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        train_shards: Optional[Union[str, List[str]]] = None,
        val_shards: Optional[Union[str, List[str]]] = None,
        test_shards: Optional[Union[str, List[str]]] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
        train_pipeline: Optional[pipelines.DataPipeline] = None,
        val_pipeline: Optional[pipelines.DataPipeline] = None,
        test_pipeline: Optional[pipelines.DataPipeline] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        num_val_workers: Optional[int] = None,
        cache_train: bool = False,
        cache_val: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        data_dir = data_dir if data_dir else get_data_root_dir()

        def get_shards_and_num_shards(shards):
            if shards is None:
                return None, 0
            if isinstance(shards, ListConfig):
                new_shards = []
                for s in shards:
                    new_shards.extend(get_shards_and_num_shards(s)[0])
                shards = new_shards
            else:
                shards = _to_abs_shard_path(shards, data_dir)
                shards = wds.shardlists.expand_urls(shards)
            return shards, len(shards)

        self.train_shards, self.num_train_shards = get_shards_and_num_shards(train_shards)
        self.val_shards, self.num_val_shards = get_shards_and_num_shards(val_shards)
        self.test_shards, self.num_test_shards = get_shards_and_num_shards(test_shards)
        self.val_size = val_size
        self.test_size = test_size
        self.samples_per_epoch = samples_per_epoch
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline if test_pipeline else val_pipeline
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_train_workers = num_workers
        self.num_val_workers = num_val_workers if num_val_workers is not None else num_workers

        if cache_dir is None and (cache_train or cache_val):
            cache_dir = tempfile.mkdtemp(prefix="wds_shardcache_", dir="/tmp")
        self.cache_dir = cache_dir
        self.cache_train = cache_train
        self.cache_val = cache_val

        self.num_nodes = None  # Set lazily

    def __str__(self) -> str:
        val_size = "?" if self.val_size is None else self.val_size
        test_size = "?" if self.test_size is None else self.test_size
        samples_per_epoch = (
            "unspecified" if self.samples_per_epoch is None else self.samples_per_epoch
        )
        res = [
            "WebdatasetDataModule",
            f"  - Number of train shards: {self.num_train_shards}",
            f"  - Number of val shards: {self.num_val_shards}",
            f"  - Number of test shards: {self.num_test_shards}",
            f"  - Assumed number of val samples: {val_size}",
            f"  - Assumed number of test samples: {test_size}",
            f"  - Specified length of training epoch: {samples_per_epoch}",
            f"  - Train batch size: {self.batch_size}",
            f"  - Eval batch size: {self.val_batch_size}",
            f"  - Number of train workers: {self.num_train_workers}",
            f"  - Number of eval workers: {self.num_val_workers}",
        ]
        return "\n".join(res)

    def _verify_settings_lazy(self):
        """Check that we have appropriate settings for the distributed setting.

        We can only do this lazily (once a dataloader is requested) because on __init__ the
        number of nodes is not known.
        """
        if self.num_nodes is not None:
            return

        self.num_nodes = wds.utils.pytorch_worker_info()[1]

        if self.num_nodes > 1:
            if self.val_shards and self.val_size is None:
                raise ValueError("Need to specify `val_size` in distributed setting")
            if self.test_shards and self.test_size is None:
                raise ValueError("Need to specify `test_size` in distributed setting")

        def _check_workers_and_shards(split: str, num_workers: int, num_shards: int) -> int:
            min_shards_per_node = num_shards // self.num_nodes

            if min_shards_per_node == 0:
                raise ValueError(
                    f"The number of compute nodes is {self.num_nodes}, but the "
                    f"number of {split} shards is only {num_shards}. Increase the number of shards."
                )

            if num_workers > min_shards_per_node:
                raise ValueError(
                    f"The number of {split} workers is {num_workers}, but the minimum number of "
                    f"shards per compute node is only {min_shards_per_node}. Reduce the number of "
                    f"workers to <={min_shards_per_node}."
                )

        if self.train_shards:
            _check_workers_and_shards("train", self.num_train_workers, self.num_train_shards)
        if self.val_shards:
            _check_workers_and_shards("val", self.num_val_workers, self.num_val_shards)
        if self.test_shards:
            _check_workers_and_shards("test", self.num_val_workers, self.num_test_shards)

    @staticmethod
    def _filter_properties(
        input_dict: Dict[str, Any], prefixes_to_keep: Tuple[str]
    ) -> Dict[str, Any]:
        prefixes_to_keep = ("_",) + prefixes_to_keep  # Keep underscore properties like "__key__"
        out_dict = {}
        for key, value in input_dict.items():
            if any(key.startswith(prefix) for prefix in prefixes_to_keep):
                out_dict[key] = value

        return out_dict

    @staticmethod
    def _remove_extensions(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {name.split(".")[0]: value for name, value in input_dict.items()}

    @staticmethod
    def _pad(dataset: Iterable[Dict[str, Any]], n_samples: int) -> Iterable[Dict[str, Any]]:
        """Iterates dataset, then adds dummy samples until reaching the specified number of samples.

        Dummy samples are constructed by copying structure of the first encountered sample.
        Also adds a special property "batch_padding_mask" to indicate which entries are padding.
        """
        example = None
        count = 0
        for sample in dataset:
            if example is None:
                example = sample
            count += 1
            yield {**sample, WebdatasetDataModule.BATCH_PADDING_MASK_KEY: np.array(False)}

        while count < n_samples:
            if example is None:
                sample = {}  # Dataset was empty, should not really happen
            else:
                sample = {
                    key: WebdatasetDataModule._get_padding(key, value)
                    for key, value in example.items()
                }
            sample[WebdatasetDataModule.BATCH_PADDING_MASK_KEY] = np.array(True)
            count += 1
            yield sample

    @staticmethod
    def _get_padding(key: str, value: Any):
        """Construct padding for property."""
        if isinstance(value, str):
            return "PADDING"
        elif isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
        else:
            return np.zeros_like(value)

    def _get_max_samples_per_worker(
        self, dataset_size: int, num_shards: int, num_workers: int
    ) -> int:
        """Estimate upper bound on the number of samples per data worker.

        It is only approximate because we don't know the exact composition of the shards. If the
        number of samples per shard is very different, this estimate may be too low.
        """
        num_workers = max(num_workers, 1)
        max_samples_per_shard = int(math.ceil(dataset_size / num_shards))
        max_shards_per_node = int(math.ceil(num_shards / self.num_nodes))
        max_shards_per_worker = int(math.ceil(max_shards_per_node / num_workers))
        return max_shards_per_worker * max_samples_per_shard

    @staticmethod
    def _get_webdataset(
        urls, resampled=False, splitter=None, cache_size=-1, cache_dir=None
    ) -> wds.FluidWrapper:
        """Create pipeline object serving same function as wds.WebDataset.

        We do this instead of directly using wds.WebDataset in order to have control over
        the `always` argument for caching.

        This method either creates a shuffling, resampling dataset for `resampled=True`, or a
        deterministic, non-shuffling for `resample=False`. We do not need other modes for now.
        """
        if resampled:
            shardlist = wds.shardlists.ResampledShards(urls, deterministic=True)
        else:
            shardlist = wds.shardlists.SimpleShardList(urls)

        dataset = wds.FluidWrapper(shardlist)

        if not resampled:
            if splitter is None:
                splitter = wds.shardlists.single_node_only
            dataset.append(splitter)
            dataset.append(wds.shardlists.split_by_worker)

        handler = wds.filters.reraise_exception
        if cache_dir is None or cache_size == 0:
            dataset.append(wds.tariterators.tarfile_to_samples(handler=handler))
        else:
            assert cache_size == -1 or cache_size > 0
            dataset.append(
                wds.cache.cached_tarfile_to_samples(
                    handler=handler,
                    verbose=False,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                    always=True,
                )
            )

        return dataset

    def _get_dataset(
        self,
        shards: Union[str, List[str]],
        shuffle: bool = False,
        pipeline: Optional[pipelines.DataPipeline] = None,
        padded_size_per_worker: Optional[int] = None,
        cache_dir: Optional[str] = None,
        cache_size: int = -1,
    ):
        if shuffle:
            # For shuffling samples, we sample shards with replacement. This means that each node
            # and worker uses all shards from the dataset (instead of splitting shards).
            dataset = self._get_webdataset(
                shards, resampled=True, cache_dir=cache_dir, cache_size=cache_size
            )
        else:
            splitter = (
                wds.shardlists.split_by_node
                if self.num_nodes > 1
                else wds.shardlists.single_node_only
            )
            dataset = self._get_webdataset(
                shards, splitter=splitter, cache_dir=cache_dir, cache_size=cache_size
            )

        # Filter unneeded properties first to avoid decoding them. If pipeline defines no keys,
        # keep everything.
        if pipeline and pipeline.keys is not None:
            dataset = dataset.map(
                partial(WebdatasetDataModule._filter_properties, prefixes_to_keep=pipeline.keys)
            )

        dataset = dataset.decode("rgb").map(WebdatasetDataModule._remove_extensions)

        if padded_size_per_worker is not None:
            # Pad dataset stream to contain a certain number of samples. This is needed to balance
            # data between nodes and workers during validation. Note that `padded_size` here refers
            # to the number of samples PER WORKER, not to the total number of samples in the dataset.
            dataset = dataset.compose(
                partial(WebdatasetDataModule._pad, n_samples=padded_size_per_worker)
            )
            # Only add length if we can be sure about the exact number of samples. If we do not add
            # padding and only know the global dataset size, this is not the case, because samples
            # may be unevenly distributed over nodes and workers.
            dataset = dataset.with_length(padded_size_per_worker)

        if pipeline:
            dataset = pipeline.apply(dataset)

            if padded_size_per_worker and pipeline.get_num_samples(padded_size_per_worker):
                dataset = dataset.with_length(pipeline.get_num_samples(padded_size_per_worker))

        return dataset

    def _get_dataloader(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        num_samples_per_epoch: Optional[int],
        partial_batches: bool = False,
    ):
        assert num_samples_per_epoch is None or num_samples_per_epoch > 0

        # Do batching within each worker
        dataset_batched = dataset.batched(
            batch_size, partial=partial_batches, collation_fn=torch_collate.default_collate
        )

        dataloader = wds.WebLoader(
            dataset_batched,
            num_workers=num_workers,
            batch_size=None,
            worker_init_fn=worker_init_function,
            persistent_workers=num_workers > 0,
            # Heuristic to check whether GPUs are used. Misses the case where num_gpus = 1
            pin_memory=(self.num_nodes > 1 and torch.cuda.is_available()),
            prefetch_factor=2,
        )

        num_batches_per_epoch = None
        if num_samples_per_epoch is not None:
            if partial_batches:
                num_batches_per_epoch = int(
                    math.ceil(num_samples_per_epoch / (batch_size * self.num_nodes))
                )
                dataloader = dataloader.with_epoch(num_batches_per_epoch)
            else:
                num_batches_per_epoch = num_samples_per_epoch // (batch_size * self.num_nodes)
                # Equalize batches across nodes and workers for DDP. This may lead to dropping some
                # samples and repeating other samples across an epoch in case the shards are
                # unevenly distributed across nodes. This seems to be unavoidable with DDP.
                # See https://github.com/webdataset/webdataset/issues/225#issuecomment-1344642570
                dataloader = dataloader.repeat(2).with_epoch(num_batches_per_epoch)
        else:
            # Set dataset to loop indefinitely
            dataloader = dataloader.repeat()

        if num_batches_per_epoch:
            dataloader = dataloader.with_length(num_batches_per_epoch)

        return dataloader

    def train_dataset(self):
        self._verify_settings_lazy()
        if self.train_shards is None:
            raise ValueError("No training split.")

        return self._get_dataset(
            self.train_shards,
            shuffle=True,
            pipeline=self.train_pipeline,
            cache_dir=self.cache_dir if self.cache_train else None,
        )

    def val_dataset(self):
        self._verify_settings_lazy()
        if self.val_shards is None:
            raise ValueError("No validation split.")

        padded_size = self._get_max_samples_per_worker(
            self.val_size, self.num_val_shards, self.num_val_workers
        )
        return self._get_dataset(
            self.val_shards,
            shuffle=False,
            pipeline=self.val_pipeline,
            padded_size_per_worker=padded_size,
            cache_dir=self.cache_dir if self.cache_val else None,
        )

    def test_dataset(self):
        self._verify_settings_lazy()
        if self.test_shards is None:
            raise ValueError("No test split.")

        padded_size = self._get_max_samples_per_worker(
            self.test_size, self.num_test_shards, self.num_val_workers
        )
        return self._get_dataset(
            self.test_shards,
            shuffle=False,
            pipeline=self.test_pipeline,
            padded_size_per_worker=padded_size,
            cache_dir=self.cache_dir if self.cache_val else None,
        )

    def train_dataloader(self):
        return self._get_dataloader(
            self.train_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_train_workers,
            num_samples_per_epoch=self.samples_per_epoch,
            partial_batches=False,
        )

    def val_dataloader(self):
        dataset = self.val_dataset()

        try:
            num_samples_per_worker = len(dataset)
            num_samples_per_epoch = (
                num_samples_per_worker * self.num_nodes * max(self.num_val_workers, 1)
            )
        except TypeError:
            num_samples_per_epoch = None

        return self._get_dataloader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            num_samples_per_epoch=num_samples_per_epoch,
            partial_batches=True,
        )

    def test_dataloader(self):
        dataset = self.val_dataset()

        try:
            num_samples_per_worker = len(dataset)
            num_samples_per_epoch = (
                num_samples_per_worker * self.num_nodes * max(self.num_val_workers, 1)
            )
        except TypeError:
            num_samples_per_epoch = None

        return self._get_dataloader(
            self.test_dataset(),
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            num_samples_per_epoch=num_samples_per_epoch,
            partial_batches=True,
        )


def _to_abs_shard_path(
    shards: Union[str, List[str]], data_root_dir: Optional[str]
) -> Union[str, List[str]]:
    """Turn relative shard path to absolute by preprending the path to the data root directory."""
    if isinstance(shards, str):
        if os.path.isabs(shards):
            return shards  # Directly use absolute paths
        elif "://" in shards:
            return shards  # Directly use URI's like `s3://abc/xyz`
        else:
            if data_root_dir is not None:
                return os.path.join(data_root_dir, shards)
            else:
                raise ValueError(
                    f"Passed relative shard path {shards}, but data root path is missing."
                )
    else:
        assert isinstance(shards, Iterable), f"Expected iterable but found {type(shards)}"
        return [_to_abs_shard_path(shard, data_root_dir) for shard in shards]


def _to_optional_list(value):
    if value is None:
        return None
    return list(value)


def _to_plain_dict(value):
    if value is None:
        return {}
    return dict(value)


def _load_jsonl(path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_size_tuple(value: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple, ListConfig)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected int or pair for image size, but got {value}")


def _normalize_lerobot_format(value: Optional[str]) -> str:
    if value is None:
        return "auto"
    value = str(value).lower()
    aliases = {
        "2": "v2",
        "2.1": "v2",
        "v2.1": "v2",
        "3": "v3",
        "3.0": "v3",
        "v3.0": "v3",
    }
    value = aliases.get(value, value)
    if value not in ("auto", "v2", "v3"):
        raise ValueError("`lerobot_format` must be one of {'auto', 'v2', 'v3'}.")
    return value


def _resolve_lerobot_format(
    requested_format: Optional[str],
    root: Optional[str],
    *,
    force_cache_sync: bool = False,
) -> str:
    requested_format = _normalize_lerobot_format(requested_format)
    if requested_format != "auto":
        return requested_format
    if root is None or force_cache_sync:
        return "v3"

    root_path = Path(root)
    info_path = root_path / "meta" / "info.json"
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            codebase_version = str(json.load(f).get("codebase_version", "")).lower()
    except FileNotFoundError:
        codebase_version = ""

    if codebase_version.startswith("v2") or (root_path / "meta" / "episodes.jsonl").exists():
        return "v2"
    return "v3"


class _LeRobotInfoMetadata:
    """Lightweight metadata view backed only by `meta/info.json`."""

    def __init__(self, info: Dict[str, Any]):
        self.info = info

    @property
    def fps(self):
        return self.info["fps"]

    @property
    def features(self) -> Dict[str, Dict[str, Any]]:
        return self.info["features"]

    @property
    def camera_keys(self) -> List[str]:
        return [
            key
            for key, feature in self.features.items()
            if feature["dtype"] in ("video", "image")
        ]

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])


class LeRobotV2Metadata:
    """Minimal v2.1 metadata view exposing the v3 fields used by this module."""

    def __init__(self, repo_id: str, root: Union[str, os.PathLike]):
        self.repo_id = repo_id
        self.root = Path(root)
        self.info = self._load_info()
        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()

    def _load_info(self) -> Dict[str, Any]:
        info_path = self.root / "meta" / "info.json"
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        codebase_version = str(info.get("codebase_version", "")).lower()
        if codebase_version and not codebase_version.startswith("v2"):
            raise ValueError(
                f"Expected LeRobot v2 metadata at `{self.root}`, "
                f"but found codebase_version={info.get('codebase_version')!r}."
            )
        return info

    def _load_tasks(self) -> Dict[int, str]:
        path = self.root / "meta" / "tasks.jsonl"
        if not path.exists():
            return {}
        return {
            int(item["task_index"]): item["task"]
            for item in sorted(_load_jsonl(path), key=lambda item: int(item["task_index"]))
        }

    def _load_episodes(self) -> List[Dict[str, Any]]:
        path = self.root / "meta" / "episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"LeRobot v2 dataset is missing `{path}`.")

        episodes = sorted(_load_jsonl(path), key=lambda item: int(item["episode_index"]))
        if len(episodes) != int(self.info["total_episodes"]):
            raise ValueError(
                f"LeRobot v2 metadata at `{self.root}` declares "
                f"{self.info['total_episodes']} episodes but has {len(episodes)} rows."
            )

        current_index = 0
        indexed_episodes = []
        for expected_idx, episode in enumerate(episodes):
            episode_index = int(episode["episode_index"])
            if episode_index != expected_idx:
                raise ValueError(
                    "LeRobot v2 episode indices must be contiguous from zero. "
                    f"Expected {expected_idx}, got {episode_index}."
                )
            length = int(episode["length"])
            indexed_episodes.append(
                {
                    **episode,
                    "episode_index": episode_index,
                    "length": length,
                    "dataset_from_index": current_index,
                    "dataset_to_index": current_index + length,
                }
            )
            current_index += length
        return indexed_episodes

    @property
    def fps(self):
        return self.info["fps"]

    @property
    def features(self) -> Dict[str, Dict[str, Any]]:
        return self.info["features"]

    @property
    def camera_keys(self) -> List[str]:
        return [
            key
            for key, feature in self.features.items()
            if feature["dtype"] in ("video", "image")
        ]

    @property
    def video_keys(self) -> List[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "video"]

    @property
    def image_keys(self) -> List[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "image"]

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> Optional[str]:
        return self.info.get("video_path")

    @property
    def chunks_size(self) -> int:
        return int(self.info.get("chunks_size", 1000))

    def _path_kwargs(self, episode_index: int, video_key: Optional[str] = None) -> Dict[str, Any]:
        episode_chunk = int(episode_index) // self.chunks_size
        return {
            "episode_chunk": episode_chunk,
            "chunk_index": episode_chunk,
            "file_index": int(episode_index),
            "episode_index": int(episode_index),
            "video_key": video_key,
        }

    def get_data_file_path(self, episode_index: int) -> Path:
        return Path(self.data_path.format(**self._path_kwargs(episode_index)))

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        if self.video_path is None:
            raise ValueError(f"LeRobot v2 dataset `{self.repo_id}` does not define videos.")
        return Path(
            self.video_path.format(**self._path_kwargs(episode_index, video_key=video_key))
        )

    def task_string(self, task_index: int) -> str:
        return self.tasks.get(int(task_index), "")


def _try_load_lerobot_info_metadata(root: Optional[str], force_cache_sync: bool = False):
    if root is None or force_cache_sync:
        return None

    info_path = os.path.join(os.fspath(root), "meta", "info.json")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return _LeRobotInfoMetadata(json.load(f))
    except FileNotFoundError:
        return None


def _episodes_or_none_if_all(episodes: List[int], total_episodes: int):
    if len(episodes) == total_episodes:
        return None
    return episodes


class LeRobotImageTransform:
    """Resize and normalize LeRobot camera frames for VideoSAUR video inputs."""

    def __init__(
        self,
        size: Union[int, Tuple[int, int], List[int]] = 224,
        mean: Tuple[float, float, float] = tuple(transforms.IMAGENET_DEFAULT_MEAN),
        std: Tuple[float, float, float] = tuple(transforms.IMAGENET_DEFAULT_STD),
    ):
        self.size = _to_size_tuple(size)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, images):
        images = torch.as_tensor(images)
        single_image = images.ndim == 3

        if images.ndim not in (3, 4):
            raise ValueError(
                "Expected LeRobot image tensor with shape [C,H,W] or [T,C,H,W], "
                f"but got shape {tuple(images.shape)}"
            )

        if images.ndim == 3 and images.shape[0] not in (1, 3) and images.shape[-1] in (1, 3):
            images = images.permute(2, 0, 1)
        elif images.ndim == 4 and images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
            images = images.permute(0, 3, 1, 2)

        if single_image:
            images = images.unsqueeze(0)

        needs_rescale = not torch.is_floating_point(images)
        images = images.float()
        if needs_rescale or images.max() > 2.0:
            images = images / 255.0

        if images.shape[-2:] != self.size:
            images = torch.nn.functional.interpolate(
                images, size=self.size, mode="bilinear", align_corners=False
            )

        mean = self.mean.to(device=images.device, dtype=images.dtype)
        std = self.std.to(device=images.device, dtype=images.dtype)
        images = (images - mean) / std

        return images.squeeze(0) if single_image else images


class LeRobotV2Dataset(torch.utils.data.Dataset):
    """Read a local LeRobot v2.1 dataset through the v3 fields used by VideoSAUR."""

    def __init__(
        self,
        repo_id: str,
        root: Union[str, os.PathLike],
        episodes: Optional[List[int]] = None,
        image_transforms: Optional[Callable] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        tolerance_s: float = 1e-3,
        video_backend: Optional[str] = "pyav",
    ):
        super().__init__()
        if root is None:
            raise ValueError("LeRobot v2 datasets must be local and require `root`.")

        self.repo_id = repo_id
        self.root = Path(root)
        self.episodes = None if episodes is None else [int(episode) for episode in episodes]
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend
        self.meta = LeRobotV2Metadata(repo_id=repo_id, root=self.root)
        self.features = self.meta.features

        try:
            from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices
        except ImportError as exc:
            raise ImportError(
                "LeRobotV2Dataset requires the `lerobot` package utilities."
            ) from exc

        self.rows = self._load_rows()
        self._absolute_to_relative_idx = {
            int(row["index"]): relative_index for relative_index, row in enumerate(self.rows)
        }
        if self.episodes is None:
            expected_indices = list(range(len(self.rows)))
            actual_indices = [int(row["index"]) for row in self.rows]
            if actual_indices == expected_indices:
                self._absolute_to_relative_idx = None

        self.delta_indices = None
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.meta.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.meta.fps)

    def _load_rows(self) -> List[Dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("LeRobotV2Dataset requires `pyarrow` to read parquet files.") from exc

        episodes = (
            list(range(self.meta.total_episodes))
            if self.episodes is None
            else sorted(self.episodes)
        )
        rows = []
        for episode_index in episodes:
            table = pq.read_table(self.root / self.meta.get_data_file_path(episode_index))
            columns = table.to_pydict()
            for row_index in range(table.num_rows):
                rows.append({key: values[row_index] for key, values in columns.items()})
        return rows

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _scalar_item(value):
        if torch.is_tensor(value):
            return value.reshape(-1)[0].item()
        if isinstance(value, np.ndarray):
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    @staticmethod
    def _stack_values(values):
        if torch.is_tensor(values):
            return values
        return torch.stack(
            [value if torch.is_tensor(value) else torch.as_tensor(value) for value in values]
        )

    def _relative_indices(self, absolute_indices: List[int]) -> List[int]:
        if self._absolute_to_relative_idx is None:
            return absolute_indices
        return [self._absolute_to_relative_idx[int(index)] for index in absolute_indices]

    def _get_query_indices(
        self, absolute_index: int, episode_index: int
    ) -> Tuple[Dict[str, List[int]], Dict[str, torch.Tensor]]:
        episode = self.meta.episodes[int(episode_index)]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        query_indices = {
            key: [
                max(episode_start, min(episode_end - 1, absolute_index + delta))
                for delta in delta_indices
            ]
            for key, delta_indices in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [
                    absolute_index + delta < episode_start
                    or absolute_index + delta >= episode_end
                    for delta in delta_indices
                ]
            )
            for key, delta_indices in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_hf_dataset(self, query_indices: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key, absolute_indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = self._relative_indices(absolute_indices)
            values = [self.rows[index][key] for index in relative_indices]
            result[key] = self._stack_values(values)
        return result

    def _get_query_timestamps(
        self,
        current_timestamp: float,
        episode_index: int,
        query_indices: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, List[float]]:
        video_keys = (
            self.meta.video_keys
            if self.delta_timestamps is None
            else [key for key in self.meta.video_keys if key in self.delta_timestamps]
        )
        if query_indices is None:
            return {key: [current_timestamp] for key in video_keys}

        episode_start = int(self.meta.episodes[int(episode_index)]["dataset_from_index"])
        return {
            key: [
                (int(absolute_index) - episode_start) / float(self.meta.fps)
                for absolute_index in query_indices[key]
            ]
            for key in video_keys
            if key in query_indices
        }

    def _query_videos(
        self, query_timestamps: Dict[str, List[float]], episode_index: int
    ) -> Dict[str, torch.Tensor]:
        try:
            from lerobot.datasets.video_utils import decode_video_frames
        except ImportError as exc:
            raise ImportError(
                "LeRobotV2Dataset requires LeRobot video decoding utilities."
            ) from exc

        item = {}
        for video_key, timestamps in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(episode_index, video_key)
            frames = decode_video_frames(
                video_path,
                timestamps,
                self.tolerance_s,
                self.video_backend,
            )
            item[video_key] = frames.squeeze(0)
        return item

    def __getitem__(self, idx):
        item = {
            key: value if isinstance(value, str) or value is None else torch.as_tensor(value)
            for key, value in self.rows[idx].items()
        }
        episode_index = int(self._scalar_item(item["episode_index"]))
        absolute_index = int(self._scalar_item(item["index"]))

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(absolute_index, episode_index)
            item = {**item, **padding}
            item.update(self._query_hf_dataset(query_indices))

        if self.meta.video_keys:
            current_timestamp = float(self._scalar_item(item["timestamp"]))
            query_timestamps = self._get_query_timestamps(
                current_timestamp,
                episode_index,
                query_indices,
            )
            item.update(self._query_videos(query_timestamps, episode_index))

        if self.image_transforms is not None:
            for camera_key in self.meta.camera_keys:
                if camera_key in item:
                    item[camera_key] = self.image_transforms(item[camera_key])

        task_index = int(self._scalar_item(item.get("task_index", -1)))
        item["task"] = self.meta.task_string(task_index)
        return item


class LeRobotVideoDataset(torch.utils.data.Dataset):
    """Adapts a LeRobotDataset sample to VideoSAUR's `video` input key."""

    def __init__(
        self,
        dataset,
        camera_key: str,
        frame_offsets: List[int],
        episodes: Optional[List[int]] = None,
        dataset_index: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.camera_key = camera_key
        self.frame_offsets = frame_offsets
        self.episodes = None if episodes is None else {int(episode) for episode in episodes}
        self.dataset_index = None if dataset_index is None else int(dataset_index)
        self.valid_indices = self._get_valid_indices()

    def __len__(self):
        return len(self.valid_indices)

    def _get_valid_indices(self) -> List[int]:
        min_offset = min(self.frame_offsets)
        max_offset = max(self.frame_offsets)
        episodes = self.episodes
        if episodes is None:
            episodes = range(len(self.dataset.meta.episodes))

        valid_indices = []
        for episode_index in sorted(episodes):
            episode = self.dataset.meta.episodes[int(episode_index)]
            episode_start = int(episode["dataset_from_index"])
            episode_end = int(episode["dataset_to_index"])
            start = max(episode_start, episode_start - min_offset)
            stop = min(episode_end, episode_end - max_offset)
            if start < stop:
                for absolute_index in range(start, stop):
                    dataset_index = self._to_dataset_index(absolute_index)
                    if dataset_index is not None:
                        valid_indices.append(dataset_index)

        if not valid_indices:
            raise ValueError(
                "No LeRobot samples have enough context for the configured frame window."
            )
        return valid_indices

    def _to_dataset_index(self, absolute_index: int):
        absolute_to_relative = getattr(self.dataset, "_absolute_to_relative_idx", None)
        if absolute_to_relative is None:
            return absolute_index
        return absolute_to_relative.get(absolute_index)

    def __getitem__(self, idx):
        item = self.dataset[self.valid_indices[idx]]
        video = item[self.camera_key]
        if video.ndim == 3:
            video = video.unsqueeze(0)
        item = {"video": video}
        if self.dataset_index is not None:
            item["dataset_index"] = torch.tensor(self.dataset_index, dtype=torch.long)
        return item


class SequentialShardSampler(torch.utils.data.Sampler):
    """Shard a sequential dataset across ranks without padding or duplication."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0:
            raise ValueError("`world_size` must be positive.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"`rank` must be in [0, {self.world_size}), got {self.rank}.")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        dataset_len = len(self.dataset)
        if self.rank >= dataset_len:
            return 0
        return ((dataset_len - 1 - self.rank) // self.world_size) + 1


class LeRobotSlotCacheDataset(torch.utils.data.Dataset):
    """LeRobot video samples with stable per-frame provenance for slot caching."""

    def __init__(
        self,
        dataset,
        camera_key: str,
        frame_offsets: List[int],
        episodes: List[int],
        *,
        dataset_index: int,
        split_index: int,
        row_offset: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.camera_key = camera_key
        self.frame_offsets = [int(offset) for offset in frame_offsets]
        self.episodes = [int(episode) for episode in episodes]
        self.dataset_index = int(dataset_index)
        self.split_index = int(split_index)
        self.row_offset = int(row_offset)
        self.entries = self._make_entries()

    def __len__(self):
        return len(self.entries)

    def _to_dataset_index(self, absolute_index: int):
        absolute_to_relative = getattr(self.dataset, "_absolute_to_relative_idx", None)
        if absolute_to_relative is None:
            return absolute_index
        return absolute_to_relative.get(absolute_index)

    def _make_context(self, absolute_index: int, episode_start: int, episode_end: int):
        context_absolute_indices = []
        context_frame_indices = []
        context_is_pad = []
        for offset in self.frame_offsets:
            raw_index = absolute_index + offset
            clamped_index = max(episode_start, min(episode_end - 1, raw_index))
            context_absolute_indices.append(clamped_index)
            context_frame_indices.append(clamped_index - episode_start)
            context_is_pad.append(raw_index < episode_start or raw_index >= episode_end)
        return context_absolute_indices, context_frame_indices, context_is_pad

    def _make_entries(self):
        entries = []
        for episode_index in sorted(self.episodes):
            episode = self.dataset.meta.episodes[int(episode_index)]
            episode_start = int(episode["dataset_from_index"])
            episode_end = int(episode["dataset_to_index"])
            for absolute_index in range(episode_start, episode_end):
                dataset_index = self._to_dataset_index(absolute_index)
                if dataset_index is None:
                    continue
                (
                    context_absolute_indices,
                    context_frame_indices,
                    context_is_pad,
                ) = self._make_context(absolute_index, episode_start, episode_end)
                entries.append(
                    {
                        "dataset_index": int(dataset_index),
                        "episode_index": int(episode_index),
                        "frame_index": int(absolute_index - episode_start),
                        "absolute_index": int(absolute_index),
                        "context_absolute_indices": context_absolute_indices,
                        "context_frame_indices": context_frame_indices,
                        "context_is_pad": context_is_pad,
                    }
                )

        if not entries:
            raise ValueError("No LeRobot frames are available for slot feature caching.")
        return entries

    @staticmethod
    def _scalar_tensor(value, dtype):
        if torch.is_tensor(value):
            value = value.reshape(-1)[0]
            return value.to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        item = self.dataset[entry["dataset_index"]]
        video = item[self.camera_key]
        if video.ndim == 3:
            video = video.unsqueeze(0)

        timestamp = self._scalar_tensor(
            item.get("timestamp", float(entry["frame_index"])), torch.float64
        )
        task_index = self._scalar_tensor(item.get("task_index", -1), torch.long)

        return {
            "video": video,
            "cache_row": torch.tensor(self.row_offset + idx, dtype=torch.long),
            "dataset_index": torch.tensor(self.dataset_index, dtype=torch.long),
            "split_index": torch.tensor(self.split_index, dtype=torch.long),
            "episode_index": torch.tensor(entry["episode_index"], dtype=torch.long),
            "frame_index": torch.tensor(entry["frame_index"], dtype=torch.long),
            "absolute_index": torch.tensor(entry["absolute_index"], dtype=torch.long),
            "timestamp": timestamp,
            "task_index": task_index,
            "context_absolute_indices": torch.tensor(
                entry["context_absolute_indices"], dtype=torch.long
            ),
            "context_frame_indices": torch.tensor(entry["context_frame_indices"], dtype=torch.long),
            "context_is_pad": torch.tensor(entry["context_is_pad"], dtype=torch.bool),
        }


class LeRobotDataModule(pl.LightningDataModule):
    """LightningDataModule for LeRobot datasets used as VideoSAUR video inputs."""

    def __init__(
        self,
        repo_id: str,
        root: Optional[str] = None,
        data_dir: Optional[str] = None,
        camera_key: Optional[str] = None,
        frame_sampling: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        num_workers: int = 0,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        image_size: Union[int, Tuple[int, int], List[int]] = 224,
        video_backend: Optional[str] = "pyav",
        tolerance_s: float = 1e-3,
        lerobot_format: str = "auto",
        val_fraction: float = 0.0,
        test_fraction: float = 0.0,
        train_episodes: Optional[List[int]] = None,
        val_episodes: Optional[List[int]] = None,
        test_episodes: Optional[List[int]] = None,
        seed: int = 42,
        revision: Optional[str] = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        drop_last: bool = True,
        pin_memory: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = self._resolve_root(root, data_dir)
        self.camera_key = camera_key
        self.frame_sampling = (
            {key: frame_sampling[key] for key in frame_sampling}
            if frame_sampling is not None
            else {}
        )
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        self.num_train_workers = num_workers
        self.num_val_workers = num_val_workers if num_val_workers is not None else num_workers
        self.num_test_workers = num_test_workers if num_test_workers is not None else num_workers
        self.image_size = image_size
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self.lerobot_format = _normalize_lerobot_format(lerobot_format)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.train_episodes = _to_optional_list(train_episodes)
        self.val_episodes = _to_optional_list(val_episodes)
        self.test_episodes = _to_optional_list(test_episodes)
        self.seed = seed
        self.revision = revision
        self.force_cache_sync = force_cache_sync
        self.download_videos = download_videos
        self.drop_last = drop_last
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.persistent_workers = persistent_workers

        self.fps = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._train_episodes = None
        self._val_episodes = None
        self._test_episodes = None
        self._selected_episodes = None
        self._lerobot_dataset = None
        self._resolved_lerobot_format = None
        self.dataset_info = []

        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("`val_fraction` must be in [0.0, 1.0).")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("`test_fraction` must be in [0.0, 1.0).")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("`val_fraction + test_fraction` must be smaller than 1.0.")

    @staticmethod
    def _resolve_root(root: Optional[str], data_dir: Optional[str]) -> Optional[str]:
        if root is None:
            return None
        root = os.fspath(root)
        if data_dir is not None and not os.path.isabs(root):
            return os.path.join(data_dir, root)
        return root

    def __str__(self) -> str:
        train_samples = "not setup" if self.train_set is None else len(self.train_set)
        val_samples = "disabled" if self.val_set is None else len(self.val_set)
        test_samples = "disabled" if self.test_set is None else len(self.test_set)
        camera_key = "auto" if self.camera_key is None else self.camera_key
        return "\n".join(
            [
                "LeRobotDataModule",
                f"  - Repo id: {self.repo_id}",
                f"  - Root: {self.root}",
                f"  - LeRobot format: {self._get_lerobot_format()}",
                f"  - Camera key: {camera_key}",
                f"  - Train batch size: {self.batch_size}",
                f"  - Eval batch size: {self.val_batch_size}",
                f"  - Test batch size: {self.test_batch_size}",
                f"  - Number of train workers: {self.num_train_workers}",
                f"  - Number of eval workers: {self.num_val_workers}",
                f"  - Number of test workers: {self.num_test_workers}",
                f"  - Train samples: {train_samples}",
                f"  - Validation samples: {val_samples}",
                f"  - Test samples: {test_samples}",
            ]
        )

    def _get_lerobot_format(self) -> str:
        if self._resolved_lerobot_format is None:
            self._resolved_lerobot_format = _resolve_lerobot_format(
                self.lerobot_format,
                self.root,
                force_cache_sync=self.force_cache_sync,
            )
        return self._resolved_lerobot_format

    def _get_metadata(self):
        if self._get_lerobot_format() == "v2":
            if self.root is None:
                raise ValueError("LeRobot v2 datasets require a local `root`.")
            return LeRobotV2Metadata(repo_id=self.repo_id, root=self.root)

        metadata = _try_load_lerobot_info_metadata(self.root, self.force_cache_sync)
        if metadata is not None:
            return metadata

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError(
                "LeRobotDataModule requires the `lerobot` package. "
                "Install it or use a different dataset module."
            ) from exc

        return LeRobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=self.force_cache_sync,
        )

    def _get_frame_offsets(self) -> List[int]:
        num_frames = self.num_frames
        include_current = bool(self.frame_sampling.get("include_current", True))
        end_offset = 0 if include_current else -1
        return list(range(end_offset - num_frames + 1, end_offset + 1))

    def _get_delta_timestamps(self) -> Dict[str, List[float]]:
        return {self.camera_key: [offset / self.fps for offset in self._get_frame_offsets()]}

    @property
    def num_frames(self) -> int:
        return int(self.frame_sampling.get("num_frames", 4))

    @staticmethod
    def _validate_episodes(name: str, episodes, total_episodes: int):
        if episodes is None:
            return None

        normalized = []
        seen = set()
        duplicates = set()
        for episode in episodes:
            episode = int(episode)
            normalized.append(episode)
            if episode in seen:
                duplicates.add(episode)
            seen.add(episode)

        if duplicates:
            raise ValueError(f"Duplicate episode ids in `{name}`: {sorted(duplicates)[:10]}")

        invalid = [
            episode
            for episode in normalized
            if episode < 0 or episode >= total_episodes
        ]
        if invalid:
            raise ValueError(
                f"Episode ids in `{name}` must be in [0, {total_episodes}), "
                f"but got {invalid[:10]}."
            )

        return sorted(normalized)

    @staticmethod
    def _check_episode_overlap(splits: Dict[str, Optional[List[int]]]):
        split_sets = {
            name: set(episodes)
            for name, episodes in splits.items()
            if episodes is not None
        }
        split_names = list(split_sets)
        for idx, left_name in enumerate(split_names):
            for right_name in split_names[idx + 1 :]:
                overlap = sorted(split_sets[left_name] & split_sets[right_name])
                if overlap:
                    raise ValueError(
                        f"LeRobot episode splits `{left_name}` and `{right_name}` overlap: "
                        f"{overlap[:10]}"
                    )

    def _split_episodes(self, total_episodes: int):
        all_episodes = list(range(total_episodes))
        train_episodes = self._validate_episodes(
            "train_episodes", self.train_episodes, total_episodes
        )
        val_episodes = self._validate_episodes(
            "val_episodes", self.val_episodes, total_episodes
        )
        test_episodes = self._validate_episodes(
            "test_episodes", self.test_episodes, total_episodes
        )

        self._check_episode_overlap(
            {
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "test_episodes": test_episodes,
            }
        )

        if train_episodes is None:
            reserved_episodes = set()
            if val_episodes is not None:
                reserved_episodes.update(val_episodes)
            if test_episodes is not None:
                reserved_episodes.update(test_episodes)
            source_episodes = [
                episode for episode in all_episodes if episode not in reserved_episodes
            ]
        else:
            source_episodes = list(train_episodes)

        split_source = list(source_episodes)
        needs_random_split = (
            (val_episodes is None and self.val_fraction > 0.0)
            or (test_episodes is None and self.test_fraction > 0.0)
        )
        if needs_random_split:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(split_source)

            val_size = (
                0
                if val_episodes is not None or self.val_fraction == 0.0
                else max(1, int(round(len(split_source) * self.val_fraction)))
            )
            test_size = (
                0
                if test_episodes is not None or self.test_fraction == 0.0
                else max(1, int(round(len(split_source) * self.test_fraction)))
            )
            if val_size + test_size >= len(split_source):
                raise ValueError(
                    "LeRobot episode fractions leave no training episodes: "
                    f"source={len(split_source)}, val={val_size}, test={test_size}."
                )

            offset = 0
            if val_size:
                val_episodes = sorted(split_source[offset : offset + val_size])
                offset += val_size
            if test_size:
                test_episodes = sorted(split_source[offset : offset + test_size])
                offset += test_size
            train_episodes = sorted(split_source[offset:])
        else:
            train_episodes = sorted(split_source)

        if train_episodes is not None and len(train_episodes) == 0:
            raise ValueError("LeRobot training episode split is empty.")
        if val_episodes is not None and len(val_episodes) == 0:
            val_episodes = None
        if test_episodes is not None and len(test_episodes) == 0:
            test_episodes = None

        self._check_episode_overlap(
            {
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "test_episodes": test_episodes,
            }
        )

        return train_episodes, val_episodes, test_episodes

    def _get_lerobot_dataset(self, episodes=None):
        if self._lerobot_dataset is not None:
            return self._lerobot_dataset

        if self._get_lerobot_format() == "v2":
            self._lerobot_dataset = LeRobotV2Dataset(
                repo_id=self.repo_id,
                root=self.root,
                episodes=episodes,
                image_transforms=LeRobotImageTransform(size=self.image_size),
                delta_timestamps=self._get_delta_timestamps(),
                tolerance_s=self.tolerance_s,
                video_backend=self.video_backend,
            )
            return self._lerobot_dataset

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "LeRobotDataModule requires the `lerobot` package. "
                "Install it or use a different dataset module."
            ) from exc

        self._lerobot_dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            episodes=episodes,
            image_transforms=LeRobotImageTransform(size=self.image_size),
            delta_timestamps=self._get_delta_timestamps(),
            tolerance_s=self.tolerance_s,
            revision=self.revision,
            force_cache_sync=self.force_cache_sync,
            download_videos=self.download_videos,
            video_backend=self.video_backend,
        )
        return self._lerobot_dataset

    def setup(self, stage=None):
        metadata = self._get_metadata()
        self.fps = metadata.fps
        if self.camera_key is None:
            if not metadata.camera_keys:
                raise ValueError("LeRobot dataset does not define any camera keys.")
            self.camera_key = metadata.camera_keys[0]
        elif self.camera_key not in metadata.camera_keys:
            raise ValueError(
                f"Camera key `{self.camera_key}` not found. "
                f"Available camera keys: {metadata.camera_keys}"
            )

        train_episodes, val_episodes, test_episodes = self._split_episodes(
            metadata.total_episodes
        )
        self._train_episodes = train_episodes
        self._val_episodes = val_episodes
        self._test_episodes = test_episodes
        selected_episodes = sorted(
            set(train_episodes)
            | set([] if val_episodes is None else val_episodes)
            | set([] if test_episodes is None else test_episodes)
        )
        self._selected_episodes = _episodes_or_none_if_all(
            selected_episodes, metadata.total_episodes
        )

        lerobot_dataset = self._get_lerobot_dataset(self._selected_episodes)
        self.dataset_info = [
            {
                "repo_id": self.repo_id,
                "root": self.root,
                "camera_key": self.camera_key,
                "lerobot_format": self._get_lerobot_format(),
                "fps": self.fps,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "test_episodes": test_episodes,
            }
        ]

        if stage in (None, "fit") and self.train_set is None:
            self.train_set = LeRobotVideoDataset(
                lerobot_dataset,
                camera_key=self.camera_key,
                frame_offsets=self._get_frame_offsets(),
                episodes=train_episodes,
            )

        if stage in (None, "fit", "validate") and val_episodes is not None and self.val_set is None:
            self.val_set = LeRobotVideoDataset(
                lerobot_dataset,
                camera_key=self.camera_key,
                frame_offsets=self._get_frame_offsets(),
                episodes=val_episodes,
            )

        if stage in (None, "test") and test_episodes is not None and self.test_set is None:
            self.test_set = LeRobotVideoDataset(
                lerobot_dataset,
                camera_key=self.camera_key,
                frame_offsets=self._get_frame_offsets(),
                episodes=test_episodes,
            )

    def _get_dataloader(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        drop_last: bool,
        sampler=None,
    ):
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "drop_last": drop_last,
            "worker_init_fn": worker_init_function,
            "pin_memory": self.pin_memory,
        }
        if sampler is None:
            dataloader_kwargs["shuffle"] = shuffle
        else:
            if shuffle:
                raise ValueError("A DataLoader cannot use both `sampler` and `shuffle=True`.")
            dataloader_kwargs["sampler"] = sampler

        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = (
                True if self.persistent_workers is None else self.persistent_workers
            )
            dataloader_kwargs["prefetch_factor"] = 2

        return torch.utils.data.DataLoader(**dataloader_kwargs)

    def _get_cache_dataset(
        self,
        lerobot_dataset,
        split_index: int,
        episodes: Optional[List[int]],
        row_offset: int,
    ):
        if episodes is None:
            return None
        return LeRobotSlotCacheDataset(
            lerobot_dataset,
            camera_key=self.camera_key,
            frame_offsets=self._get_frame_offsets(),
            episodes=episodes,
            dataset_index=0,
            split_index=split_index,
            row_offset=row_offset,
        )

    def cache_datasets(self, splits: Optional[List[str]] = None):
        if splits is None:
            splits = ["train", "val", "test"]
        self.setup("predict")
        split_episodes = {
            "train": self._train_episodes,
            "val": self._val_episodes,
            "test": self._test_episodes,
        }
        lerobot_dataset = self._get_lerobot_dataset(self._selected_episodes)

        datasets = []
        row_offset = 0
        for split_index, split_name in enumerate(splits):
            if split_name not in split_episodes:
                raise ValueError(f"Unknown LeRobot split `{split_name}`.")
            dataset = self._get_cache_dataset(
                lerobot_dataset,
                split_index=split_index,
                episodes=split_episodes[split_name],
                row_offset=row_offset,
            )
            if dataset is None:
                continue
            datasets.append(dataset)
            row_offset += len(dataset)
        if not datasets:
            raise ValueError("No LeRobot cache datasets were built for the requested splits.")
        return datasets

    def cache_dataloaders(
        self,
        splits: Optional[List[str]] = None,
        *,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        requested_splits = ["train", "val", "test"] if splits is None else list(splits)
        datasets = self.cache_datasets(splits)
        loaders = []
        for dataset in datasets:
            split_name = requested_splits[dataset.split_index]
            if split_name == "train":
                batch_size = self.batch_size
                num_workers = self.num_train_workers
            elif split_name == "val":
                batch_size = self.val_batch_size
                num_workers = self.num_val_workers
            else:
                batch_size = self.test_batch_size
                num_workers = self.num_test_workers
            sampler = (
                SequentialShardSampler(dataset, rank=rank, world_size=world_size)
                if distributed and world_size > 1
                else None
            )
            loaders.append(
                self._get_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False,
                    sampler=sampler,
                )
            )
        return loaders

    def predict_dataloader(self):
        return self.cache_dataloaders()

    def train_dataloader(self):
        if self.train_set is None:
            self.setup("fit")
        return self._get_dataloader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_train_workers,
            shuffle=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        if self.val_set is None:
            self.setup("validate")
        if self.val_set is None:
            return None
        return self._get_dataloader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_set is None:
            self.setup("test")
        if self.test_set is None:
            return None
        return self._get_dataloader(
            self.test_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_test_workers,
            shuffle=False,
            drop_last=False,
        )


class MixedLeRobotDataModule(pl.LightningDataModule):
    """LightningDataModule mixing multiple LeRobot datasets without physically merging them."""

    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        root: Optional[str] = None,
        data_dir: Optional[str] = None,
        frame_sampling: Optional[Dict[str, Any]] = None,
        sampling: Optional[Dict[str, Any]] = None,
        return_dataset_index: bool = True,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        num_workers: int = 0,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        image_size: Union[int, Tuple[int, int], List[int]] = 224,
        video_backend: Optional[str] = "pyav",
        tolerance_s: float = 1e-3,
        lerobot_format: str = "auto",
        val_fraction: float = 0.0,
        test_fraction: float = 0.0,
        train_episodes: Optional[List[int]] = None,
        val_episodes: Optional[List[int]] = None,
        test_episodes: Optional[List[int]] = None,
        seed: int = 42,
        revision: Optional[str] = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        drop_last: bool = True,
        pin_memory: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
    ):
        super().__init__()
        if not datasets:
            raise ValueError("`datasets` must contain at least one LeRobot dataset config.")

        self.root = LeRobotDataModule._resolve_root(root, data_dir)
        self.lerobot_format = _normalize_lerobot_format(lerobot_format)
        self.dataset_configs = [
            self._normalize_dataset_config(dataset_config, data_dir)
            for dataset_config in datasets
        ]
        self.frame_sampling = _to_plain_dict(frame_sampling)
        self.sampling = _to_plain_dict(sampling)
        self.return_dataset_index = bool(return_dataset_index)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        self.num_train_workers = num_workers
        self.num_val_workers = num_val_workers if num_val_workers is not None else num_workers
        self.num_test_workers = num_test_workers if num_test_workers is not None else num_workers
        self.image_size = image_size
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.train_episodes = _to_optional_list(train_episodes)
        self.val_episodes = _to_optional_list(val_episodes)
        self.test_episodes = _to_optional_list(test_episodes)
        self.seed = seed
        self.revision = revision
        self.force_cache_sync = force_cache_sync
        self.download_videos = download_videos
        self.drop_last = drop_last
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.persistent_workers = persistent_workers

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.dataset_info = []
        self._cache_dataset_entries = []
        self._is_setup = False

        self._validate_split_fractions(self.val_fraction, self.test_fraction)
        sampling_mode = self.sampling.get("mode", "natural")
        if sampling_mode not in ("natural", "weighted"):
            raise ValueError("`sampling.mode` must be one of {'natural', 'weighted'}.")
        weights = self.sampling.get("weights")
        if weights is not None and len(weights) != len(self.dataset_configs):
            raise ValueError(
                "`sampling.weights` must contain one weight per configured dataset: "
                f"got {len(weights)} weights for {len(self.dataset_configs)} datasets."
            )
        num_samples = self.sampling.get("num_samples")
        if num_samples is not None and int(num_samples) <= 0:
            raise ValueError("`sampling.num_samples` must be positive when specified.")

    def _normalize_dataset_config(self, dataset_config, data_dir: Optional[str]) -> Dict[str, Any]:
        dataset_config = _to_plain_dict(dataset_config)
        if "repo_id" not in dataset_config:
            raise ValueError("Each mixed LeRobot dataset config must define `repo_id`.")

        dataset_config["repo_id"] = str(dataset_config["repo_id"])
        dataset_config["root"] = LeRobotDataModule._resolve_root(
            dataset_config.get("root"), data_dir
        )
        if dataset_config["root"] is None and self.root is not None:
            dataset_config["root"] = os.path.join(self.root, dataset_config["repo_id"])
        dataset_config["lerobot_format"] = _normalize_lerobot_format(
            dataset_config.get("lerobot_format", self.lerobot_format)
        )
        return dataset_config

    @staticmethod
    def _validate_split_fractions(val_fraction: float, test_fraction: float):
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("`val_fraction` must be in [0.0, 1.0).")
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError("`test_fraction` must be in [0.0, 1.0).")
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("`val_fraction + test_fraction` must be smaller than 1.0.")

    def __str__(self) -> str:
        train_samples = "not setup" if self.train_set is None else len(self.train_set)
        val_samples = "disabled" if self.val_set is None else len(self.val_set)
        test_samples = "disabled" if self.test_set is None else len(self.test_set)
        lines = [
            "MixedLeRobotDataModule",
            f"  - Number of datasets: {len(self.dataset_configs)}",
            f"  - Sampling mode: {self.sampling.get('mode', 'natural')}",
            f"  - Train batch size: {self.batch_size}",
            f"  - Eval batch size: {self.val_batch_size}",
            f"  - Test batch size: {self.test_batch_size}",
            f"  - Number of train workers: {self.num_train_workers}",
            f"  - Number of eval workers: {self.num_val_workers}",
            f"  - Number of test workers: {self.num_test_workers}",
            f"  - Train samples: {train_samples}",
            f"  - Validation samples: {val_samples}",
            f"  - Test samples: {test_samples}",
        ]
        for idx, dataset_config in enumerate(self.dataset_configs):
            camera_key = dataset_config.get("camera_key", "auto")
            lines.append(
                f"  - Dataset {idx}: repo_id={dataset_config['repo_id']}, "
                f"root={dataset_config.get('root')}, "
                f"format={dataset_config.get('_resolved_lerobot_format', dataset_config['lerobot_format'])}, "
                f"camera_key={camera_key}"
            )
        return "\n".join(lines)

    @property
    def num_frames(self) -> int:
        return int(self.frame_sampling.get("num_frames", 4))

    def _get_frame_offsets(self) -> List[int]:
        include_current = bool(self.frame_sampling.get("include_current", True))
        end_offset = 0 if include_current else -1
        return list(range(end_offset - self.num_frames + 1, end_offset + 1))

    def _get_delta_timestamps(self, camera_key: str, fps: float) -> Dict[str, List[float]]:
        if fps is None or fps <= 0:
            raise ValueError(f"Invalid LeRobot fps for `{camera_key}`: {fps}")
        return {camera_key: [offset / fps for offset in self._get_frame_offsets()]}

    def _get_lerobot_format(self, dataset_config: Dict[str, Any]) -> str:
        resolved_format = dataset_config.get("_resolved_lerobot_format")
        if resolved_format is None:
            resolved_format = _resolve_lerobot_format(
                dataset_config.get("lerobot_format", self.lerobot_format),
                dataset_config.get("root"),
                force_cache_sync=dataset_config.get("force_cache_sync", self.force_cache_sync),
            )
            dataset_config["_resolved_lerobot_format"] = resolved_format
        return resolved_format

    def _get_metadata(self, dataset_config: Dict[str, Any]):
        dataset_format = self._get_lerobot_format(dataset_config)
        if dataset_format == "v2":
            root = dataset_config.get("root")
            if root is None:
                raise ValueError(
                    f"LeRobot v2 dataset `{dataset_config['repo_id']}` requires a local `root`."
                )
            return LeRobotV2Metadata(repo_id=dataset_config["repo_id"], root=root)

        metadata = _try_load_lerobot_info_metadata(
            dataset_config.get("root"),
            dataset_config.get("force_cache_sync", self.force_cache_sync),
        )
        if metadata is not None:
            return metadata

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError(
                "MixedLeRobotDataModule requires the `lerobot` package. "
                "Install it or use a different dataset module."
            ) from exc

        return LeRobotDatasetMetadata(
            repo_id=dataset_config["repo_id"],
            root=dataset_config.get("root"),
            revision=dataset_config.get("revision", self.revision),
            force_cache_sync=dataset_config.get("force_cache_sync", self.force_cache_sync),
        )

    def _get_lerobot_dataset(
        self,
        dataset_config: Dict[str, Any],
        delta_timestamps: Dict[str, List[float]],
        episodes: Optional[List[int]],
    ):
        if self._get_lerobot_format(dataset_config) == "v2":
            return LeRobotV2Dataset(
                repo_id=dataset_config["repo_id"],
                root=dataset_config.get("root"),
                episodes=episodes,
                image_transforms=LeRobotImageTransform(size=self.image_size),
                delta_timestamps=delta_timestamps,
                tolerance_s=dataset_config.get("tolerance_s", self.tolerance_s),
                video_backend=dataset_config.get("video_backend", self.video_backend),
            )

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "MixedLeRobotDataModule requires the `lerobot` package. "
                "Install it or use a different dataset module."
            ) from exc

        return LeRobotDataset(
            repo_id=dataset_config["repo_id"],
            root=dataset_config.get("root"),
            episodes=episodes,
            image_transforms=LeRobotImageTransform(size=self.image_size),
            delta_timestamps=delta_timestamps,
            tolerance_s=dataset_config.get("tolerance_s", self.tolerance_s),
            revision=dataset_config.get("revision", self.revision),
            force_cache_sync=dataset_config.get("force_cache_sync", self.force_cache_sync),
            download_videos=dataset_config.get("download_videos", self.download_videos),
            video_backend=dataset_config.get("video_backend", self.video_backend),
        )

    def _split_episodes(self, dataset_config: Dict[str, Any], total_episodes: int):
        repo_id = dataset_config["repo_id"]
        train_episodes = LeRobotDataModule._validate_episodes(
            f"{repo_id}.train_episodes",
            dataset_config.get("train_episodes", self.train_episodes),
            total_episodes,
        )
        val_episodes = LeRobotDataModule._validate_episodes(
            f"{repo_id}.val_episodes",
            dataset_config.get("val_episodes", self.val_episodes),
            total_episodes,
        )
        test_episodes = LeRobotDataModule._validate_episodes(
            f"{repo_id}.test_episodes",
            dataset_config.get("test_episodes", self.test_episodes),
            total_episodes,
        )
        val_fraction = float(dataset_config.get("val_fraction", self.val_fraction))
        test_fraction = float(dataset_config.get("test_fraction", self.test_fraction))
        seed = int(dataset_config.get("seed", self.seed))
        self._validate_split_fractions(val_fraction, test_fraction)

        LeRobotDataModule._check_episode_overlap(
            {
                f"{repo_id}.train_episodes": train_episodes,
                f"{repo_id}.val_episodes": val_episodes,
                f"{repo_id}.test_episodes": test_episodes,
            }
        )

        all_episodes = list(range(total_episodes))
        if train_episodes is None:
            reserved_episodes = set()
            if val_episodes is not None:
                reserved_episodes.update(val_episodes)
            if test_episodes is not None:
                reserved_episodes.update(test_episodes)
            source_episodes = [
                episode for episode in all_episodes if episode not in reserved_episodes
            ]
        else:
            source_episodes = list(train_episodes)

        split_source = list(source_episodes)
        needs_random_split = (
            (val_episodes is None and val_fraction > 0.0)
            or (test_episodes is None and test_fraction > 0.0)
        )
        if needs_random_split:
            rng = np.random.RandomState(seed)
            rng.shuffle(split_source)

            val_size = (
                0
                if val_episodes is not None or val_fraction == 0.0
                else max(1, int(round(len(split_source) * val_fraction)))
            )
            test_size = (
                0
                if test_episodes is not None or test_fraction == 0.0
                else max(1, int(round(len(split_source) * test_fraction)))
            )
            if val_size + test_size >= len(split_source):
                raise ValueError(
                    f"LeRobot episode fractions leave no training episodes for `{repo_id}`: "
                    f"source={len(split_source)}, val={val_size}, test={test_size}."
                )

            offset = 0
            if val_size:
                val_episodes = sorted(split_source[offset : offset + val_size])
                offset += val_size
            if test_size:
                test_episodes = sorted(split_source[offset : offset + test_size])
                offset += test_size
            train_episodes = sorted(split_source[offset:])
        else:
            train_episodes = sorted(split_source)

        if train_episodes is not None and len(train_episodes) == 0:
            raise ValueError(f"LeRobot training episode split is empty for `{repo_id}`.")
        if val_episodes is not None and len(val_episodes) == 0:
            val_episodes = None
        if test_episodes is not None and len(test_episodes) == 0:
            test_episodes = None

        LeRobotDataModule._check_episode_overlap(
            {
                f"{repo_id}.train_episodes": train_episodes,
                f"{repo_id}.val_episodes": val_episodes,
                f"{repo_id}.test_episodes": test_episodes,
            }
        )

        return train_episodes, val_episodes, test_episodes

    def _get_camera_key(self, metadata, dataset_config: Dict[str, Any]) -> str:
        camera_key = dataset_config.get("camera_key")
        if camera_key is None:
            if not metadata.camera_keys:
                raise ValueError(
                    f"LeRobot dataset `{dataset_config['repo_id']}` does not define any "
                    "camera keys."
                )
            camera_key = metadata.camera_keys[0]
        elif camera_key not in metadata.camera_keys:
            raise ValueError(
                f"Camera key `{camera_key}` not found for `{dataset_config['repo_id']}`. "
                f"Available camera keys: {metadata.camera_keys}"
            )
        return camera_key

    def _make_video_dataset(
        self,
        dataset,
        camera_key: str,
        episodes: List[int],
        dataset_index: int,
    ):
        return LeRobotVideoDataset(
            dataset,
            camera_key=camera_key,
            frame_offsets=self._get_frame_offsets(),
            episodes=episodes,
            dataset_index=dataset_index if self.return_dataset_index else None,
        )

    @staticmethod
    def _concat_or_none(datasets):
        if not datasets:
            return None
        return torch.utils.data.ConcatDataset(datasets)

    def setup(self, stage=None):
        if self._is_setup:
            return

        build_standard_sets = stage != "predict"
        train_sets = []
        val_sets = []
        test_sets = []
        self.dataset_info = []
        self._cache_dataset_entries = []
        for dataset_index, dataset_config in enumerate(self.dataset_configs):
            metadata = self._get_metadata(dataset_config)
            fps = metadata.fps
            camera_key = self._get_camera_key(metadata, dataset_config)
            train_episodes, val_episodes, test_episodes = self._split_episodes(
                dataset_config, metadata.total_episodes
            )
            delta_timestamps = self._get_delta_timestamps(camera_key, fps)
            selected_episodes = sorted(
                set(train_episodes)
                | set([] if val_episodes is None else val_episodes)
                | set([] if test_episodes is None else test_episodes)
            )
            dataset_episodes = _episodes_or_none_if_all(
                selected_episodes, metadata.total_episodes
            )
            lerobot_dataset = self._get_lerobot_dataset(
                dataset_config, delta_timestamps, dataset_episodes
            )

            self.dataset_info.append(
                {
                    "repo_id": dataset_config["repo_id"],
                    "root": dataset_config.get("root"),
                    "camera_key": camera_key,
                    "lerobot_format": self._get_lerobot_format(dataset_config),
                    "fps": fps,
                    "train_episodes": train_episodes,
                    "val_episodes": val_episodes,
                    "test_episodes": test_episodes,
                }
            )
            self._cache_dataset_entries.append(
                {
                    "dataset": lerobot_dataset,
                    "camera_key": camera_key,
                    "dataset_index": dataset_index,
                    "train_episodes": train_episodes,
                    "val_episodes": val_episodes,
                    "test_episodes": test_episodes,
                }
            )

            if build_standard_sets:
                train_sets.append(
                    self._make_video_dataset(
                        lerobot_dataset, camera_key, train_episodes, dataset_index
                    )
                )
                if val_episodes is not None:
                    val_sets.append(
                        self._make_video_dataset(
                            lerobot_dataset, camera_key, val_episodes, dataset_index
                        )
                    )
                if test_episodes is not None:
                    test_sets.append(
                        self._make_video_dataset(
                            lerobot_dataset, camera_key, test_episodes, dataset_index
                        )
                    )

        if build_standard_sets:
            self.train_set = self._concat_or_none(train_sets)
            self.val_set = self._concat_or_none(val_sets)
            self.test_set = self._concat_or_none(test_sets)
            if self.train_set is None:
                raise ValueError("Mixed LeRobot training split is empty.")
            self._is_setup = True

    def _get_train_sampler(self):
        if self.sampling.get("mode", "natural") == "natural":
            return None

        datasets = self.train_set.datasets
        weights = self.sampling.get("weights")
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        else:
            weights = [float(weight) for weight in weights]

        sample_weights = []
        for dataset, dataset_weight in zip(datasets, weights):
            dataset_len = len(dataset)
            if dataset_len <= 0:
                raise ValueError("Cannot sample from an empty mixed LeRobot child dataset.")
            sample_weights.extend([dataset_weight / dataset_len] * dataset_len)

        num_samples = int(self.sampling.get("num_samples", len(sample_weights)))
        generator = torch.Generator()
        generator.manual_seed(int(self.sampling.get("seed", self.seed)))
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )

    def _get_dataloader(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        drop_last: bool,
        sampler=None,
    ):
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "drop_last": drop_last,
            "worker_init_fn": worker_init_function,
            "pin_memory": self.pin_memory,
        }
        if sampler is None:
            dataloader_kwargs["shuffle"] = shuffle
        else:
            if shuffle:
                raise ValueError("A DataLoader cannot use both `sampler` and `shuffle=True`.")
            dataloader_kwargs["sampler"] = sampler

        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = (
                True if self.persistent_workers is None else self.persistent_workers
            )
            dataloader_kwargs["prefetch_factor"] = 2

        return torch.utils.data.DataLoader(**dataloader_kwargs)

    def cache_datasets(self, splits: Optional[List[str]] = None):
        if splits is None:
            splits = ["train", "val", "test"]
        self.setup("predict")

        datasets = []
        row_offset = 0
        for split_index, split_name in enumerate(splits):
            if split_name not in ("train", "val", "test"):
                raise ValueError(f"Unknown LeRobot split `{split_name}`.")

            split_datasets = []
            for entry in self._cache_dataset_entries:
                episodes = entry[f"{split_name}_episodes"]
                if episodes is None:
                    continue
                dataset = LeRobotSlotCacheDataset(
                    entry["dataset"],
                    camera_key=entry["camera_key"],
                    frame_offsets=self._get_frame_offsets(),
                    episodes=episodes,
                    dataset_index=entry["dataset_index"],
                    split_index=split_index,
                    row_offset=row_offset,
                )
                row_offset += len(dataset)
                split_datasets.append(dataset)

            split_dataset = self._concat_or_none(split_datasets)
            if split_dataset is not None:
                datasets.append(split_dataset)

        if not datasets:
            raise ValueError("No mixed LeRobot cache datasets were built for the requested splits.")
        return datasets

    @staticmethod
    def _cache_split_index(dataset) -> int:
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            return dataset.datasets[0].split_index
        return dataset.split_index

    def cache_dataloaders(
        self,
        splits: Optional[List[str]] = None,
        *,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        requested_splits = ["train", "val", "test"] if splits is None else list(splits)
        datasets = self.cache_datasets(splits)
        loaders = []
        for dataset in datasets:
            split_name = requested_splits[self._cache_split_index(dataset)]
            if split_name == "train":
                batch_size = self.batch_size
                num_workers = self.num_train_workers
            elif split_name == "val":
                batch_size = self.val_batch_size
                num_workers = self.num_val_workers
            else:
                batch_size = self.test_batch_size
                num_workers = self.num_test_workers
            sampler = (
                SequentialShardSampler(dataset, rank=rank, world_size=world_size)
                if distributed and world_size > 1
                else None
            )
            loaders.append(
                self._get_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False,
                    sampler=sampler,
                )
            )
        return loaders

    def predict_dataloader(self):
        return self.cache_dataloaders()

    def train_dataloader(self):
        if self.train_set is None:
            self.setup("fit")
        sampler = self._get_train_sampler()
        return self._get_dataloader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_train_workers,
            shuffle=sampler is None,
            drop_last=self.drop_last,
            sampler=sampler,
        )

    def val_dataloader(self):
        if self.val_set is None:
            self.setup("validate")
        if self.val_set is None:
            return None
        return self._get_dataloader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_set is None:
            self.setup("test")
        if self.test_set is None:
            return None
        return self._get_dataloader(
            self.test_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_test_workers,
            shuffle=False,
            drop_last=False,
        )


class DummyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_size: int,
        val_size: int,
        batch_size: int,
        shapes: Dict[str, Tuple[int]],
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.shapes = shapes
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def __str__(self) -> str:
        res = ["DummyDataModule"]
        res.append(f"  - Number of train samples: {self.train_size}")
        res.append(f"  - Number of val samples: {self.val_size}")
        res.append(f"  - Number of test samples: {self.test_size}")
        res.append(f"  - Batch size: {self.batch_size}")
        return "\n".join(res)

    @staticmethod
    def _make_random_dataset(shapes, size: int, seed: int):
        rng = np.random.RandomState(seed)
        dataset = []
        for idx in range(size):
            data = {"__key__": str(idx)}
            for name, shape in shapes.items():
                data[name] = rng.randint(0, 255, size=shape, dtype=np.uint8)
            dataset.append(data)

        return dataset

    @staticmethod
    def _make_squares_dataset(shapes, size: int, seed: int, n_objects: int = 3):
        rng = np.random.RandomState(seed)
        dataset = []
        for idx in range(size):
            data = {"__key__": str(idx)}
            for name, shape in shapes.items():
                height, width = shape[-3:-1]
                # Place random black squares on white background
                if "mask" in name:
                    array = np.zeros(shape, dtype=np.uint8)
                else:
                    array = np.ones(shape, dtype=np.uint8) * 255
                for idx in range(n_objects):
                    x = rng.randint(0, width)
                    y = rng.randint(0, height)
                    size = rng.randint(0, int(0.3 * (height + width) / 2))
                    if "mask" in name:
                        array[..., y : y + size, x : x + size, :] = idx + 1
                    else:
                        array[..., y : y + size, x : x + size, :] = 0

                data[name] = array
            dataset.append(data)

        return dataset

    def setup(self, stage):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, data, transforms):
                super().__init__()
                self.data = data
                self.transforms = transforms

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                data = {**self.data[idx]}  # Copy dict
                if self.transforms:
                    for name, transform in self.transforms.items():
                        data[name] = transform(data[name])

                return data

        train_data = self._make_squares_dataset(self.shapes, self.train_size, 42)
        self.train_set = Dataset(train_data, self.train_transforms)
        val_data = self._make_squares_dataset(self.shapes, self.val_size, 42)
        self.val_set = Dataset(val_data, self.val_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
