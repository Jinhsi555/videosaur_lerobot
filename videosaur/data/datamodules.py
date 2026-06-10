import math
import os
import tempfile
from functools import partial
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


def _to_size_tuple(value: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple, ListConfig)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected int or pair for image size, but got {value}")


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


class LeRobotVideoDataset(torch.utils.data.Dataset):
    """Adapts a LeRobotDataset sample to VideoSAUR's `video` input key."""

    def __init__(
        self,
        dataset,
        camera_key: str,
        frame_sampling: Dict[str, Any],
        random_sampling: bool,
    ):
        super().__init__()
        self.dataset = dataset
        self.camera_key = camera_key
        self.mode = frame_sampling.get("mode", "random_frames")
        self.num_frames = int(frame_sampling.get("num_frames", 4))
        self.random_sampling = random_sampling

        if self.mode not in ("random_frames", "contiguous_clip"):
            raise ValueError(
                f"Unsupported LeRobot frame sampling mode `{self.mode}`. "
                "Supported modes are `random_frames` and `contiguous_clip`."
            )
        if self.num_frames <= 0:
            raise ValueError("`frame_sampling.num_frames` must be positive.")

    def __len__(self):
        return len(self.dataset)

    def _sample_indices(self, num_candidates: int) -> torch.Tensor:
        if num_candidates < self.num_frames:
            raise ValueError(
                f"Need at least {self.num_frames} candidate frames, but got {num_candidates}."
            )

        if self.mode == "contiguous_clip":
            max_start = num_candidates - self.num_frames
            if self.random_sampling and max_start > 0:
                start = torch.randint(max_start + 1, (1,)).item()
            else:
                start = max_start // 2
            return torch.arange(start, start + self.num_frames)

        if self.random_sampling:
            indices = torch.randperm(num_candidates)[: self.num_frames]
            return indices.sort().values

        if self.num_frames == 1:
            return torch.tensor([num_candidates - 1], dtype=torch.long)
        return torch.linspace(0, num_candidates - 1, self.num_frames).round().long()

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video = item[self.camera_key]
        if video.ndim == 3:
            video = video.unsqueeze(0)

        indices = self._sample_indices(video.shape[0])
        return {"video": video[indices]}


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

    def _get_metadata(self):
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

    def _get_delta_timestamps(self) -> Dict[str, List[float]]:
        window_before = int(self.frame_sampling.get("window_before", self.num_frames - 1))
        include_current = bool(self.frame_sampling.get("include_current", True))
        end_offset = 0 if include_current else -1
        frame_offsets = list(range(-window_before, end_offset + 1))

        if len(frame_offsets) < self.num_frames:
            raise ValueError(
                "Frame sampling window is too small: "
                f"got {len(frame_offsets)} candidates for {self.num_frames} requested frames."
            )

        return {self.camera_key: [offset / self.fps for offset in frame_offsets]}

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

    def _make_lerobot_dataset(self, episodes):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "LeRobotDataModule requires the `lerobot` package. "
                "Install it or use a different dataset module."
            ) from exc

        return LeRobotDataset(
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

        if stage in (None, "fit") and self.train_set is None:
            train_dataset = self._make_lerobot_dataset(train_episodes)
            self.train_set = LeRobotVideoDataset(
                train_dataset,
                camera_key=self.camera_key,
                frame_sampling=self.frame_sampling,
                random_sampling=True,
            )

        if stage in (None, "fit", "validate") and val_episodes is not None and self.val_set is None:
            val_dataset = self._make_lerobot_dataset(val_episodes)
            self.val_set = LeRobotVideoDataset(
                val_dataset,
                camera_key=self.camera_key,
                frame_sampling=self.frame_sampling,
                random_sampling=False,
            )

        if stage in (None, "test") and test_episodes is not None and self.test_set is None:
            test_dataset = self._make_lerobot_dataset(test_episodes)
            self.test_set = LeRobotVideoDataset(
                test_dataset,
                camera_key=self.camera_key,
                frame_sampling=self.frame_sampling,
                random_sampling=False,
            )

    def _get_dataloader(
        self, dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool
    ):
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": drop_last,
            "worker_init_fn": worker_init_function,
            "pin_memory": self.pin_memory,
        }

        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = (
                True if self.persistent_workers is None else self.persistent_workers
            )
            dataloader_kwargs["prefetch_factor"] = 2

        return torch.utils.data.DataLoader(**dataloader_kwargs)

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
