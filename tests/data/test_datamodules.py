import json
import itertools
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from videosaur.data import datamodules


@pytest.fixture(scope="session")
def dummy_shards(tmp_path_factory):
    import webdataset as wds

    data_dir = tmp_path_factory.mktemp("data")

    def write_split(split, n_samples, n_samples_per_shard):
        pattern = str(data_dir / f"{split}-%06d.tar")
        with wds.ShardWriter(pattern, maxcount=n_samples_per_shard) as sink:
            sink.verbose = 0
            for idx in range(n_samples):
                sink.write({"__key__": str(idx), "tensor.pth": torch.randn(2, 2)})

        return sorted([str(p) for p in data_dir.glob(f"{split}-*.tar")])

    train_size, val_size = 10, 7
    train_shards = write_split("train", train_size, 4)
    val_shards = write_split("validation", val_size, 4)
    return {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "train_size": train_size,
        "val_size": val_size,
        "tensor_key": "tensor",
    }


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_webdataset_datamodule(dummy_shards, num_workers):
    batch_size, val_batch_size = 3, 2
    datamodule = datamodules.WebdatasetDataModule(
        train_shards=dummy_shards["train_shards"],
        val_shards=dummy_shards["val_shards"],
        val_size=dummy_shards["val_size"],
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        num_val_workers=num_workers,
    )

    # Check that training iterator contains an infinite stream of data, i.e. that dataloader does
    # not stop prematurely
    expected_n_batches = (dummy_shards["train_size"] // batch_size) + 3
    n_batches = 0
    for batch in itertools.islice(datamodule.train_dataloader(), expected_n_batches):
        assert batch[dummy_shards["tensor_key"]].shape[0] == batch_size
        n_batches += 1

    assert n_batches == expected_n_batches

    # Check that validation iterator contains exactly the validation data, modulo padding
    max_batches = dummy_shards["val_size"]  # Set upper bound to avoid infinite iteration
    keys = []
    for batch in itertools.islice(datamodule.val_dataloader(), max_batches):
        assert batch[dummy_shards["tensor_key"]].shape[0] == val_batch_size
        keys.extend([key for key in batch["__key__"] if key != "PADDING"])

    assert len(keys) == dummy_shards["val_size"]
    if num_workers <= 1:
        # For one worker, we expect the samples to be iterated in order
        assert sorted(list(set(keys))) == keys
    else:
        # For more than one worker, the samples are interleaved, so we need to sort to compare
        assert sorted(list(set(keys))) == sorted(keys)


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_webdataset_datamodule_fixed_epoch_len(dummy_shards, num_workers):
    batch_size = 3
    samples_per_epoch = dummy_shards["train_size"]
    datamodule = datamodules.WebdatasetDataModule(
        train_shards=dummy_shards["train_shards"],
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Check that training iterator stops after having sampled roughly `samples_per_epoch` samples
    max_batches = dummy_shards["train_size"]  # Set upper bound to avoid infinite iteration
    n_batches = 0
    for batch in itertools.islice(datamodule.train_dataloader(), max_batches):
        assert batch[dummy_shards["tensor_key"]].shape[0] == batch_size
        n_batches += 1

    expected_n_batches = samples_per_epoch // batch_size
    assert n_batches == expected_n_batches


class _FakeLeRobotMetadata:
    def __init__(self, fps, camera_key, total_episodes=4):
        self.fps = fps
        self.camera_keys = [camera_key]
        self.total_episodes = total_episodes


class _FakeLeRobotDataset:
    def __init__(
        self,
        camera_key,
        delta_timestamps,
        episodes,
        total_episodes=4,
        frames_per_episode=3,
    ):
        self.camera_key = camera_key
        self.delta_timestamps = delta_timestamps
        if episodes is None:
            episodes = list(range(total_episodes))
        self.episodes = episodes
        absolute_indices = [
            absolute_index
            for episode_idx in episodes
            for absolute_index in range(
                episode_idx * frames_per_episode, (episode_idx + 1) * frames_per_episode
            )
        ]
        self.absolute_indices = absolute_indices
        self.frames_per_episode = frames_per_episode
        self._absolute_to_relative_idx = {
            absolute_index: relative_index
            for relative_index, absolute_index in enumerate(absolute_indices)
        }
        self.meta = SimpleNamespace(
            episodes=[
                {
                    "dataset_from_index": episode_idx * frames_per_episode,
                    "dataset_to_index": (episode_idx + 1) * frames_per_episode,
                }
                for episode_idx in range(total_episodes)
            ]
        )

    def __getitem__(self, idx):
        num_frames = len(self.delta_timestamps[self.camera_key])
        absolute_index = self.absolute_indices[idx]
        episode_index = absolute_index // self.frames_per_episode
        frame_index = absolute_index % self.frames_per_episode
        return {
            self.camera_key: torch.full(
                (num_frames, 3, 2, 2), fill_value=float(idx), dtype=torch.float32
            ),
            "index": torch.tensor(absolute_index, dtype=torch.long),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
            "timestamp": torch.tensor(float(frame_index) / 10.0, dtype=torch.float32),
            "task_index": torch.tensor(episode_index % 2, dtype=torch.long),
        }


class _FakeMixedLeRobotDataModule(datamodules.MixedLeRobotDataModule):
    FAKE_DATASETS = {
        "ego": {"fps": 10.0, "camera_key": "observation.images.top_head"},
        "ssv2": {"fps": 12.0, "camera_key": "observation.images.egocentric"},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_datasets = []

    def _get_metadata(self, dataset_config):
        dataset = self.FAKE_DATASETS[dataset_config["repo_id"]]
        return _FakeLeRobotMetadata(
            fps=dataset["fps"],
            camera_key=dataset["camera_key"],
        )

    def _get_lerobot_dataset(self, dataset_config, delta_timestamps, episodes):
        camera_key = next(iter(delta_timestamps))
        self.created_datasets.append(
            {
                "repo_id": dataset_config["repo_id"],
                "root": dataset_config.get("root"),
                "camera_key": camera_key,
                "delta_timestamps": delta_timestamps,
                "episodes": episodes,
            }
        )
        return _FakeLeRobotDataset(camera_key, delta_timestamps, episodes)


def _make_mixed_lerobot_datamodule(**kwargs):
    config = {
        "datasets": [
            {
                "repo_id": "ego",
                "root": "/data/ego",
                "camera_key": "observation.images.top_head",
            },
            {
                "repo_id": "ssv2",
                "root": "/data/ssv2",
                "camera_key": "observation.images.egocentric",
            },
        ],
        "frame_sampling": {"num_frames": 2},
        "batch_size": 2,
        "num_workers": 0,
        "val_fraction": 0.25,
        "test_fraction": 0.25,
    }
    config.update(kwargs)
    return _FakeMixedLeRobotDataModule(**config)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_fake_lerobot_v2(root, camera_key="observation.images.image"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir()
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000" / camera_key).mkdir(parents=True)

    episode_lengths = [3, 2]
    total_frames = sum(episode_lengths)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_lengths),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            camera_key: {
                "dtype": "video",
                "shape": [4, 4, 3],
                "names": ["height", "width", "rgb"],
                "info": {"video.fps": 10},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(root / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f)

    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": episode_idx, "tasks": ["debug task"], "length": length}
            for episode_idx, length in enumerate(episode_lengths)
        ],
    )
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "debug task"}])

    absolute_index = 0
    for episode_index, length in enumerate(episode_lengths):
        frame_indices = list(range(length))
        table = pa.table(
            {
                "timestamp": [frame_index / 10.0 for frame_index in frame_indices],
                "frame_index": frame_indices,
                "episode_index": [episode_index] * length,
                "index": list(range(absolute_index, absolute_index + length)),
                "task_index": [0] * length,
            }
        )
        pq.write_table(
            table,
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
        )
        (root / "videos" / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4").touch()
        absolute_index += length

    return {"camera_key": camera_key, "episode_lengths": episode_lengths}


def test_lerobot_v2_metadata_rebuilds_v3_like_episode_offsets(tmp_path):
    fixture = _write_fake_lerobot_v2(tmp_path)

    metadata = datamodules.LeRobotV2Metadata(repo_id="libero", root=tmp_path)

    assert datamodules._resolve_lerobot_format("auto", str(tmp_path)) == "v2"
    assert metadata.camera_keys == [fixture["camera_key"]]
    assert metadata.episodes[0]["dataset_from_index"] == 0
    assert metadata.episodes[0]["dataset_to_index"] == 3
    assert metadata.episodes[1]["dataset_from_index"] == 3
    assert metadata.episodes[1]["dataset_to_index"] == 5


def test_lerobot_v2_dataset_decodes_clamped_context(monkeypatch, tmp_path):
    fixture = _write_fake_lerobot_v2(tmp_path)
    camera_key = fixture["camera_key"]
    calls = []

    def fake_decode_video_frames(video_path, timestamps, tolerance_s, backend):
        calls.append((video_path, timestamps, tolerance_s, backend))
        return torch.stack(
            [
                torch.full((3, 2, 2), fill_value=float(idx), dtype=torch.float32)
                for idx, _ in enumerate(timestamps)
            ]
        )

    monkeypatch.setattr(
        "lerobot.datasets.video_utils.decode_video_frames",
        fake_decode_video_frames,
    )

    dataset = datamodules.LeRobotV2Dataset(
        repo_id="libero",
        root=tmp_path,
        delta_timestamps={camera_key: [-0.1, 0.0]},
        video_backend="pyav",
    )
    item = dataset[0]

    assert item[camera_key].shape == (2, 3, 2, 2)
    assert item[f"{camera_key}_is_pad"].tolist() == [True, False]
    assert calls[0][1] == pytest.approx([0.0, 0.0])
    assert calls[0][3] == "pyav"
    assert item["episode_index"].item() == 0
    assert item["frame_index"].item() == 0
    assert item["task"] == "debug task"


def test_mixed_lerobot_datamodule_concats_splits_and_uses_per_dataset_fps():
    datamodule = _make_mixed_lerobot_datamodule()
    datamodule.setup("fit")

    assert isinstance(datamodule.train_set, torch.utils.data.ConcatDataset)
    assert len(datamodule.train_set.datasets) == 2
    assert len(datamodule.train_set) == 8
    assert len(datamodule.val_set) == 4
    assert len(datamodule.test_set) == 4

    assert datamodule.created_datasets[0]["delta_timestamps"][
        "observation.images.top_head"
    ] == pytest.approx([-0.1, 0.0])
    assert datamodule.created_datasets[1]["delta_timestamps"][
        "observation.images.egocentric"
    ] == pytest.approx([-1.0 / 12.0, 0.0])
    assert datamodule.created_datasets[0]["episodes"] is None
    assert datamodule.created_datasets[1]["episodes"] is None

    assert datamodule.train_set.datasets[0][0]["dataset_index"].item() == 0
    assert datamodule.train_set.datasets[1][0]["dataset_index"].item() == 1

    batch = next(iter(datamodule.train_dataloader()))
    assert batch["video"].shape == (2, 2, 3, 2, 2)
    assert batch["dataset_index"].shape == (2,)


def test_mixed_lerobot_datamodule_natural_sampling_uses_random_sampler():
    datamodule = _make_mixed_lerobot_datamodule(val_fraction=0.0, test_fraction=0.0)
    loader = datamodule.train_dataloader()

    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)


def test_mixed_lerobot_datamodule_weighted_sampling_uses_weighted_sampler():
    datamodule = _make_mixed_lerobot_datamodule(
        sampling={"mode": "weighted", "weights": [0.5, 0.5], "num_samples": 6},
        val_fraction=0.0,
        test_fraction=0.0,
    )
    loader = datamodule.train_dataloader()

    assert isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
    assert len(loader.sampler) == 6
    batch = next(iter(loader))
    assert batch["video"].shape == (2, 2, 3, 2, 2)


def test_mixed_lerobot_datamodule_empty_eval_splits_return_none():
    datamodule = _make_mixed_lerobot_datamodule(val_fraction=0.0, test_fraction=0.0)
    datamodule.setup("fit")

    assert datamodule.val_set is None
    assert datamodule.test_set is None
    assert datamodule.val_dataloader() is None
    assert datamodule.test_dataloader() is None


def test_mixed_lerobot_datamodule_can_disable_dataset_index():
    datamodule = _make_mixed_lerobot_datamodule(
        return_dataset_index=False, val_fraction=0.0, test_fraction=0.0
    )
    datamodule.setup("fit")

    assert "dataset_index" not in datamodule.train_set.datasets[0][0]


def test_mixed_lerobot_datamodule_passes_episode_subset_to_lerobot_dataset():
    datamodule = _make_mixed_lerobot_datamodule(
        train_episodes=[2],
        val_fraction=0.0,
        test_fraction=0.0,
    )
    datamodule.setup("fit")

    assert datamodule.created_datasets[0]["episodes"] == [2]
    assert datamodule.created_datasets[1]["episodes"] == [2]
    assert len(datamodule.train_set) == 2 * 2


def test_mixed_lerobot_datamodule_records_per_dataset_format():
    datamodule = _make_mixed_lerobot_datamodule(
        datasets=[
            {
                "repo_id": "ego",
                "root": "/data/ego",
                "camera_key": "observation.images.top_head",
                "lerobot_format": "v2",
            },
            {
                "repo_id": "ssv2",
                "root": "/data/ssv2",
                "camera_key": "observation.images.egocentric",
                "lerobot_format": "v3",
            },
        ],
        val_fraction=0.0,
        test_fraction=0.0,
    )
    datamodule.setup("fit")

    assert datamodule.dataset_info[0]["lerobot_format"] == "v2"
    assert datamodule.dataset_info[1]["lerobot_format"] == "v3"


def test_mixed_lerobot_cache_dataloader_covers_episode_boundaries():
    datamodule = _make_mixed_lerobot_datamodule(val_fraction=0.0, test_fraction=0.0)
    loaders = datamodule.cache_dataloaders(["train"])

    assert len(loaders) == 1
    assert len(loaders[0].dataset) == 2 * 4 * 3

    first = loaders[0].dataset[0]
    assert first["cache_row"].item() == 0
    assert first["dataset_index"].item() == 0
    assert first["split_index"].item() == 0
    assert first["episode_index"].item() == 0
    assert first["frame_index"].item() == 0
    assert first["absolute_index"].item() == 0
    assert first["context_absolute_indices"].tolist() == [0, 0]
    assert first["context_frame_indices"].tolist() == [0, 0]
    assert first["context_is_pad"].tolist() == [True, False]


def test_mixed_lerobot_cache_rows_are_stable_across_splits_and_datasets():
    datamodule = _make_mixed_lerobot_datamodule()
    loaders = datamodule.cache_dataloaders(["train", "val", "test"])

    rows = []
    split_indices = []
    for loader in loaders:
        for sample_idx in range(len(loader.dataset)):
            sample = loader.dataset[sample_idx]
            rows.append(sample["cache_row"].item())
            split_indices.append(sample["split_index"].item())

    assert rows == list(range(len(rows)))
    assert split_indices == sorted(split_indices)
    assert len(rows) == 2 * 4 * 3


class _FakeLeRobotDataModule(datamodules.LeRobotDataModule):
    def _get_metadata(self):
        return _FakeLeRobotMetadata(
            fps=10.0,
            camera_key="observation.images.egocentric",
            total_episodes=3,
        )

    def _get_lerobot_dataset(self, episodes=None):
        if self._lerobot_dataset is None:
            self._lerobot_dataset = _FakeLeRobotDataset(
                "observation.images.egocentric",
                self._get_delta_timestamps(),
                episodes,
                total_episodes=3,
                frames_per_episode=3,
            )
        return self._lerobot_dataset


def test_lerobot_cache_dataloader_single_dataset_provenance():
    datamodule = _FakeLeRobotDataModule(
        repo_id="ssv2",
        frame_sampling={"num_frames": 2},
        batch_size=2,
        num_workers=0,
        val_fraction=0.0,
        test_fraction=0.0,
    )
    loader = datamodule.cache_dataloaders(["train"])[0]

    assert len(loader.dataset) == 3 * 3
    batch = next(iter(loader))
    assert batch["video"].shape == (2, 2, 3, 2, 2)
    assert batch["cache_row"].tolist() == [0, 1]
    assert batch["dataset_index"].tolist() == [0, 0]
    assert batch["episode_index"].tolist() == [0, 0]
    assert batch["frame_index"].tolist() == [0, 1]


def test_lerobot_v2_cache_dataloader_provenance(monkeypatch, tmp_path):
    fixture = _write_fake_lerobot_v2(tmp_path)
    camera_key = fixture["camera_key"]

    def fake_decode_video_frames(video_path, timestamps, tolerance_s, backend):
        return torch.zeros(len(timestamps), 3, 4, 4)

    monkeypatch.setattr(
        "lerobot.datasets.video_utils.decode_video_frames",
        fake_decode_video_frames,
    )

    datamodule = datamodules.LeRobotDataModule(
        repo_id="libero",
        root=str(tmp_path),
        camera_key=camera_key,
        lerobot_format="auto",
        frame_sampling={"num_frames": 2},
        batch_size=2,
        num_workers=0,
        train_episodes=[0],
        val_fraction=0.0,
        test_fraction=0.0,
        image_size=4,
    )
    loader = datamodule.cache_dataloaders(["train"])[0]
    batch = next(iter(loader))

    assert len(loader.dataset) == fixture["episode_lengths"][0]
    assert datamodule.dataset_info[0]["lerobot_format"] == "v2"
    assert batch["video"].shape == (2, 2, 3, 4, 4)
    assert batch["cache_row"].tolist() == [0, 1]
    assert batch["episode_index"].tolist() == [0, 0]
    assert batch["frame_index"].tolist() == [0, 1]
    assert batch["absolute_index"].tolist() == [0, 1]
    assert batch["context_absolute_indices"].tolist() == [[0, 0], [0, 1]]
    assert batch["context_frame_indices"].tolist() == [[0, 0], [0, 1]]
    assert batch["context_is_pad"].tolist() == [[True, False], [False, False]]
