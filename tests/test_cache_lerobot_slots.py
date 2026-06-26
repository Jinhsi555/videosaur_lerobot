from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from videosaur.cache_lerobot_slots import (
    CACHE_FORMAT,
    FEATURE_SHARD_DIR,
    FEATURE_SHARD_PREFIX,
    INDEX_FRAGMENT_DIR,
    INDEX_FRAGMENT_PREFIX,
    SlotFeatureCache,
    SlotCachePredictDataModule,
    SlotFeaturePredictor,
    SlotFeaturePredictionWriter,
    _build_metadata,
    finalize_sharded_cache,
)


class _DummyVideoModel(torch.nn.Module):
    def forward(self, batch):
        batch_size, n_frames = batch["video"].shape[:2]
        state = torch.zeros(batch_size, n_frames, 2, 3)
        for timestep in range(n_frames):
            state[:, timestep] = float(timestep)
        return {"processor": {"state": state}}


def _make_batch(batch_size=2, n_frames=4):
    return {
        "video": torch.zeros(batch_size, n_frames, 3, 2, 2),
        "cache_row": torch.arange(batch_size, dtype=torch.long),
        "dataset_index": torch.zeros(batch_size, dtype=torch.long),
        "split_index": torch.zeros(batch_size, dtype=torch.long),
        "episode_index": torch.zeros(batch_size, dtype=torch.long),
        "frame_index": torch.arange(batch_size, dtype=torch.long),
        "absolute_index": torch.arange(batch_size, dtype=torch.long),
        "timestamp": torch.arange(batch_size, dtype=torch.float64),
        "task_index": torch.zeros(batch_size, dtype=torch.long),
        "context_absolute_indices": torch.zeros(batch_size, n_frames, dtype=torch.long),
        "context_frame_indices": torch.zeros(batch_size, n_frames, dtype=torch.long),
        "context_is_pad": torch.zeros(batch_size, n_frames, dtype=torch.bool),
    }


def _prediction(rows, values):
    rows = torch.tensor(rows, dtype=torch.long)
    batch_size = len(rows)
    provenance = _make_batch(batch_size=batch_size, n_frames=2)
    provenance["cache_row"] = rows
    provenance["frame_index"] = rows.clone()
    provenance["absolute_index"] = rows.clone()
    return {
        "features": torch.tensor(values, dtype=torch.float16).reshape(batch_size, 2, 3),
        "provenance": provenance,
    }


class _DummyTrainer:
    global_rank = 0


def _dataset_info():
    return [
        {
            "repo_id": "repo",
            "root": "/data/repo",
            "camera_key": "observation.images.egocentric",
            "fps": 10.0,
        }
    ]


def _metadata(target_shard_mb=0.000001):
    return {
        "schema_version": 2,
        "cache_format": CACHE_FORMAT,
        "complete": False,
        "dtype": "float16",
        "target_shard_mb": target_shard_mb,
    }


def _writer(cache_dir, target_shard_mb=0.000001):
    writer = SlotFeaturePredictionWriter(
        cache_dir,
        split_names=["train"],
        dataset_info=_dataset_info(),
        dtype="float16",
        target_shard_mb=target_shard_mb,
    )
    writer.setup(None, None)
    return writer


def _write_prediction(writer, prediction):
    writer.write_on_batch_end(
        _DummyTrainer(),
        None,
        prediction,
        batch_indices=None,
        batch=None,
        batch_idx=0,
        dataloader_idx=0,
    )


def _flush_writer(writer):
    writer.write_on_epoch_end(_DummyTrainer(), None, predictions=[], batch_indices=None)


def _finalize(cache_dir, expected_rows=3):
    finalize_sharded_cache(
        cache_dir,
        expected_rows=expected_rows,
        dtype="float16",
        metadata=_metadata(),
    )


def test_slot_feature_predictor_uses_last_video_timestep_and_dtype():
    predictor = SlotFeaturePredictor(_DummyVideoModel(), output_dtype=torch.float16)

    output = predictor.predict_step(_make_batch(batch_size=2, n_frames=4), batch_idx=0)

    assert output["features"].shape == (2, 2, 3)
    assert output["features"].dtype == torch.float16
    assert torch.all(output["features"] == 3)
    assert output["provenance"]["cache_row"].tolist() == [0, 1]


def test_slot_cache_predict_datamodule_builds_sharded_loader_after_ddp_init(monkeypatch):
    calls = []

    class _DummyDataModule:
        def cache_dataloaders(self, splits, *, distributed, rank, world_size):
            calls.append(
                {
                    "splits": splits,
                    "distributed": distributed,
                    "rank": rank,
                    "world_size": world_size,
                }
            )
            return ["loader"]

    monkeypatch.setattr(
        "videosaur.cache_lerobot_slots._distributed_rank_world",
        lambda: (1, 2),
    )

    datamodule = SlotCachePredictDataModule(_DummyDataModule(), ["train"])

    assert datamodule.predict_dataloader() == ["loader"]
    assert calls == [
        {
            "splits": ["train"],
            "distributed": True,
            "rank": 1,
            "world_size": 2,
        }
    ]


def test_build_metadata_records_cache_seed():
    args = SimpleNamespace(
        config="configs/videosaur/Libero_slot_cache.yml",
        checkpoint="checkpoint.ckpt",
        dtype="float16",
        target_shard_mb=256.0,
        config_overrides=["dataset.train_episodes=[0]"],
    )

    metadata = _build_metadata(
        args=args,
        splits=["train"],
        dataset_info=_dataset_info(),
        expected_rows=3,
        cache_seed=123,
    )

    assert metadata["cache_seed"] == 123


def test_prediction_writer_writes_sharded_features_and_index_fragments(tmp_path):
    writer = _writer(tmp_path)

    _write_prediction(writer, _prediction([0], np.arange(6, dtype=np.float16)))
    _write_prediction(writer, _prediction([1], np.arange(6, dtype=np.float16) + 20))
    _flush_writer(writer)

    feature_parts = sorted(
        (tmp_path / FEATURE_SHARD_DIR).glob(f"{FEATURE_SHARD_PREFIX}*.npy")
    )
    fragments = sorted(
        (tmp_path / INDEX_FRAGMENT_DIR).glob(f"{INDEX_FRAGMENT_PREFIX}*.parquet")
    )
    assert len(feature_parts) == 2
    assert len(fragments) == 2

    first_feature = np.load(feature_parts[0])
    first_index = pq.read_table(fragments[0])
    assert first_feature.shape == (1, 2, 3)
    assert first_feature.dtype == np.float16
    assert first_index["shard_offset"].to_pylist() == [0]


def test_finalize_sharded_cache_writes_index_metadata_and_reader(tmp_path):
    writer = _writer(tmp_path)
    _write_prediction(writer, _prediction([0, 2], np.arange(12, dtype=np.float16)))
    _write_prediction(writer, _prediction([1], np.arange(6, dtype=np.float16) + 20))
    _flush_writer(writer)
    _finalize(tmp_path)

    cache = SlotFeatureCache(tmp_path)
    assert cache.metadata["cache_format"] == CACHE_FORMAT
    assert cache.metadata["complete"] is True
    assert cache.metadata["feature_shape"] == [3, 2, 3]
    assert cache.lookup(repo_id="repo", episode_index=0, frame_index=1) == 1
    np.testing.assert_array_equal(
        cache.get_by_key(dataset_index=0, episode_index=0, frame_index=2),
        np.arange(6, 12, dtype=np.float16).reshape(2, 3),
    )
    np.testing.assert_array_equal(
        cache.get_many([1, 0]),
        np.stack(
            [
                np.arange(6, dtype=np.float16).reshape(2, 3) + 20,
                np.arange(6, dtype=np.float16).reshape(2, 3),
            ]
        ),
    )


def test_finalize_sharded_cache_rejects_duplicate_rows(tmp_path):
    writer = _writer(tmp_path)
    _write_prediction(writer, _prediction([0], np.arange(6, dtype=np.float16)))
    _write_prediction(writer, _prediction([0], np.arange(6, dtype=np.float16)))
    _flush_writer(writer)

    with pytest.raises(ValueError, match="Duplicate cache rows"):
        _finalize(tmp_path, expected_rows=1)


def test_finalize_sharded_cache_rejects_missing_rows(tmp_path):
    writer = _writer(tmp_path)
    _write_prediction(writer, _prediction([0], np.arange(6, dtype=np.float16)))
    _flush_writer(writer)

    with pytest.raises(ValueError, match="Missing cache rows"):
        _finalize(tmp_path, expected_rows=2)


def test_finalize_sharded_cache_rejects_missing_shard(tmp_path):
    writer = _writer(tmp_path)
    _write_prediction(writer, _prediction([0], np.arange(6, dtype=np.float16)))
    _flush_writer(writer)
    for path in (tmp_path / FEATURE_SHARD_DIR).glob(f"{FEATURE_SHARD_PREFIX}*.npy"):
        path.unlink()

    with pytest.raises(FileNotFoundError, match="missing"):
        _finalize(tmp_path, expected_rows=1)
