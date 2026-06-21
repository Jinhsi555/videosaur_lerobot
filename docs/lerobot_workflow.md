# LeRobot VideoSAUR Workflow

This document covers the LeRobot-specific VideoSAUR workflow in this repository:

- how to configure training, inference, and slot feature caching
- how config sections map to framework components
- how to inspect generated slot feature caches

## Config Entry Points

Use these configs as starting points:

- `configs/videosaur/lerobot_mixed_debug.yml`: small mixed LeRobot debug config.
- `configs/videosaur/lerobot_mixed_dataset.yml`: full mixed LeRobot training config.
- `configs/videosaur/lerobot_something_something_v2.yml`: single LeRobot dataset training config.
- `configs/inference/movi_c.yml`: current inference config using a trained LeRobot checkpoint.

Run training:

```bash
.venv/bin/python -m videosaur.train configs/videosaur/lerobot_mixed_dataset.yml
```

Run slot cache generation:

```bash
.venv/bin/python -m videosaur.cache_lerobot_slots \
  configs/videosaur/lerobot_mixed_debug.yml \
  --checkpoint logs/videosaur/2026-06-16-16-48-05_lerobot_mixed_3/checkpoints/step=50000.ckpt \
  --output-dir slot_cache_debug/lerobot_mixed_debug \
  --splits train \
  --target-shard-mb 256 \
  --overwrite
```

Any trailing `key=value` arguments are OmegaConf dotlist overrides:

```bash
.venv/bin/python -m videosaur.cache_lerobot_slots \
  configs/videosaur/lerobot_mixed_debug.yml \
  --checkpoint logs/videosaur/2026-06-16-16-48-05_lerobot_mixed_3/checkpoints/step=50000.ckpt \
  --output-dir slot_cache_debug/lerobot_mixed_debug \
  --splits train \
  --overwrite \
  dataset.train_episodes=[0] \
  dataset.val_episodes=null \
  dataset.test_episodes=null \
  dataset.batch_size=2
```

## Config Structure

Top-level config sections are loaded by `videosaur.configuration.load_config`.

- `globals`: shared constants and derived values. Other fields reference these with `${globals.NAME}`.
- `trainer`: passed into `pytorch_lightning.Trainer` after `videosaur.train._setup_trainer_config`.
- `optimizer`: used by `videosaur.optimizers.OptimizerBuilder`.
- `model`: used by `videosaur.models.build`.
- `dataset`: used by `videosaur.data.build`.
- `train_metrics` / `val_metrics`: optional metric configs built by `videosaur.metrics.build`.
- `checkpoint_*`: used by the training script to configure `ModelCheckpoint`.

Useful OmegaConf resolvers:

- `${mul: a, b}`: multiply values.
- `${eval: 'expr', ...}`: compute derived values such as learning rate scaling.
- `${config_prop: VIT_PARAMS, ${.VIT_MODEL}, FEAT_DIM}`: read predefined ViT metadata from `videosaur.configuration`.

## Framework Component Mapping

`dataset.name` selects the LightningDataModule:

| Config value | Implementation |
| --- | --- |
| `LeRobotDataModule` | single LeRobot dataset adapter |
| `MixedLeRobotDataModule` | multiple LeRobot datasets concatenated or sampled together |
| `WebdatasetDataModule` | original WebDataset pipeline |
| `DummyDataModule` | test/debug data |

For LeRobot configs:

- `dataset.datasets`: mixed dataset list. Each item defines `repo_id`, `root`, and `camera_key`.
- `dataset.lerobot_format`: LeRobot storage format, one of `auto`, `v2`, or `v3`. Defaults to `auto`.
- `dataset.datasets[].lerobot_format`: per-child override for mixed datasets.
- `dataset.frame_sampling.num_frames`: number of video frames per VideoSAUR sample.
- `dataset.return_dataset_index`: adds `dataset_index` to mixed dataset batches.
- `dataset.sampling.mode`: `natural` keeps dataset order/size; `weighted` uses `WeightedRandomSampler`.
- `dataset.train_episodes`, `val_episodes`, `test_episodes`: explicit episode splits.
- `dataset.val_fraction`, `test_fraction`: automatic episode split fractions when explicit lists are absent.

### LeRobot v2 / v3 Format

The dataloaders default to `lerobot_format: auto`:

- v3 datasets continue to use LeRobot's official `LeRobotDataset`.
- v2.1 datasets are adapted at read time into the v3-like interface used by VideoSAUR.
- Auto detection treats `meta/info.json` with `codebase_version: v2.1`, or a dataset with `meta/episodes.jsonl`, as v2.

LIBERO v2 example:

```yaml
dataset:
  name: MixedLeRobotDataModule
  datasets:
    - repo_id: Libero
      root: /mnt/data/licy/FastWAM-AuxSignals/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot
      lerobot_format: v2
      camera_key: observation.images.image
```

The v2 adapter reads legacy files such as:

- `data/chunk-000/episode_000000.parquet`
- `videos/chunk-000/{video_key}/episode_000000.mp4`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`

For training and cache generation it exposes the same fields this repo already consumes:
`dataset_from_index`, `dataset_to_index`, `episode_index`, `frame_index`, `timestamp`,
`task_index`, and the selected camera as a `[T,C,H,W]` tensor.

`model` maps to `ObjectCentricModel` components:

| Config section | Built by | Runtime role |
| --- | --- | --- |
| `model.initializer` | `modules.build_initializer` | creates initial slot tensors |
| `model.encoder` | `modules.build_encoder(..., "FrameEncoder")` | encodes frames into patch features |
| `model.grouper` | `modules.build_grouper` | updates slots from frame features |
| `model.latent_processor` | `modules.build_video(..., "LatentProcessor")` | configures recurrent video slot updates |
| `model.predictor` | `modules.build_module` | predicts next slot state between frames |
| `model.decoder` | `modules.build_decoder` | reconstructs targets from slots |
| `model.losses` | `losses.build` | defines training objectives |
| `model.mask_resizers` | `modules.build_utils(..., "Resizer")` | resizes masks for visualization/metrics |

For `model.input_type: video`, `models.build` wraps frame-level modules:

- encoder becomes `MapOverTime(encoder)`
- decoder becomes `MapOverTime(decoder)`
- processor becomes `ScanOverTime(LatentProcessor(...))`

That means `outputs["processor"]["state"]` has shape `[batch, frames, n_slots, slot_dim]`.
The slot cache script saves `outputs["processor"]["state"][:, -1]`, so each cache row represents the current frame.

## Slot Feature Cache Layout

A generated cache directory contains:

- `features/rank=*-part=*.npy`: slot feature shards with shape `[num_shard_rows, n_slots, slot_dim]`.
- `index.parquet`: frame provenance table. `row_id` is globally stable, and features are addressed by `shard_path + shard_offset`.
- `metadata.json`: cache metadata and the completion marker. Downstream code should only read caches with `complete: true`.

During generation, the cache writer also creates temporary `_index_fragments/rank=*-part=*.parquet` files. Rank 0 validates and merges these fragments into the final `index.parquet`, then deletes the fragment files.

For object storage, do not use a single mmap feature file. The cache format is fixed to `sharded_npy`; feature shards live under `features/`, and generation-time index fragments live under `_index_fragments/`. If an OSS mount does not support normal POSIX writes, point `--output-dir` at a CPFS or local staging directory first, then sync the completed cache to OSS:

```bash
.venv/bin/python -m videosaur.cache_lerobot_slots \
  configs/videosaur/lerobot_mixed_debug.yml \
  --checkpoint logs/videosaur/2026-06-16-16-48-05_lerobot_mixed_3/checkpoints/step=50000.ckpt \
  --output-dir /mnt/oss_data/libero_slot_cache \
  --splits train \
  --target-shard-mb 256 \
  --overwrite
```

The default target shard size is 256 MiB; the script derives rows per shard from `[n_slots, slot_dim]` and dtype.

Cache rows are keyed by:

- `dataset_index`
- `repo_id`
- `episode_index`
- `frame_index`
- `absolute_index`
- `shard_path`
- `shard_offset`

The cache dataloader also records context fields:

- `context_absolute_indices`
- `context_frame_indices`
- `context_is_pad`

These show which LeRobot frames formed the input window and whether any frame was padded/clamped at episode boundaries.

## Read Cache Files

Use `scripts/read_slot_cache.py` for command-line inspection and feature lookup.

```bash
.venv/bin/python scripts/read_slot_cache.py slot_cache_debug/lerobot_mixed_debug
```

Read a feature by `row_id`:

```bash
.venv/bin/python scripts/read_slot_cache.py \
  slot_cache_debug/lerobot_mixed_debug \
  --row 0
```

Read a feature by LeRobot provenance:

```bash
.venv/bin/python scripts/read_slot_cache.py \
  slot_cache_debug/lerobot_mixed_debug \
  --repo-id ego_exo4d \
  --episode-index 0 \
  --frame-index 0
```

Save a selected feature to a standalone `.npy` file:

```bash
.venv/bin/python scripts/read_slot_cache.py \
  slot_cache_debug/lerobot_mixed_debug \
  --row 0 \
  --save-feature /tmp/slot_feature.npy
```

## Debug Cache Loading

Use `scripts/read_slot_cache_minimal.py` as a minimal debugger entrypoint.

```bash
.venv/bin/python scripts/read_slot_cache_minimal.py
```

The script only loads:

- `features`
- `index`
- `metadata`

and prints one summary line. Set a breakpoint after the load statements to inspect the objects directly.
