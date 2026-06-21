# LeRobot VideoSAUR 工作流

这份文档记录当前仓库里 LeRobot 相关的 VideoSAUR 工作流：

- 如何配置训练、推理和 slot feature cache
- config 里的字段如何对应到框架组件
- 如何读取生成好的 slot feature cache

## Config 入口

常用配置文件：

- `configs/videosaur/lerobot_mixed_debug.yml`：混合 LeRobot 数据集的小规模调试配置。
- `configs/videosaur/lerobot_mixed_dataset.yml`：混合 LeRobot 数据集的完整训练配置。
- `configs/videosaur/lerobot_something_something_v2.yml`：单个 LeRobot 数据集训练配置。
- `configs/inference/movi_c.yml`：当前推理配置，使用已经训练好的 LeRobot checkpoint。

训练命令：

```bash
.venv/bin/python -m videosaur.train configs/videosaur/lerobot_mixed_dataset.yml
```

生成 slot feature cache：

```bash
.venv/bin/python -m videosaur.cache_lerobot_slots \
  configs/videosaur/lerobot_mixed_debug.yml \
  --checkpoint logs/videosaur/2026-06-16-16-48-05_lerobot_mixed_3/checkpoints/step=50000.ckpt \
  --output-dir slot_cache_debug/lerobot_mixed_debug \
  --splits train \
  --target-shard-mb 256 \
  --overwrite
```

命令最后可以追加 `key=value` 形式的 OmegaConf dotlist override：

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

## Config 结构

顶层 config 由 `videosaur.configuration.load_config` 读取。

- `globals`：共享常量和派生变量。其他字段可以用 `${globals.NAME}` 引用。
- `trainer`：经过 `videosaur.train._setup_trainer_config` 处理后传给 `pytorch_lightning.Trainer`。
- `optimizer`：用于构建 `videosaur.optimizers.OptimizerBuilder`。
- `model`：传给 `videosaur.models.build`，用于构建 VideoSAUR 模型。
- `dataset`：传给 `videosaur.data.build`，用于构建 LightningDataModule。
- `train_metrics` / `val_metrics`：可选指标配置，由 `videosaur.metrics.build` 构建。
- `checkpoint_*`：训练脚本用来配置 `ModelCheckpoint`。

常用 OmegaConf resolver：

- `${mul: a, b}`：乘法。
- `${eval: 'expr', ...}`：计算派生值，例如按 batch size 缩放学习率。
- `${config_prop: VIT_PARAMS, ${.VIT_MODEL}, FEAT_DIM}`：从 `videosaur.configuration` 里预定义的 ViT 参数表读取值。

## 框架组件对应关系

`dataset.name` 选择使用哪个 LightningDataModule：

| Config 值 | 实现 |
| --- | --- |
| `LeRobotDataModule` | 单个 LeRobot 数据集适配器 |
| `MixedLeRobotDataModule` | 多个 LeRobot 数据集混合/拼接 |
| `WebdatasetDataModule` | 原始 WebDataset 数据管线 |
| `DummyDataModule` | 测试和调试数据 |

LeRobot 相关字段：

- `dataset.datasets`：混合数据集列表。每个子项定义 `repo_id`、`root` 和 `camera_key`。
- `dataset.lerobot_format`：LeRobot 数据格式，支持 `auto`、`v2`、`v3`，默认 `auto`。
- `dataset.datasets[].lerobot_format`：混合数据集里单个子数据集的格式覆盖项。
- `dataset.frame_sampling.num_frames`：每个 VideoSAUR 样本包含多少帧。
- `dataset.return_dataset_index`：混合数据集 batch 里是否返回 `dataset_index`。
- `dataset.sampling.mode`：`natural` 保持原始顺序和数据量；`weighted` 使用 `WeightedRandomSampler`。
- `dataset.train_episodes`、`val_episodes`、`test_episodes`：显式 episode 切分。
- `dataset.val_fraction`、`test_fraction`：没有显式 split 时，用比例自动切分 episode。

### LeRobot v2 / v3 格式

当前 dataloader 默认使用 `lerobot_format: auto`：

- v3 数据继续走 LeRobot 官方 `LeRobotDataset`。
- v2.1 数据会在读取时适配成 v3-like 接口，不会修改原始 LIBERO 目录。
- v2.1 自动识别依据是 `meta/info.json` 里的 `codebase_version: v2.1`，或存在 `meta/episodes.jsonl`。

LIBERO v2 示例：

```yaml
dataset:
  name: MixedLeRobotDataModule
  datasets:
    - repo_id: Libero
      root: /mnt/data/licy/FastWAM-AuxSignals/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot
      lerobot_format: v2
      camera_key: observation.images.image
```

v2 适配层会读取旧格式的：

- `data/chunk-000/episode_000000.parquet`
- `videos/chunk-000/{video_key}/episode_000000.mp4`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`

对上游训练和 cache 来说，它仍然提供 `dataset_from_index`、`dataset_to_index`、`episode_index`、`frame_index`、`timestamp`、`task_index` 和选定 camera 的 `[T,C,H,W]` 视频张量。

`model` 会被映射到 `ObjectCentricModel` 的各个组件：

| Config 字段 | 构建函数 | 运行时作用 |
| --- | --- | --- |
| `model.initializer` | `modules.build_initializer` | 生成初始 slot tensor |
| `model.encoder` | `modules.build_encoder(..., "FrameEncoder")` | 把图像帧编码成 patch feature |
| `model.grouper` | `modules.build_grouper` | 根据当前帧 feature 更新 slots |
| `model.latent_processor` | `modules.build_video(..., "LatentProcessor")` | 配置视频里的 recurrent slot 更新逻辑 |
| `model.predictor` | `modules.build_module` | 在帧之间预测下一步 slot state |
| `model.decoder` | `modules.build_decoder` | 根据 slots 重建目标 |
| `model.losses` | `losses.build` | 定义训练目标 |
| `model.mask_resizers` | `modules.build_utils(..., "Resizer")` | 为可视化/指标 resize mask |

当 `model.input_type: video` 时，`models.build` 会把帧级模块包装成视频级模块：

- encoder 变成 `MapOverTime(encoder)`
- decoder 变成 `MapOverTime(decoder)`
- processor 变成 `ScanOverTime(LatentProcessor(...))`

因此 `outputs["processor"]["state"]` 的 shape 是 `[batch, frames, n_slots, slot_dim]`。
slot cache 脚本保存的是 `outputs["processor"]["state"][:, -1]`，也就是每个样本当前帧对应的 slot feature。

## Slot Feature Cache 结构

生成好的 cache 目录包含：

- `features/rank=*-part=*.npy`：slot feature 分片文件，shape 为 `[num_shard_rows, n_slots, slot_dim]`。
- `index.parquet`：帧级 provenance 表。`row_id` 是全局稳定 id，feature 由 `shard_path + shard_offset` 定位。
- `metadata.json`：cache 元信息，也是完成标记。只有 `complete: true` 的 cache 才应该被下游读取。

生成过程中还会临时写入 `_index_fragments/rank=*-part=*.parquet`。rank0 完成校验并合并出最终 `index.parquet` 后会删除这些 fragment 文件。

面向对象存储 OSS 时不要使用单体 memmap 文件。当前 cache 格式固定为 `sharded_npy`，feature shard 放在 `features/` 子目录，生成期 index fragment 放在 `_index_fragments/` 子目录。若 OSS 挂载不支持普通 POSIX 写入，建议先把 `--output-dir` 指向 CPFS 或本地 staging 目录，完成后再同步到 OSS：

```bash
.venv/bin/python -m videosaur.cache_lerobot_slots \
  configs/videosaur/lerobot_mixed_debug.yml \
  --checkpoint logs/videosaur/2026-06-16-16-48-05_lerobot_mixed_3/checkpoints/step=50000.ckpt \
  --output-dir /mnt/oss_data/libero_slot_cache \
  --splits train \
  --target-shard-mb 256 \
  --overwrite
```

默认 shard 目标大小是 256 MiB；脚本会根据 `[n_slots, slot_dim]` 和 dtype 自动决定每个 shard 容纳多少行。

cache row 可以通过这些字段溯源：

- `dataset_index`
- `repo_id`
- `episode_index`
- `frame_index`
- `absolute_index`
- `shard_path`
- `shard_offset`

cache dataloader 还会记录输入窗口上下文：

- `context_absolute_indices`
- `context_frame_indices`
- `context_is_pad`

这些字段说明当前 feature 是由哪些 LeRobot 帧构成输入窗口，以及窗口边界处是否发生了 padding/clamp。

## 读取 Cache

使用 `scripts/read_slot_cache.py` 查看 cache 摘要和读取 feature。

```bash
.venv/bin/python scripts/read_slot_cache.py slot_cache_debug/lerobot_mixed_debug
```

按 `row_id` 读取 feature：

```bash
.venv/bin/python scripts/read_slot_cache.py \
  slot_cache_debug/lerobot_mixed_debug \
  --row 0
```

按 LeRobot provenance 读取 feature：

```bash
.venv/bin/python scripts/read_slot_cache.py \
  slot_cache_debug/lerobot_mixed_debug \
  --repo-id ego_exo4d \
  --episode-index 0 \
  --frame-index 0
```

把选中的 feature 保存成单独的 `.npy`：

```bash
.venv/bin/python scripts/read_slot_cache.py \
  slot_cache_debug/lerobot_mixed_debug \
  --row 0 \
  --save-feature /tmp/slot_feature.npy
```

## Debug Cache 读取

使用 `scripts/read_slot_cache_minimal.py` 作为最小 debugger 入口：

```bash
.venv/bin/python scripts/read_slot_cache_minimal.py
```

这个脚本只加载：

- `features`
- `index`
- `metadata`

然后打印一行摘要。可以在加载语句后面打断点，直接在 debugger 里查看对象内容。
