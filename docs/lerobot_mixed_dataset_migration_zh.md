# 混合多个 LeRobot 数据集的改动历程与迁移记录

这份文档记录本仓库从“单个 LeRobot 数据集训练”演进到“多个 LeRobot 数据集混合训练与 slot cache”的关键改动，方便把同一套能力迁移到其他分支或仓库。

## 目标

混合数据集能力的核心目标不是把多个 LeRobot 目录物理合并，而是在训练和 cache 时保持每个 LeRobot repo 独立读取，再由 VideoSAUR datamodule 统一暴露 batch：

- 每个子数据集保留自己的 `repo_id`、`root`、`camera_key`、`fps` 和 episode split。
- 每个子数据集按自己的 fps 计算 `delta_timestamps`，避免跨数据集 fps 不一致导致时间窗错位。
- 训练侧使用 `ConcatDataset` 或 weighted sampler 混合样本。
- batch 可返回稳定的 `dataset_index`，供后续 cache、policy join 和调试溯源。
- slot cache 侧保存 `dataset_index/repo_id/episode_index/frame_index` 等 provenance，而不是只保存裸 feature。

## 提交演进

### 1. 单数据集 LeRobot 接入

提交：`e0bfbc0 Add LeRobot dataset training support`

这一阶段先把单个 LeRobot repo 接到 VideoSAUR：

- 在 `videosaur/data/datamodules.py` 中加入 `LeRobotImageTransform`，负责 resize、归一化以及 `[H,W,C]` / `[C,H,W]` 输入兼容。
- 加入 `LeRobotVideoDataset`，把 LeRobot 返回的 camera key 映射成 VideoSAUR 期望的 `video` key。
- 加入 `LeRobotDataModule`，负责 metadata 读取、camera key 选择、fps 获取、`delta_timestamps` 构造和 dataloader 创建。
- 增加单数据集配置 `configs/videosaur/lerobot_something_something_v2.yml`。

迁移要点：先确保单个 repo 可以训练，再做 mixed。Mixed 只是组合多个已经可用的单数据集适配器。

### 2. 补齐 train/val/test episode split

提交：`36ab7d8 feat(data): add LeRobot train/val/test split`

这一阶段把 split 逻辑补完整：

- `LeRobotDataModule` 支持 `train_episodes`、`val_episodes`、`test_episodes`。
- 支持 `val_fraction` 和 `test_fraction`，没有显式 split 时用 seed 随机划分。
- 加入重复 episode、越界 episode、split 重叠和空训练集校验。
- 增加 `test_batch_size`、`num_test_workers` 和 `test_dataloader`。

迁移要点：episode id 是每个 LeRobot repo 内部的局部 id。后面 mixed 配置里同名字段会应用到每个子数据集，子数据集也可以单独覆盖。

### 3. 固定时间窗与共享 LeRobotDataset

提交：`189888a feat(train): update LeRobot training workflow`

这一阶段把视频窗口改为固定 frame offsets：

- `frame_sampling.num_frames` 决定输入窗口长度。
- `_get_frame_offsets()` 默认返回 `[-num_frames + 1, ..., 0]`，即包含当前帧。
- `_get_delta_timestamps()` 用 `offset / fps` 传给 LeRobot。
- `LeRobotVideoDataset` 根据 episode 边界过滤有足够上下文的中心帧。
- 单数据集 datamodule 尽量复用同一个底层 `LeRobotDataset`，避免 train/val/test 反复初始化下载和缓存。

迁移要点：mixed 数据集必须按子数据集 fps 分别计算 `delta_timestamps`。不要在 mixed datamodule 顶层共享一个 fps。

### 4. 引入 MixedLeRobotDataModule

提交：`9baeb08 feat(data): add mixed LeRobot dataset loading`

这是混合数据集的主体提交，新增 `MixedLeRobotDataModule`：

- 配置入口改为 `dataset.name: MixedLeRobotDataModule`。
- `dataset.datasets` 是子数据集列表，每项至少包含 `repo_id`，通常还包含 `root` 和 `camera_key`。
- 每个子数据集独立读取 metadata，独立 resolve camera key、fps、split 和 `delta_timestamps`。
- 每个子数据集构造自己的 `LeRobotDataset` 和 `LeRobotVideoDataset`。
- train/val/test split 分别用 `torch.utils.data.ConcatDataset` 拼接。
- `return_dataset_index: true` 时，样本中额外带上该子数据集在 `dataset.datasets` 里的下标。
- `sampling.mode: natural` 保持拼接后的自然分布；`sampling.mode: weighted` 用 `WeightedRandomSampler` 按数据集权重重采样。

新增配置：

- `configs/videosaur/lerobot_mixed_dataset.yml`
- `configs/videosaur/lerobot_mixed_debug.yml`

迁移要点：Mixed 不复用单个 `LeRobotDataModule` 实例，而是把单数据集适配逻辑拆成可复用 helper，然后在 Mixed 内部逐个子数据集调用。这样可以保留每个子数据集的 camera/fps/split 差异。

### 5. slot cache、v2.1 本地数据和 provenance

提交：`186c052 feat(cache): add LeRobot slot feature caching`

这一阶段让 mixed 数据集可用于离线 slot feature cache：

- 加入 `LeRobotV2Metadata` 和 `LeRobotV2Dataset`，支持本地 LeRobot v2.1/LIBERO 数据，不改原始目录。
- 单数据集和 mixed datamodule 都暴露 `dataset_info`，记录 repo、root、camera、format、fps 和 split。
- 加入 `LeRobotSlotCacheDataset`，为每个原始帧生成稳定 cache row，并保存输入窗口上下文。
- `MixedLeRobotDataModule.cache_datasets()` 按 requested split 顺序、子数据集顺序、episode/frame 顺序生成 cache dataset。
- `cache_dataloaders()` 支持分布式顺序分片，避免 cache 时重复或漏行。
- `videosaur/cache_lerobot_slots.py` 写出 sharded npy feature、parquet index 和 `metadata.json`。

新增配置与文档：

- `configs/videosaur/Libero_slot_cache.yml`
- `docs/lerobot_workflow.md`
- `docs/lerobot_workflow_zh.md`

迁移要点：如果下游要根据策略数据重新查 slot feature，必须迁移 cache provenance 字段，尤其是 `dataset_index`、`repo_id`、`episode_index`、`frame_index`、`context_*`。

## 当前实现结构

主要代码在 `videosaur/data/datamodules.py`：

- `_resolve_lerobot_format()`：自动识别 `v2`/`v3`，v2.1 本地数据走适配层。
- `_try_load_lerobot_info_metadata()`：只读 `meta/info.json` 做轻量 metadata 规划，避免不必要地初始化完整数据集。
- `_episodes_or_none_if_all()`：当选择了所有 episode 时传 `None` 给 LeRobot，避免无意义过滤。
- `LeRobotImageTransform`：统一相机帧输入格式。
- `LeRobotV2Dataset`：把本地 v2.1 数据适配成训练/cache 需要的 v3-like item。
- `LeRobotVideoDataset`：把底层 LeRobot item 转成 `{"video": ...}`，并可附加 `dataset_index`。
- `LeRobotSlotCacheDataset`：面向 cache 的逐帧 dataset，带完整 provenance。
- `LeRobotDataModule`：单个 LeRobot repo。
- `MixedLeRobotDataModule`：多个 LeRobot repo 的训练、验证、测试和 cache 入口。

## 最小 mixed 配置模板

```yaml
dataset:
  name: MixedLeRobotDataModule
  datasets:
    - repo_id: ego_exo4d
      root: /path/to/ego_exo4d
      camera_key: observation.images.top_head
    - repo_id: something_something_v2
      root: /path/to/something_something_v2
      camera_key: observation.images.egocentric

  batch_size: 32
  val_batch_size: 32
  test_batch_size: 32
  num_workers: 0
  num_val_workers: 0
  num_test_workers: 0

  image_size: 224
  video_backend: pyav
  tolerance_s: 1e-3

  frame_sampling:
    num_frames: 4

  sampling:
    mode: natural
  return_dataset_index: true

  val_fraction: 0.05
  test_fraction: 0.0
```

LIBERO/v2.1 本地数据可以在单个子数据集上覆盖格式：

```yaml
dataset:
  name: MixedLeRobotDataModule
  datasets:
    - repo_id: Libero
      root: /path/to/libero_10_no_noops_lerobot
      lerobot_format: v2
      camera_key: observation.images.image
```

## 迁移步骤

1. 先迁移单数据集路径。
   搬 `LeRobotImageTransform`、`LeRobotVideoDataset`、`LeRobotDataModule` 以及 split 校验逻辑，确认单个 repo 能构建 train dataloader。

2. 再迁移 metadata 规划。
   搬 `_try_load_lerobot_info_metadata()` 和 `_episodes_or_none_if_all()`，确保只靠 `meta/info.json` 可以拿到 fps、camera keys 和 total episodes。

3. 固定时间窗。
   搬 `_get_frame_offsets()` 与 `_get_delta_timestamps()`。注意 mixed 时必须使用每个子数据集自己的 fps。

4. 迁移 `MixedLeRobotDataModule`。
   保留子数据集配置 normalize、每 repo 独立 split、每 repo 独立 dataset 构造、`ConcatDataset`、natural/weighted sampler 和 `dataset_index`。

5. 迁移配置。
   从 `configs/videosaur/lerobot_mixed_dataset.yml` 拷贝 `dataset` 段，再替换 `repo_id/root/camera_key`。

6. 如果需要 cache，迁移 cache 专用路径。
   搬 `LeRobotSlotCacheDataset`、`SequentialShardSampler`、`cache_datasets()`、`cache_dataloaders()`，以及 `videosaur/cache_lerobot_slots.py` 中的 writer/finalize/reader。

7. 最后迁移文档和最小读取脚本。
   `docs/lerobot_workflow_zh.md` 是运行手册；`scripts/read_slot_cache.py` 和 `scripts/read_slot_cache_minimal.py` 是 cache 检查入口。

## 验证清单

基础语法检查：

```bash
.venv/bin/python -m compileall \
  videosaur/data/datamodules.py \
  videosaur/cache_lerobot_slots.py \
  scripts/read_slot_cache.py \
  scripts/read_slot_cache_minimal.py
```

配置能否构建：

```bash
.venv/bin/python - <<'PY'
from videosaur import configuration, data

config = configuration.load_config("configs/videosaur/lerobot_mixed_dataset.yml")
dm = data.build(config.dataset)
print(dm)
PY
```

用少量 episode 做 dataloader smoke：

```bash
.venv/bin/python - <<'PY'
from videosaur import configuration, data

overrides = [
    "dataset.train_episodes=[0,1]",
    "dataset.val_episodes=null",
    "dataset.test_episodes=null",
    "dataset.batch_size=2",
    "dataset.num_workers=0",
]
config = configuration.load_config("configs/videosaur/lerobot_mixed_dataset.yml", overrides)
dm = data.build(config.dataset)
batch = next(iter(dm.train_dataloader()))
print(batch["video"].shape)
print(batch.get("dataset_index"))
PY
```

cache 相关测试：

```bash
.venv/bin/python -m pytest tests/test_cache_lerobot_slots.py
```

如果目标环境没有 `pytest`，至少运行 `compileall` 和一个本地 writer/finalize/reader smoke，确认 `features/`、`index.parquet`、`metadata.json` 都能生成并读取。

## 常见迁移坑

- 不同 LeRobot repo 的 camera key 往往不同，必须逐个配置或逐个自动选择。
- 不同 repo 的 fps 可能不同，`delta_timestamps` 必须按子数据集 fps 计算。
- `train_episodes`、`val_episodes`、`test_episodes` 是子数据集内部 episode id，不是 mixed 后的全局 id。
- `return_dataset_index` 建议保持 `true`，否则 cache 或下游 join 时很难区分同名 episode/frame 来自哪个 repo。
- `sampling.mode: natural` 会按真实样本量混合，数据大的 repo 权重自然更高；如果要控制比例，用 `weighted`。
- v2.1/LIBERO 适配需要本地 `root`，并依赖 parquet 和 LeRobot 的视频解码工具。
- cache row ordering 要保持稳定：requested split 顺序、子数据集顺序、episode 顺序、frame 顺序。迁移时不要为了并行方便打乱这个顺序。
- 对 OSS 或其他对象存储，优先使用 sharded npy + parquet index，不要迁移到单体 memmap 文件。
