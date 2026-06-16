from videosaur.data.datamodules import MixedLeRobotDataModule

dm = MixedLeRobotDataModule(
    datasets=[
        {
            "repo_id": "ego_exo4d",
            "root": "/mnt/oss_data/lerobotv3_processed/ego_exo4d",
            "camera_key": "observation.images.top_head",
        },
        {
            "repo_id": "something_something_v2",
            "root": "/mnt/oss_data/lerobotv3_processed/something_something_v2",
            "camera_key": "observation.images.egocentric",
        },
    ],
    frame_sampling={"num_frames": 16},
    batch_size=4,
    num_workers=0,
    image_size=224,
    video_backend="pyav",
    val_fraction=0.05,
    test_fraction=0.05,
)
loader = dm.train_dataloader()
batch = next(iter(loader))
print(batch["video"].shape)
print(batch["dataset_index"].shape)
print([info["fps"] for info in dm.dataset_info])