from videosaur import inference


def test_load_inference_config_applies_dotlist_overrides(tmp_path):
    config_path = tmp_path / "inference.yml"
    config_path.write_text(
        "\n".join(
            [
                "checkpoint: checkpoint.ckpt",
                "model_config: model.yml",
                "n_slots: 5",
                "fps: 30",
                "input:",
                "  path: old_input.mp4",
                "  type: video",
                "output:",
                "  save_path: old_output.mp4",
            ]
        )
    )

    config = inference.load_inference_config(
        str(config_path),
        [
            "input.path=data/libero/sample.mp4",
            "output.save_path=inference_output/sample-mask.mp4",
            "output.interactive.enabled=true",
            "output.interactive.viewer_dir=inference_output/sample-viewer",
            "n_slots=7",
        ],
    )

    assert config.input.path == "data/libero/sample.mp4"
    assert config.output.save_path == "inference_output/sample-mask.mp4"
    assert config.output.interactive.enabled is True
    assert config.output.interactive.viewer_dir == "inference_output/sample-viewer"
    assert config.n_slots == 7
