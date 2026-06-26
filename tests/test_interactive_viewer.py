import http.client
import json
import threading

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from videosaur.interactive_export import (
    compute_frame_heatmaps,
    export_interactive_viewer,
    resolve_prediction_dims,
)
from videosaur.interactive_viewer import create_server


def _softmax(values):
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def _make_model_config(pred_dims=(2, 6), threshold=None, temperature=1.0):
    transform = {
        "name": "utils.FeatureTimeSimilarity",
        "softmax": True,
        "temperature": temperature,
    }
    if threshold is not None:
        transform["threshold"] = threshold
    return OmegaConf.create(
        {
            "model": {
                "losses": {
                    "loss_timesim": {
                        "name": "CrossEntropyLoss",
                        "pred_dims": list(pred_dims),
                        "target_transform": transform,
                    }
                }
            }
        }
    )


def _make_export_fixture(tmp_path):
    viewer_dir = tmp_path / "viewer"
    config = OmegaConf.create(
        {
            "fps": 12,
            "input": {"type": "video", "path": "demo.mp4"},
            "output": {
                "save_path": str(tmp_path / "demo-mask.mp4"),
                "interactive": {
                    "enabled": True,
                    "viewer_dir": str(viewer_dir),
                    "feature_key": "encoder.vit_block_keys12",
                    "prediction_key": "decoder.reconstruction",
                },
            },
        }
    )

    video = torch.linspace(0, 1, 1 * 3 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 3, 8, 8)
    features = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    prediction = torch.zeros(1, 3, 4, 6, dtype=torch.float32)
    prediction[0, 0, 0, 2:6] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    prediction[0, 0, 1, 2:6] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    prediction[0, 1, 0, 2:6] = torch.tensor([0.5, 0.0, -0.5, -1.0])

    soft_masks = torch.zeros(1, 3, 2, 8, 8, dtype=torch.float32)
    soft_masks[:, :, 0, :, :4] = 0.8
    soft_masks[:, :, 0, :, 4:] = 0.2
    soft_masks[:, :, 1, :, :4] = 0.2
    soft_masks[:, :, 1, :, 4:] = 0.8
    hard_masks = torch.zeros_like(soft_masks)
    hard_masks[:, :, 0, :, :4] = 1
    hard_masks[:, :, 1, :, 4:] = 1

    inputs = {"video_visualization": video}
    outputs = {
        "encoder": {"vit_block_keys12": features},
        "decoder": {
            "reconstruction": prediction,
            "masks": soft_masks.flatten(-2),
        },
    }
    aux_outputs = {
        "decoder_masks": soft_masks,
        "decoder_masks_vis_hard": hard_masks,
    }
    model_config = _make_model_config(pred_dims=(2, 6), threshold=None, temperature=1.0)
    return config, model_config, inputs, outputs, aux_outputs, viewer_dir


def test_export_interactive_viewer_writes_expected_assets(tmp_path):
    config, model_config, inputs, outputs, aux_outputs, viewer_dir = _make_export_fixture(tmp_path)

    export_interactive_viewer(config, model_config, inputs, outputs, aux_outputs)

    manifest_path = viewer_dir / "manifest.json"
    assert manifest_path.is_file()
    assert (viewer_dir / "index.html").is_file()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_frames"] == 3
    assert manifest["n_slots"] == 2
    assert manifest["patch_grid"] == [2, 2]
    assert manifest["atlas_tile_size"] == [2, 2]
    assert manifest["atlas_grid"] == [2, 2]
    assert manifest["prediction_dims"] == [2, 6]
    assert manifest["prediction_frames"] == 2
    assert manifest["prediction_remove_last_n_frames"] == 1
    assert manifest["prediction_target_offset"] == 1
    assert (viewer_dir / manifest["frame_paths"][0]).is_file()
    assert (viewer_dir / manifest["mask_soft_paths"][0][0]).is_file()
    assert (viewer_dir / manifest["mask_hard_paths"][0][1]).is_file()
    assert (viewer_dir / manifest["slot_overlay_paths"]["all"][0]).is_file()
    assert (viewer_dir / manifest["slot_overlay_paths"]["slots"][0][1]).is_file()
    assert set(manifest["heatmap_atlas_paths"]) == {
        "feature",
        "target",
        "prediction",
        "difference",
    }
    assert set(manifest["heatmap_atlas_fixed_paths"]) == {
        "target",
        "prediction",
        "difference",
    }
    assert manifest["heatmap_fixed_scale"] == {
        "probability_max": 0.4,
        "difference_abs_max": 0.25,
    }
    assert (viewer_dir / manifest["heatmap_atlas_paths"]["feature"][0]).is_file()
    assert (viewer_dir / manifest["heatmap_atlas_paths"]["target"][0]).is_file()
    assert (viewer_dir / manifest["heatmap_atlas_paths"]["prediction"][0]).is_file()
    assert (viewer_dir / manifest["heatmap_atlas_paths"]["difference"][0]).is_file()
    assert (viewer_dir / manifest["heatmap_atlas_fixed_paths"]["target"][0]).is_file()
    assert (viewer_dir / manifest["heatmap_atlas_fixed_paths"]["prediction"][0]).is_file()
    assert (viewer_dir / manifest["heatmap_atlas_fixed_paths"]["difference"][0]).is_file()
    assert "arrays" not in manifest

    atlas = imageio.imread(viewer_dir / manifest["heatmap_atlas_paths"]["feature"][0])
    assert atlas.shape[:2] == (4, 4)
    fixed_atlas = imageio.imread(viewer_dir / manifest["heatmap_atlas_fixed_paths"]["prediction"][0])
    assert fixed_atlas.shape[:2] == (4, 4)
    overlay = imageio.imread(viewer_dir / manifest["slot_overlay_paths"]["all"][0])
    assert overlay.shape == (8, 8, 4)


def test_resolve_prediction_dims_prefers_timesim_loss_config():
    model_config = _make_model_config(pred_dims=(3, 7))

    assert resolve_prediction_dims({}, model_config, prediction_dim=9, n_patches=4) == (3, 7)


@pytest.mark.parametrize(
    ("prediction_dim", "n_patches", "expected"),
    [
        (4, 4, (0, 4)),
        (8, 4, (4, 8)),
        (5, 4, (0, 4)),
    ],
)
def test_resolve_prediction_dims_fallbacks(prediction_dim, n_patches, expected):
    model_config = OmegaConf.create({"model": {"losses": None}})

    assert resolve_prediction_dims({}, model_config, prediction_dim, n_patches) == expected


def test_compute_frame_heatmaps_matches_similarity_prediction_and_difference(tmp_path):
    config, model_config, inputs, outputs, aux_outputs, viewer_dir = _make_export_fixture(tmp_path)
    export_interactive_viewer(config, model_config, inputs, outputs, aux_outputs)

    features = outputs["encoder"]["vit_block_keys12"][0].numpy()
    logits = outputs["decoder"]["reconstruction"][0, :2, :, 2:6].numpy()
    manifest = json.loads((viewer_dir / "manifest.json").read_text())
    heatmaps = compute_frame_heatmaps(features, logits, frame=0, similarity_config=manifest["similarity"])

    expected_similarity = np.array([1.0, 0.0, -1.0, 2**-0.5], dtype=np.float32)
    assert np.allclose(heatmaps["feature"][0], expected_similarity, atol=1e-6)
    assert np.allclose(heatmaps["target"][0], _softmax(expected_similarity), atol=1e-6)

    expected_prediction = _softmax([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(heatmaps["prediction"][0], expected_prediction, atol=1e-6)

    expected_compare = expected_prediction - _softmax(expected_similarity)
    assert np.allclose(heatmaps["difference"][0], expected_compare, atol=1e-6)


def test_interactive_viewer_http_smoke(tmp_path):
    config, model_config, inputs, outputs, aux_outputs, viewer_dir = _make_export_fixture(tmp_path)
    export_interactive_viewer(config, model_config, inputs, outputs, aux_outputs)
    server = create_server(viewer_dir, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 200
        html = response.read()
        assert b"VideoSAUR Interactive Viewer" in html
        assert b"/api/similarity" not in html
        assert b"/api/prediction" not in html
        assert b"/api/compare" not in html

        conn.request("GET", "/manifest.json")
        response = conn.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        atlas_path = "/" + payload["heatmap_atlas_fixed_paths"]["prediction"][0]

        conn.request("GET", atlas_path)
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/png"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
