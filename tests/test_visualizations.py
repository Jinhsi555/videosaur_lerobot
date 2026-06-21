import numpy as np
import torch

from videosaur.visualizations import color_map, mix_inputs_with_masks_and_slot_overlay


def _make_video_overlay_inputs(n_slots=5, n_frames=2, height=224, width=224):
    video = torch.zeros(1, 3, n_frames, height, width)
    masks = torch.zeros(1, n_frames, n_slots, height, width)

    for slot_idx in range(n_slots):
        start = slot_idx * width // n_slots
        end = (slot_idx + 1) * width // n_slots
        masks[:, :, slot_idx, :, start:end] = 1

    inputs = {"video_visualization": video}
    outputs = {"decoder": {"masks": masks.flatten(-2)}}
    aux_outputs = {"decoder_masks_vis_hard": masks.bool()}
    return inputs, outputs, aux_outputs


def test_mix_inputs_with_masks_and_slot_overlay_shape_and_dtype():
    inputs, outputs, aux_outputs = _make_video_overlay_inputs(n_slots=4, n_frames=3)

    frames = mix_inputs_with_masks_and_slot_overlay(inputs, outputs, aux_outputs)

    grid_height = 2 * 224 + 2
    grid_width = 6 * (224 + 2) - 2
    assert len(frames) == 3
    for frame in frames:
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.shape[0] == grid_height
        assert frame.shape[1] > grid_width + 224


def test_mix_inputs_with_masks_and_slot_overlay_uses_one_color_per_slot():
    n_slots = 5
    inputs, outputs, aux_outputs = _make_video_overlay_inputs(n_slots=n_slots, n_frames=1)

    frame = mix_inputs_with_masks_and_slot_overlay(
        inputs, outputs, aux_outputs, alpha=1.0
    )[0]

    grid_width = 6 * (224 + 2) - 2
    overlay = frame[:, grid_width + 2 :, :]
    for color in color_map(n_slots):
        assert np.any(np.all(overlay == np.array(color, dtype=np.uint8), axis=-1))


def test_mix_inputs_with_masks_and_slot_overlay_falls_back_to_decoder_masks():
    inputs, outputs, _ = _make_video_overlay_inputs(n_slots=3, n_frames=2)

    frames = mix_inputs_with_masks_and_slot_overlay(inputs, outputs, aux_outputs=None)

    assert len(frames) == 2
    assert all(frame.dtype == np.uint8 for frame in frames)
