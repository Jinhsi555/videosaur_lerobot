import argparse
import torch
import torch.nn.functional as F
from torchvision.io import read_video, read_image
from omegaconf import OmegaConf
from videosaur import configuration, models
from videosaur.data.transforms import CropResize, Normalize, Resize, build_inference_transform
from videosaur.data.datamodules import LeRobotImageTransform
from videosaur.interactive_export import export_interactive_viewer, interactive_enabled
import os
import numpy as np
import imageio
from torchvision import transforms as tvt
from videosaur.visualizations import (
    mix_inputs_with_masks,
    mix_inputs_with_masks_and_slot_overlay,
    draw_segmentation_masks_on_image,
    color_map,
)
import matplotlib.pyplot as plt


def load_inference_config(config_path: str, overrides=None):
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config


def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, config


def _resolve_device(config):
    device_name = str(config.get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`device: cuda` was requested, but torch.cuda.is_available() is False.")
    return device


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _get_transform_style(transform_config) -> str:
    if not transform_config:
        return "default"
    return str(transform_config.get("style", "default")).lower()


def _get_input_size(transform_config):
    size = transform_config.get("input_size", 224) if transform_config else 224
    if isinstance(size, int):
        return (size, size)
    if len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Expected int or pair for input size, got {size}")


def _prepare_video_visualization(video: torch.Tensor, size):
    video_vis = torch.as_tensor(video)
    if video_vis.ndim != 4:
        raise ValueError(f"Expected video tensor with shape [T,H,W,C], got {tuple(video_vis.shape)}")

    if video_vis.shape[-1] in (1, 3):
        video_vis = video_vis.permute(0, 3, 1, 2)
    elif video_vis.shape[1] not in (1, 3):
        raise ValueError(f"Expected RGB video tensor, got shape {tuple(video_vis.shape)}")

    needs_rescale = not torch.is_floating_point(video_vis)
    video_vis = video_vis.float()
    if needs_rescale or video_vis.max() > 2.0:
        video_vis = video_vis / 255.0

    if video_vis.shape[-2:] != size:
        video_vis = F.interpolate(video_vis, size=size, mode="bilinear", align_corners=False)

    return video_vis.permute(1, 0, 2, 3)


def _prepare_lerobot_video(video: torch.Tensor, transform_config):
    image_transform = LeRobotImageTransform(size=transform_config.get("input_size", 224))
    video_vis = _prepare_video_visualization(video, image_transform.size)
    video = image_transform(video)

    inputs = {
        "video": video.unsqueeze(0),
        "video_visualization": video_vis.unsqueeze(0),
    }
    return inputs


def prepare_video(video_path: str, transform_config=None):
    # Load video
    video, _, _ = read_video(video_path, pts_unit="sec")

    if _get_transform_style(transform_config) == "lerobot":
        return _prepare_lerobot_video(video, transform_config)

    video = video.float() / 255.0
    #change size of the video to 224x224
    video_vis = video.permute(0, 3, 1, 2)
    video_vis = tvt.Resize(_get_input_size(transform_config))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)

    
    if transform_config:
        tfs = build_inference_transform(transform_config)
        video = video.permute(3, 0, 1, 2)
        video = tfs(video).permute(1, 0, 2, 3)
    else:
        video = video.permute(0, 3, 1, 2)
     # Add batch dimension
    inputs = {"video": video.unsqueeze(0), 
              "video_visualization": video_vis.unsqueeze(0)}
    return inputs


def prepare_image(image_path: str, transfom_config=None):
    image = read_image(image_path)
    image = image.float() / 255.0
    resize = CropResize(dataset_type="image", 
                        crop_type="short_side_resize_central", 
                        size=transfom_config.input_size, 
                        resize_mode="bilinear")
    image_vis =resize(image)
    
    if transfom_config:
        tfs = build_inference_transform(transfom_config)
        image = tfs(image)
     # Add batch dimension
    inputs = {"image": image.unsqueeze(0), 
              "image_visualization": image_vis.unsqueeze(0)}
    return inputs



def main(config):
    # Load the model from checkpoint
    
    device = _resolve_device(config)
    print(f"Inference device: {device}")
    model, model_config = load_model_from_checkpoint(config.checkpoint, config.model_config)
    model.initializer.n_slots = config.n_slots
    model = model.to(device)
    # Prepare the video dict
    if config.input.type == "video":
        prepare_inputs = prepare_video
    elif config.input.type == "image":
        prepare_inputs = prepare_image
        
    inputs = prepare_inputs(config.input.path, config.input.transforms)
    inputs = _move_to_device(inputs, device)
    # Perform inference
    with torch.inference_mode():
        outputs = model(inputs)
        aux_outputs = model.aux_forward(inputs, outputs)
    inputs_cpu = None
    outputs_cpu = None
    aux_outputs_cpu = None
    if config.input.type == "video" and (
        config.output.get("save_path") or interactive_enabled(config)
    ):
        inputs_cpu = _move_to_device(inputs, torch.device("cpu"))
        outputs_cpu = _move_to_device(outputs, torch.device("cpu"))
        aux_outputs_cpu = _move_to_device(aux_outputs, torch.device("cpu"))

    if config.input.type=="video" and config.output.save_path:
        # Save the results
        save_dir = os.path.dirname(config.output.save_path)
        os.makedirs(save_dir, exist_ok=True)
        layout = str(config.output.get("layout", "mask_grid"))
        if layout == "mask_grid":
            masked_video_frames = mix_inputs_with_masks(inputs_cpu, outputs_cpu)
        elif layout == "mask_grid_with_slot_overlay":
            masked_video_frames = mix_inputs_with_masks_and_slot_overlay(
                inputs_cpu,
                outputs_cpu,
                aux_outputs_cpu,
                alpha=float(config.output.get("slot_overlay_alpha", 0.75)),
            )
        else:
            raise ValueError(f"Unknown video output layout: {layout}")
        with imageio.get_writer(config.output.save_path, fps=config.fps) as writer:
            for frame in masked_video_frames:
                writer.append_data(frame)
        writer.close()
    if config.input.type == "video" and interactive_enabled(config):
        viewer_dir = export_interactive_viewer(
            config,
            model_config,
            inputs_cpu,
            outputs_cpu,
            aux_outputs_cpu,
        )
        print(f"Interactive viewer: {viewer_dir}")
    elif config.input.type=="image" and config.output.save_path:
        save_dir = os.path.dirname(config.output.save_path)
        os.makedirs(save_dir, exist_ok=True)
        inputs_cpu = _move_to_device(inputs, torch.device("cpu"))
        aux_outputs_cpu = _move_to_device(aux_outputs, torch.device("cpu"))
        masks = aux_outputs_cpu["decoder_masks_hard"][0].bool()
        cmap = color_map(masks.shape[0])
        image = (inputs_cpu["image_visualization"]*256)[0].type(torch.uint8)
        mixed_image = draw_segmentation_masks_on_image(image, masks, colors=cmap)
        # Save the results
        plt.imsave(config.output.save_path, mixed_image.permute(1, 2, 0).numpy())
    print("Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on a video or image.")
    parser.add_argument(
        "--config",
        default="configs/inference/movi_c.yml",
        help="Configuration to run",
    )
    parser.add_argument("config_overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()
    config = load_inference_config(args.config, args.config_overrides)
    main(config)
