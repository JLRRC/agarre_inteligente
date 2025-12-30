#!/usr/bin/env python3
# Summary: Runs SimpleGraspCNN on an RGB image and writes grasp params to JSON.
import argparse
import json
from pathlib import Path

import cv2
import torch

from graspnet.models.simple_cnn import SimpleGraspCNN
from graspnet.models.resnet18_grasp import ResNet18Grasp


def load_image_rgb(path: Path, img_size: int = 224):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"No se pudo leer la imagen: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = rgb.shape
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype("float32") / 255.0
    x = x.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(x).unsqueeze(0), orig_w, orig_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    img_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    x, orig_w, orig_h = load_image_rgb(img_path, img_size=args.img_size)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    keys = list(state_dict.keys())
    if any(k.startswith("backbone.") for k in keys):
        conv_key = "backbone.conv1.weight"
        in_ch = 3
        if conv_key in state_dict and hasattr(state_dict[conv_key], "shape"):
            try:
                in_ch = int(state_dict[conv_key].shape[1])
            except Exception:
                in_ch = 3
        if in_ch != 3:
            raise RuntimeError(f"Checkpoint requiere {in_ch} canales (RGBD). infer_grasp_rgb es solo RGB.")
        model = ResNet18Grasp(in_channels=in_ch, pretrained=False)
    elif any(k.startswith("features.") for k in keys):
        conv_key = "features.0.weight"
        in_ch = 3
        if conv_key in state_dict and hasattr(state_dict[conv_key], "shape"):
            try:
                in_ch = int(state_dict[conv_key].shape[1])
            except Exception:
                in_ch = 3
        if in_ch != 3:
            raise RuntimeError(f"Checkpoint requiere {in_ch} canales (RGBD). infer_grasp_rgb es solo RGB.")
        model = SimpleGraspCNN(in_channels=in_ch, img_size=args.img_size)
    else:
        raise RuntimeError("Checkpoint incompatible: no reconozco keys de modelo.")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy().tolist()

    cx, cy, w, h, angle_deg = pred
    scale_x = orig_w / float(args.img_size)
    scale_y = orig_h / float(args.img_size)
    cx *= scale_x
    cy *= scale_y
    w *= scale_x
    h *= scale_y

    out = {
        "cx": float(cx),
        "cy": float(cy),
        "w": float(w),
        "h": float(h),
        "angle_deg": float(angle_deg),
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
