#!/usr/bin/env python3
# Summary: Runs SimpleGraspCNN on an RGB image and writes grasp params to JSON.
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import torch

def _ensure_graspnet_on_path() -> None:
    vision_dir = os.environ.get("VISION_DIR", "~/TFM/agarre_inteligente")
    vision_dir = os.path.expanduser(vision_dir)
    src_dir = os.path.join(vision_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_ensure_graspnet_on_path()

from graspnet.models.simple_cnn import SimpleGraspCNN
from graspnet.models.resnet18_grasp import ResNet18Grasp


def load_image_rgb(path: Path, img_size: int = 224, roi: Optional[Tuple[int, int, int]] = None):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"No se pudo leer la imagen: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = rgb.shape
    roi_info = None
    if roi:
        roi_cx, roi_cy, roi_size = roi
        roi_size = int(max(1, roi_size))
        roi_size = int(min(roi_size, orig_w, orig_h))
        cx = int(round(max(0, min(orig_w - 1, roi_cx))))
        cy = int(round(max(0, min(orig_h - 1, roi_cy))))
        x0 = int(round(cx - roi_size / 2.0))
        y0 = int(round(cy - roi_size / 2.0))
        x0 = max(0, min(orig_w - roi_size, x0))
        y0 = max(0, min(orig_h - roi_size, y0))
        x1 = x0 + roi_size
        y1 = y0 + roi_size
        rgb = rgb[y0:y1, x0:x1]
        roi_info = (x0, y0, x1 - x0, y1 - y0)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype("float32") / 255.0
    x = x.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(x).unsqueeze(0), orig_w, orig_h, roi_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--roi-cx", type=int, default=None)
    parser.add_argument("--roi-cy", type=int, default=None)
    parser.add_argument("--roi-size", type=int, default=None)
    args = parser.parse_args()

    img_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    roi = None
    if args.roi_size is not None and args.roi_cx is not None and args.roi_cy is not None:
        if args.roi_size > 0:
            roi = (args.roi_cx, args.roi_cy, args.roi_size)
    x, orig_w, orig_h, roi_info = load_image_rgb(img_path, img_size=args.img_size, roi=roi)

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
    if roi_info:
        roi_x, roi_y, roi_w, roi_h = roi_info
        scale_x = roi_w / float(args.img_size)
        scale_y = roi_h / float(args.img_size)
    else:
        roi_x = 0
        roi_y = 0
        roi_w = orig_w
        roi_h = orig_h
        scale_x = orig_w / float(args.img_size)
        scale_y = orig_h / float(args.img_size)
    cx *= scale_x
    cy *= scale_y
    w *= scale_x
    h *= scale_y
    cx += float(roi_x)
    cy += float(roi_y)

    out = {
        "cx": float(cx),
        "cy": float(cy),
        "w": float(w),
        "h": float(h),
        "angle_deg": float(angle_deg),
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
    }
    if roi_info:
        out["roi"] = {
            "x": int(roi_x),
            "y": int(roi_y),
            "w": int(roi_w),
            "h": int(roi_h),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
