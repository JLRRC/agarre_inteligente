#!/usr/bin/env python3
# Summary: Genera Ilustración 1-2 (RGB + Depth) con rectángulo orientado y parámetros.
import argparse
from pathlib import Path
import math
import cv2
import numpy as np

try:
    from graspnet.datasets.cornell_dataset import CornellGraspDataset
except Exception:
    CornellGraspDataset = None


def _rect_points(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = w / 2.0
    dy = h / 2.0
    pts = np.array(
        [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]],
        dtype=np.float32,
    )
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts = pts @ rot.T
    pts += np.array([cx, cy], dtype=np.float32)
    return pts.astype(np.int32)


def _draw_axes(img: np.ndarray, cx: float, cy: float, w: float, h: float, angle_deg: float) -> None:
    angle = math.radians(angle_deg)
    vx = math.cos(angle)
    vy = math.sin(angle)
    wx = vx * (w / 2.0)
    wy = vy * (w / 2.0)
    hx = -vy * (h / 2.0)
    hy = vx * (h / 2.0)
    p0 = (int(cx - wx), int(cy - wy))
    p1 = (int(cx + wx), int(cy + wy))
    p2 = (int(cx - hx), int(cy - hy))
    p3 = (int(cx + hx), int(cy + hy))
    cv2.line(img, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, p2, p3, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), 4, (0, 255, 0), -1)
    cv2.putText(img, "w", (int(cx + wx + 4), int(cy + wy + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img, "h", (int(cx + hx + 4), int(cy + hy + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _make_panel(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    gap = 6
    h = max(rgb.shape[0], depth.shape[0])
    def _resize(img):
        if img.shape[0] == h:
            return img
        scale = h / float(img.shape[0])
        return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
    rgb = _resize(rgb)
    depth = _resize(depth)
    canvas = np.full((h, rgb.shape[1] + depth.shape[1] + gap, 3), 255, dtype=np.uint8)
    canvas[:, : rgb.shape[1]] = rgb
    canvas[:, rgb.shape[1] + gap:] = depth
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera Ilustracion 1-2 (RGB/Depth con rectangulo orientado).")
    parser.add_argument("--out", required=True, help="Salida PNG")
    parser.add_argument("--cornell-root", default="data/cornell_raw", help="Ruta Cornell raw")
    args = parser.parse_args()

    if CornellGraspDataset is None:
        raise SystemExit("graspnet no disponible (CornellGraspDataset no importable).")
    root = Path(args.cornell_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Cornell root no existe: {root}")

    ds = CornellGraspDataset(
        root_dir=str(root),
        split="train",
        img_size=224,
        random_grasp=False,
        use_depth=True,
    )
    sample = ds[0]
    rgb = (sample["rgb"].numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = sample["depth"].numpy().squeeze()
    depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    cx, cy, w, h, angle = sample["grasp"].numpy().tolist()
    pts = _rect_points(cx, cy, w, h, angle)
    cv2.polylines(rgb, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.polylines(depth, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    _draw_axes(rgb, cx, cy, w, h, angle)
    _draw_axes(depth, cx, cy, w, h, angle)
    cv2.putText(rgb, "Imagen RGB", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(depth, "Depth image", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panel = _make_panel(rgb, depth)
    cv2.putText(panel, "(a)", (panel.shape[1] // 4, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(panel, "(b)", (panel.shape[1] * 3 // 4, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), panel):
        raise SystemExit(f"No se pudo guardar: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
