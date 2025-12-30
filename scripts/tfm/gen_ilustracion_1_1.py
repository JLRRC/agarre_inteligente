#!/usr/bin/env python3
# Summary: Build Illustration 1-1 from (a) Gazebo capture, (b) Cornell RGB-D with grasp rectangle, (c) real clutter photo.
import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    torch = None

try:
    from graspnet.datasets.cornell_dataset import CornellGraspDataset
except Exception as exc:
    CornellGraspDataset = None


def _latest_file(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _find_latest(patterns: List[str], root: Path) -> Optional[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    return _latest_file(matches)


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"No se pudo leer la imagen: {path}")
    return img


def _rect_points(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = w / 2.0
    dy = h / 2.0
    pts = np.array(
        [
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy],
        ],
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


def _cornell_panel(root_dir: Path) -> np.ndarray:
    if CornellGraspDataset is None:
        raise RuntimeError("CornellGraspDataset no disponible (graspnet no importable).")
    ds = CornellGraspDataset(
        root_dir=str(root_dir),
        split="train",
        img_size=224,
        random_grasp=False,
        use_depth=True,
    )
    sample = ds[0]
    rgb = sample["rgb"].numpy().transpose(1, 2, 0)
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = sample["depth"].numpy().squeeze()
    depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    cx, cy, w, h, angle = sample["grasp"].numpy().tolist()
    pts = _rect_points(cx, cy, w, h, angle)
    rgb_rect = rgb.copy()
    depth_rect = depth.copy()
    rgb_axes = rgb.copy()
    depth_axes = depth.copy()
    for img in (rgb_rect, depth_rect, rgb_axes, depth_axes):
        cv2.polylines(img, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
    _draw_axes(rgb_axes, cx, cy, w, h, angle)
    _draw_axes(depth_axes, cx, cy, w, h, angle)
    cv2.putText(rgb_rect, "Imagen RGB", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(depth_rect, "Imagen de profundidad", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    top = _hstack_with_gap(rgb_rect, depth_rect, 6)
    bottom = _hstack_with_gap(rgb_axes, depth_axes, 6)
    panel = _vstack_with_gap(top, bottom, 6)
    caption = "Ejemplo del Cornell Grasping Dataset: anotaciones de rectangulos orientados"
    return _add_caption(panel, caption)


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] == width:
        return img
    canvas = np.full((img.shape[0], width, 3), 255, dtype=np.uint8)
    canvas[:, : img.shape[1]] = img
    return canvas


def _hstack_with_gap(left: np.ndarray, right: np.ndarray, gap: int) -> np.ndarray:
    target_h = max(left.shape[0], right.shape[0])
    left = _resize_to_height(left, target_h)
    right = _resize_to_height(right, target_h)
    canvas = np.full((target_h, left.shape[1] + right.shape[1] + gap, 3), 255, dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] + gap:] = right
    return canvas


def _vstack_with_gap(top: np.ndarray, bottom: np.ndarray, gap: int) -> np.ndarray:
    width = max(top.shape[1], bottom.shape[1])
    top = _pad_to_width(top, width)
    bottom = _pad_to_width(bottom, width)
    canvas = np.full((top.shape[0] + bottom.shape[0] + gap, width, 3), 255, dtype=np.uint8)
    canvas[: top.shape[0], :] = top
    canvas[top.shape[0] + gap:, :] = bottom
    return canvas


def _add_caption(panel: np.ndarray, text: str) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = text_h + 10
    canvas = np.full((panel.shape[0] + pad, panel.shape[1], 3), 255, dtype=np.uint8)
    canvas[: panel.shape[0], :] = panel
    x = max(6, (panel.shape[1] - text_w) // 2)
    y = panel.shape[0] + text_h + 4
    cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def _compose_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    target_h = max(a.shape[0], b.shape[0], c.shape[0])
    a = _resize_to_height(a, target_h)
    b = _resize_to_height(b, target_h)
    c = _resize_to_height(c, target_h)
    gap = 8
    border = 2
    total_w = a.shape[1] + b.shape[1] + c.shape[1] + gap * 2 + border * 2
    total_h = target_h + border * 2
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    x = border
    canvas[border:border + target_h, x:x + a.shape[1]] = a
    x += a.shape[1] + gap
    canvas[border:border + target_h, x:x + b.shape[1]] = b
    x += b.shape[1] + gap
    canvas[border:border + target_h, x:x + c.shape[1]] = c
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera Ilustracion 1-1 (a,b,c).")
    parser.add_argument("--out", required=True, help="Salida PNG")
    parser.add_argument("--ws-root", default=os.environ.get("WS_DIR", "~/TFM/agarre_ros2_ws"))
    parser.add_argument("--gazebo", default="")
    parser.add_argument("--real", default="")
    parser.add_argument("--cornell-panel", default="", help="Imagen manual para el panel Cornell (b).")
    parser.add_argument("--cornell-root", default="data/cornell_raw")
    args = parser.parse_args()

    ws_root = Path(args.ws_root).expanduser().resolve()
    manual_dir = ws_root / "reports" / "tfm_evidencias" / "manual"
    figures_dir = ws_root / "experiments" / "figures_memoria"

    if args.gazebo:
        a_path = Path(args.gazebo).expanduser().resolve()
    else:
        a_path = _find_latest(["ilustracion_1_1_a_gazebo.*"], manual_dir) if manual_dir.exists() else None
        if a_path is None and figures_dir.exists():
            a_path = _find_latest(
                ["*Mesa*.png", "*Mesa*.jpg", "*Mesa*.jpeg", "*camera_overhead*.png", "*camera_overhead*.jpg"],
                figures_dir,
            )

    if args.real:
        c_path = Path(args.real).expanduser().resolve()
    else:
        c_path = _find_latest(["ilustracion_1_1_c_real.*"], manual_dir) if manual_dir.exists() else None

    if a_path is None or not a_path.exists():
        raise SystemExit("FALTA (a): captura Gazebo Mesa no disponible.")
    if c_path is None or not c_path.exists():
        raise SystemExit("FALTA (c): foto real no disponible. Sube una imagen.")

    a_img = _load_image(a_path)
    c_img = _load_image(c_path)

    if args.cornell_panel:
        b_path = Path(args.cornell_panel).expanduser().resolve()
    else:
        b_path = _find_latest(["ilustracion_1_1_b_cornell.*"], manual_dir) if manual_dir.exists() else None
    if b_path and b_path.exists():
        b_panel = _load_image(b_path)
    else:
        cornell_root = Path(args.cornell_root)
        if not cornell_root.is_dir():
            raise SystemExit(f"FALTA (b): dataset Cornell no encontrado en {cornell_root}")
        b_panel = _cornell_panel(cornell_root)

    composite = _compose_three(a_img, b_panel, c_img)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), composite)
    if not ok:
        raise SystemExit(f"No se pudo guardar: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
