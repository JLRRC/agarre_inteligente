import math
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------
#  Conversión parámetros <-> rectángulo / bbox
# ---------------------------------------------------------------------
def params_to_rect(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
) -> np.ndarray:
    """
    Convierte (cx, cy, w, h, angle_deg) a rectángulo (4 puntos).
    Devuelve rect: np.ndarray de shape (4, 2)
    """
    theta = math.radians(angle_deg)

    # Vector a lo largo del ancho (w)
    dx_w = (w / 2.0) * math.cos(theta)
    dy_w = (w / 2.0) * math.sin(theta)

    # Vector a lo largo de la altura (h), perpendicular al anterior
    dx_h = -(h / 2.0) * math.sin(theta)
    dy_h = (h / 2.0) * math.cos(theta)

    p0 = (cx - dx_w - dx_h, cy - dy_w - dy_h)
    p1 = (cx + dx_w - dx_h, cy + dy_w - dy_h)
    p2 = (cx + dx_w + dx_h, cy + dy_w + dy_h)
    p3 = (cx - dx_w + dx_h, cy - dy_w + dy_h)

    rect = np.array([p0, p1, p2, p3], dtype=np.float32)
    return rect


def rect_to_bbox(rect: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convierte un rectángulo (4 x 2) en caja eje-alineada:
        (xmin, ymin, xmax, ymax)
    """
    if rect.shape != (4, 2):
        raise ValueError(f"rect shape inválido: {rect.shape}, esperado (4, 2)")

    xs = rect[:, 0]
    ys = rect[:, 1]

    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())

    return xmin, ymin, xmax, ymax


# ---------------------------------------------------------------------
#  IoU de cajas y diferencia angular
# ---------------------------------------------------------------------
def bbox_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """
    IoU entre dos cajas (xmin, ymin, xmax, ymax).
    Devuelve un escalar en [0, 1].
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Área de intersección
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Áreas individuales
    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0

    return float(inter_area / union)


def angle_diff_deg(a1: float, a2: float) -> float:
    """
    Diferencia mínima entre dos ángulos en grados (resultado en [0, 180]).
    """
    diff = abs(a1 - a2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)


# ---------------------------------------------------------------------
#  Grasp success tipo Cornell
# ---------------------------------------------------------------------
def _to_numpy5(x) -> np.ndarray:
    """
    Convierte un tensor/lista/array de 5 elem. a np.ndarray [5].
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] != 5:
        raise ValueError(f"params debe tener 5 elementos, tiene {x.shape[0]}")
    return x


def compute_grasp_success(
    pred_params,
    gt_params,
    iou_thresh: float = 0.25,
    angle_thresh: float = 30.0,
) -> bool:
    """
    Criterio de grasp success tipo Cornell.

    Éxito si:
        IoU(bbox_pred, bbox_gt) >= iou_thresh
        y
        angle_diff_deg(pred_angle, gt_angle) <= angle_thresh
    """
    pred = _to_numpy5(pred_params)
    gt = _to_numpy5(gt_params)

    cx_p, cy_p, w_p, h_p, ang_p = map(float, pred)
    cx_g, cy_g, w_g, h_g, ang_g = map(float, gt)

    rect_p = params_to_rect(cx_p, cy_p, w_p, h_p, ang_p)
    rect_g = params_to_rect(cx_g, cy_g, w_g, h_g, ang_g)

    bbox_p = rect_to_bbox(rect_p)
    bbox_g = rect_to_bbox(rect_g)

    iou = bbox_iou(bbox_p, bbox_g)
    a_diff = angle_diff_deg(ang_p, ang_g)

    return (iou >= iou_thresh) and (a_diff <= angle_thresh)
