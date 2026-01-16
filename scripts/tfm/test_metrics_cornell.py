#!/usr/bin/env python3
"""Quick checks for Cornell metrics (IoU, angle diff, success)."""
from __future__ import annotations

import math
import sys

from graspnet.utils.metrics import angle_diff_deg, compute_grasp_success, grasp_iou


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    pred = [112.0, 112.0, 60.0, 30.0, 0.0]
    gt_same = [112.0, 112.0, 60.0, 30.0, 0.0]
    iou_same = grasp_iou(pred, gt_same)
    _check(abs(iou_same - 1.0) < 1e-6, f"IoU identidad != 1.0 ({iou_same})")
    _check(angle_diff_deg(0.0, 0.0) == 0.0, "angle_diff 0 vs 0 != 0")
    _check(compute_grasp_success(pred, gt_same), "success identidad deberia ser True")

    gt_rot = [112.0, 112.0, 60.0, 30.0, 90.0]
    iou_rot = grasp_iou(pred, gt_rot)
    _check(0.0 <= iou_rot <= 1.0, f"IoU fuera de rango: {iou_rot}")
    _check(angle_diff_deg(0.0, 90.0) == 90.0, "angle_diff 0 vs 90 != 90")
    _check(not compute_grasp_success(pred, gt_rot), "success con 90 deg deberia ser False")

    print("[OK] Cornell metrics quick test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
