#!/usr/bin/env python3
import os, argparse
from datetime import datetime
import numpy as np
import cv2
from graspnet.datasets.cornell_dataset import CornellGraspDataset

def rect_corners(cx, cy, w, h, ang_deg):
    import math
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = w/2.0, h/2.0
    pts = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    out=[]
    for x,y in pts:
        out.append((x*ca - y*sa + cx, x*sa + y*ca + cy))
    return np.array(out, dtype=np.float32)

def draw_rect(img_bgr, corners, color):
    pts = corners.reshape(-1,1,2).astype(np.int32)
    cv2.polylines(img_bgr, [pts], True, color, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cornell-root", default="data/cornell_raw")
    ap.add_argument("--use-depth", action="store_true")
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    root = os.path.expanduser("~/TFM/agarre_inteligente")
    cornell_root = args.cornell_root
    if not os.path.isabs(cornell_root):
        cornell_root = os.path.join(root, cornell_root)

    ds = CornellGraspDataset(root_dir=cornell_root, split="val", use_depth=args.use_depth)

    out_dir = os.path.join(root, "experiments", "figures_cualitativa",
                           f"cornell_GT_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # exporta k muestras con GT (verde)
    for j in range(args.k):
        s = ds[j]
        rgb = s["rgb"]          # tensor o np
        y   = s["y"]            # grasp [cx,cy,w,h,angle,...]
        rgb_np = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
            rgb_np = np.transpose(rgb_np, (1,2,0))
        if rgb_np.max() <= 1.0:
            rgb_np = np.clip(rgb_np*255.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        yy = y.detach().cpu().numpy().reshape(-1) if hasattr(y, "detach") else np.asarray(y).reshape(-1)
        cx,cy,w,h,ang = map(float, yy[:5])
        gt = rect_corners(cx,cy,w,h,ang)
        draw_rect(img, gt, (0,255,0))
        cv2.putText(img, f"idx={j}", (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"gt_{j:02d}.png"), img)

    print("[OK] Export GT-only:", out_dir)

if __name__ == "__main__":
    main()
