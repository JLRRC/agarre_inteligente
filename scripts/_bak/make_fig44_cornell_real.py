#!/usr/bin/env python3
import argparse, math, glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch

from graspnet.models.simple_cnn import SimpleGraspCNN


def find_gt_pos_file(img_path: Path) -> Path:
    d = img_path.parent
    base = img_path.stem  # pcdXXXXr

    candidates = []
    candidates += [d / (base.replace("r", "cpos") + ".txt")]
    candidates += [d / (base.replace("r", "pos") + ".txt")]
    # búsqueda amplia
    candidates += [Path(p) for p in glob.glob(str(d / (base.replace("r", "") + "*pos*.txt")))]
    candidates += [Path(p) for p in glob.glob(str(d / ("*" + base.replace("r", "") + "*pos*.txt")))]

    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No encuentro GT pos/cpos para {img_path}")


def read_first_grasp_polygon(pos_file: Path):
    pts = []
    with pos_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0]); y = float(parts[1])
                    pts.append((x, y))
                except:
                    pass
            if len(pts) >= 4:
                break
    if len(pts) < 4:
        raise ValueError(f"GT inválido en {pos_file}")
    return pts[:4]


def poly_from_rect(cx, cy, w, h, theta_deg):
    t = math.radians(theta_deg)
    ct, st = math.cos(t), math.sin(t)
    dx, dy = w/2.0, h/2.0
    corners = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    out = []
    for x, y in corners:
        xr = cx + x*ct - y*st
        yr = cy + x*st + y*ct
        out.append((xr, yr))
    return out


def rect_params_from_poly(pts):
    p0, p1, p2, p3 = pts
    cx = sum(p[0] for p in pts)/4.0
    cy = sum(p[1] for p in pts)/4.0
    w = math.dist(p0, p1)
    h = math.dist(p1, p2)
    theta = math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))
    return cx, cy, w, h, theta


def angle_diff_deg(a, b):
    d = abs(a-b) % 180.0
    return d if d <= 90.0 else 180.0 - d


def mask_from_poly(size_hw, poly):
    H, W = size_hw
    m = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(m)
    dr.polygon([(float(x), float(y)) for x, y in poly], fill=1)
    return np.array(m, dtype=np.uint8)


def iou_from_polys(size_hw, poly_a, poly_b):
    ma = mask_from_poly(size_hw, poly_a)
    mb = mask_from_poly(size_hw, poly_b)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter) / (float(union) + 1e-9)


def is_normalized(v: np.ndarray) -> bool:
    # Heurística: si está “cerca” de [0,1], asumimos normalizado
    return (np.nanmax(np.abs(v)) <= 1.5)


def decode_pred(y: torch.Tensor, img_size: int, default_h: float):
    """
    Soporta:
      - Mapas densos: [B,C,H,W] (GG-CNN style)
      - Regresión:    [B,K]
    Devuelve (cx, cy, w, h, theta_deg)
    """
    y = y.detach().cpu()
    if isinstance(y, torch.Tensor) and y.ndim == 4:
        # [B,C,H,W] GG-CNN style
        y_np = y[0].numpy()
        C, H, W = y_np.shape
        if C >= 4:
            q = y_np[0]
            c = y_np[1]
            s = y_np[2]
            wmap = y_np[3]
            ang = 0.5 * np.degrees(np.arctan2(s, c))
        else:
            raise RuntimeError(f"Salida densa inesperada C={C}")

        yy, xx = np.unravel_index(np.argmax(q), q.shape)
        cx, cy = float(xx), float(yy)
        theta = float(ang[yy, xx])
        w = float(wmap[yy, xx])
        h = float(default_h)
        return cx, cy, w, h, theta

    if isinstance(y, torch.Tensor) and y.ndim == 2:
        # [B,K] regresión directa
        v = y[0].numpy().astype(np.float64)
        K = v.shape[0]

        # Debug: imprime K y rango (esto te dirá si está normalizado)
        print(f"[DEBUG] output ndim=2, K={K}, min={np.min(v):.4f}, max={np.max(v):.4f}")

        # Caso 5: cx,cy,w,h,theta
        if K >= 5:
            cx, cy, w, h, theta = v[0], v[1], v[2], v[3], v[4]
        # Caso 4: cx,cy,w,theta
        elif K == 4:
            cx, cy, w, theta = v[0], v[1], v[2], v[3]
            h = default_h
        # Caso 6 típico: cx,cy,cos,sin,w,h (o similar)
        elif K == 6:
            cx, cy, c, s, w, h = v[0], v[1], v[2], v[3], v[4], v[5]
            theta = 0.5 * math.degrees(math.atan2(s, c))
        else:
            raise RuntimeError(f"No sé decodificar salida K={K}. Imprime v y lo ajusto.")

        # Si está normalizado, escalar a píxeles
        if is_normalized(np.array([cx, cy])):
            cx *= img_size
            cy *= img_size
        if is_normalized(np.array([w, h])):
            w *= img_size
            h *= img_size

        return float(cx), float(cy), float(w), float(h), float(theta)

    raise RuntimeError(f"Salida modelo no soportada: type={type(y)} ndim={getattr(y,'ndim',None)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Imagen Cornell *r.png")
    ap.add_argument("--ckpt", required=True, help="best.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--default-h", type=float, default=30.0)
    a = ap.parse_args()

    img_path = Path(a.image)
    pos_path = find_gt_pos_file(img_path)

    # Load image
    img0 = Image.open(img_path).convert("RGB")
    W0, H0 = img0.size

    img = img0.resize((a.img_size, a.img_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # Load model
    model = SimpleGraspCNN()
    sd = torch.load(a.ckpt, map_location="cpu")
    if isinstance(sd, dict) and any(k in sd for k in ["model", "state_dict"]):
        sd = sd.get("model", sd.get("state_dict", sd))
    model.load_state_dict(sd, strict=False)
    model.eval()

    with torch.no_grad():
        y = model(x)

    # Normaliza y si viene en dict/tuple
    if isinstance(y, (list, tuple)):
        y = y[0]
    if isinstance(y, dict):
        y = next(iter(y.values()))

    pred_cx, pred_cy, pred_w, pred_h, pred_theta = decode_pred(y, a.img_size, a.default_h)
    pred_poly = poly_from_rect(pred_cx, pred_cy, pred_w, pred_h, pred_theta)

    # GT → reescala a resized
    gt_poly0 = read_first_grasp_polygon(pos_path)
    sx = a.img_size / float(W0)
    sy = a.img_size / float(H0)
    gt_poly = [(x*sx, y*sy) for x, y in gt_poly0]
    gt_cx, gt_cy, gt_w, gt_h, gt_theta = rect_params_from_poly(gt_poly)

    iou = iou_from_polys((a.img_size, a.img_size), gt_poly, pred_poly)
    dth = angle_diff_deg(gt_theta, pred_theta)

    # Plot
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_title("Figura 4.4 — Acierto según Cornell (IoU y Δθ)", fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    def draw_poly(poly, color, ls, label):
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, color=color, linewidth=2.5, linestyle=ls, label=label)

    draw_poly(gt_poly,  "lime", "--", "GT")
    draw_poly(pred_poly, "red", "-",  "Pred")

    ax.text(0.02, 0.02, f"IoU = {iou:.3f}   |   Δθ = {dth:.1f}°",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black"))
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] OUT={out}")
    print(f"[OK] image={img_path}")
    print(f"[OK] gt={pos_path}")
    print(f"[OK] IoU={iou:.4f}  dtheta={dth:.2f}")


if __name__ == "__main__":
    main()
