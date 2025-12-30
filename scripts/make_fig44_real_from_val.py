#!/usr/bin/env python3
"""
make_fig44_real_from_val.py

Genera Figura 4.4 REAL desde el pipeline de validación (mismo preprocesado),
elige automáticamente una muestra "acierto" (success) si existe, o la mejor IoU.

Salida: Figura_4-4_Acierto_Cornell_IoU_DeltaTheta.png
"""

import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch

# Reutilizamos funciones del entrenamiento (mismo pipeline)
from graspnet.train.train_cornell import (
    load_config,
    make_dataloaders,
    make_model,
    _extract_model_output,
    angle_diff_deg,
)

def poly_from_rect(cx, cy, w, h, theta_deg):
    t = math.radians(theta_deg)
    ct, st = math.cos(t), math.sin(t)
    dx, dy = w/2.0, h/2.0
    corners = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    out=[]
    for x,y in corners:
        xr = cx + x*ct - y*st
        yr = cy + x*st + y*ct
        out.append((xr,yr))
    return out

def mask_from_poly(size_hw, poly):
    H,W = size_hw
    m = Image.new("L", (W,H), 0)
    dr = ImageDraw.Draw(m)
    dr.polygon([(float(x),float(y)) for x,y in poly], fill=1)
    return np.array(m, dtype=np.uint8)

def iou_from_polys(size_hw, poly_a, poly_b):
    ma = mask_from_poly(size_hw, poly_a)
    mb = mask_from_poly(size_hw, poly_b)
    inter = np.logical_and(ma,mb).sum()
    union = np.logical_or(ma,mb).sum()
    return float(inter)/(float(union)+1e-9)

def to_vis_rgb(x3chw: torch.Tensor):
    """Convierte tensor [3,H,W] a imagen 0..1 sin asumir mean/std."""
    x = x3chw.detach().cpu().float()
    # clamp por si viene normalizada
    x = x - x.min()
    denom = (x.max() - x.min()) + 1e-9
    x = x / denom
    x = x.permute(1,2,0).numpy()
    x = np.clip(x, 0.0, 1.0)
    return x

def find_ckpt_and_config(exp_root: Path):
    # preferir RGB (no RGBD), y evitar BORRAR/_legacy
    cands = []
    for p in exp_root.rglob("best.pth"):
        s = str(p).lower()
        if "/borrar/" in s or "/_legacy/" in s:
            continue
        cands.append(p)
    if not cands:
        raise FileNotFoundError("No encontré best.pth válido en experiments/ (sin BORRAR/_legacy).")
    # preferir exp con 'rgb' y no 'rgbd'
    def score(p: Path):
        name = p.parents[1].name.lower()  # experiments/<EXP>/checkpoints/best.pth
        sc = 0
        if "rgb" in name: sc += 10
        if "rgbd" in name: sc -= 20
        return sc
    cands = sorted(cands, key=lambda p: (score(p), p.stat().st_mtime), reverse=True)
    ckpt = cands[0]
    exp_dir = ckpt.parents[1]  # experiments/<EXP>
    # config_used.yaml es lo ideal (train_cornell lo copia)
    cfg = exp_dir / "config_used.yaml"
    if not cfg.exists():
        # fallback: busca cualquier yaml en el exp dir
        y = list(exp_dir.glob("*.yaml"))
        if y:
            cfg = y[0]
        else:
            raise FileNotFoundError(f"No encontré config_used.yaml ni yaml en {exp_dir}")
    return ckpt, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="experiments", help="Ruta a experiments/")
    ap.add_argument("--ckpt", default="", help="Ruta explícita a best.pth (opcional)")
    ap.add_argument("--config", default="", help="Ruta explícita a config yaml (opcional)")
    ap.add_argument("--out", required=True, help="Salida PNG")
    ap.add_argument("--max_samples", type=int, default=200, help="Máx muestras a escanear en val")
    a = ap.parse_args()

    root = Path(".").resolve()
    exp_root = (root / a.experiments).resolve()

    if a.ckpt and a.config:
        ckpt = Path(a.ckpt).resolve()
        cfg_path = Path(a.config).resolve()
    else:
        ckpt, cfg_path = find_ckpt_and_config(exp_root)

    print(f"[OK] CKPT={ckpt}")
    print(f"[OK] CFG ={cfg_path}")

    cfg = load_config(str(cfg_path))

    metrics_cfg = cfg.get("metrics", {})
    iou_thresh = float(metrics_cfg.get("iou_thresh", 0.25))
    ang_thresh = float(metrics_cfg.get("angle_thresh", 30.0))

    data_cfg = cfg.get("data", {})
    use_depth = bool(data_cfg.get("use_depth", False))
    img_size = int(data_cfg.get("img_size", 224))
    in_channels = 4 if use_depth else 3

    device = torch.device("cpu")
    train_loader, val_loader = make_dataloaders(cfg, use_depth=use_depth)
    model = make_model(cfg, device=device, in_channels=in_channels)
    model.eval()

    # cargar checkpoint (estructura igual que train_cornell: state["model"])
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 20:
        print("[WARN] Demasiadas claves missing -> probablemente ckpt incompatible (RGBD vs RGB, etc.)")

    best = None  # (success, iou, ang, rgb_img_np, gt, pred)

    seen = 0
    with torch.no_grad():
        for batch in val_loader:
            # batch puede ser dict o tuple
            if isinstance(batch, dict):
                rgb = batch["rgb"]
                grasp = batch["grasp"]
                depth = batch.get("depth", None)
            else:
                if len(batch) == 2:
                    rgb, grasp = batch
                    depth = None
                else:
                    rgb, depth, grasp = batch

            # construir input x según use_depth
            if use_depth:
                if depth is None:
                    continue
                x = torch.cat([rgb, depth], dim=1)
            else:
                x = rgb

            outputs_raw = model(x)
            outputs = _extract_model_output(outputs_raw)   # tensor [B,5]
            preds_np = outputs.detach().cpu().numpy()
            grasp_np = grasp.detach().cpu().numpy()

            B = preds_np.shape[0]
            for i in range(B):
                p = preds_np[i].astype(np.float64)
                g = grasp_np[i].astype(np.float64)

                # asegurar w/h positivos como hace train_cornell
                p[2] = max(float(abs(p[2])), 1e-6)
                p[3] = max(float(abs(p[3])), 1e-6)

                pred_poly = poly_from_rect(p[0], p[1], p[2], p[3], p[4])
                gt_poly   = poly_from_rect(g[0], g[1], g[2], g[3], g[4])

                iou = iou_from_polys((img_size, img_size), gt_poly, pred_poly)
                ang = angle_diff_deg(float(p[4]), float(g[4]))
                success = (iou >= iou_thresh) and (ang <= ang_thresh)

                rgb_img = to_vis_rgb(rgb[i][:3, :, :])

                cand = (success, iou, ang, rgb_img, gt_poly, pred_poly)
                if best is None:
                    best = cand
                else:
                    # preferir success, luego mayor IoU, luego menor ang
                    if cand[0] and not best[0]:
                        best = cand
                    elif cand[0] == best[0]:
                        if (cand[1] > best[1]) or (cand[1] == best[1] and cand[2] < best[2]):
                            best = cand

                seen += 1
                if best and best[0] and seen >= 5:
                    # si ya hay un success pronto, no hace falta recorrer mucho
                    pass

                if seen >= a.max_samples and best is not None:
                    break

            if seen >= a.max_samples and best is not None:
                break

    if best is None:
        raise RuntimeError("No pude evaluar ninguna muestra de val.")

    success, iou, ang, rgb_img, gt_poly, pred_poly = best
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6), dpi=220)
    ax = fig.add_subplot(111)
    ax.imshow(rgb_img)
    ax.set_title("Figura 4.4 — Ejemplo visual de “acierto” según Cornell: IoU y Δθ",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    def draw(poly, color, ls, label):
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.5, label=label)

    draw(gt_poly,  "lime", "--", "GT")
    draw(pred_poly, "red", "-",  "Pred")

    ax.text(0.02, 0.02,
            f"IoU = {iou:.3f}   |   Δθ = {ang:.2f}°   |   success = {success}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black"))

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] OUT={out}")
    print(f"[OK] success={success} iou={iou:.4f} ang={ang:.2f} (th_iou={iou_thresh}, th_ang={ang_thresh})")

if __name__ == "__main__":
    main()
