#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch

from graspnet.train.train_cornell import (
    load_config, make_dataloaders, make_model, _extract_model_output, angle_diff_deg
)

def poly_from_rect(cx, cy, w, h, theta_deg):
    t = math.radians(theta_deg); ct, st = math.cos(t), math.sin(t)
    dx, dy = w/2.0, h/2.0
    corners = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    return [(cx + x*ct - y*st, cy + x*st + y*ct) for x,y in corners]

def mask_from_poly(size_hw, poly):
    H,W=size_hw
    m=Image.new("L",(W,H),0)
    ImageDraw.Draw(m).polygon([(float(x),float(y)) for x,y in poly], fill=1)
    return np.array(m, dtype=np.uint8)

def iou_from_polys(size_hw, a, b):
    ma=mask_from_poly(size_hw,a); mb=mask_from_poly(size_hw,b)
    inter=np.logical_and(ma,mb).sum()
    union=np.logical_or(ma,mb).sum()
    return float(inter)/(float(union)+1e-9)

def to_vis_rgb(x3chw):
    x=x3chw.detach().cpu().float()
    x=x-x.min(); x=x/((x.max()-x.min())+1e-9)
    return np.clip(x.permute(1,2,0).numpy(),0,1)

def clamp_rect(p, img_size):
    # cx,cy
    p[0]=float(max(0.0, min(img_size-1.0, p[0])))
    p[1]=float(max(0.0, min(img_size-1.0, p[1])))
    # w,h
    p[2]=float(max(2.0, min(img_size, abs(p[2]))))
    p[3]=float(max(2.0, min(img_size, abs(p[3]))))
    # theta a [-90,90)
    p[4]=float(((p[4] + 90.0) % 180.0) - 90.0)
    return p

def maybe_rad_to_deg(theta):
    # si parece radianes
    if abs(theta) <= 3.5:
        return float(theta * 180.0 / math.pi)
    return float(theta)

def align_pred_to_gt(p, g, img_size):
    """
    Heurística:
    - Si GT está en píxeles (cx>5) y pred está muy pequeña (|cx|<2, |w|<2),
      interpretamos pred como normalizada y la pasamos a píxeles.
    """
    p = p.astype(np.float64).copy()
    g = g.astype(np.float64).copy()

    # ángulo: rad->deg si toca
    p[4]=maybe_rad_to_deg(p[4])
    g[4]=maybe_rad_to_deg(g[4])

    gt_pixel_like = (abs(g[0]) > 5.0 or abs(g[1]) > 5.0)
    pred_small = (abs(p[0]) < 2.0 and abs(p[1]) < 2.0 and abs(p[2]) < 2.0 and abs(p[3]) < 2.0)

    if gt_pixel_like and pred_small:
        # Asumimos cx,cy en [-1,1] alrededor del centro; w,h en [0,1] relativo
        p[0] = (img_size*0.5) + (p[0]*img_size)
        p[1] = (img_size*0.5) + (p[1]*img_size)
        p[2] = abs(p[2]) * img_size
        p[3] = abs(p[3]) * img_size

    return clamp_rect(p, img_size), clamp_rect(g, img_size)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=8000)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Override DataLoader num_workers (0 evita multiprocessing).")
    a=ap.parse_args()

    cfg=load_config(a.config)
    cfg.setdefault("train", {})
    cfg["train"]["num_workers"] = int(a.num_workers)
    metrics_cfg=cfg.get("metrics",{})
    iou_th=float(metrics_cfg.get("iou_thresh",0.25))
    ang_th=float(metrics_cfg.get("angle_thresh",30.0))
    data_cfg=cfg.get("data",{})
    use_depth=bool(data_cfg.get("use_depth",False))
    img_size=int(data_cfg.get("img_size",224))
    in_channels=4 if use_depth else 3

    device=torch.device("cpu")
    _, val_loader = make_dataloaders(cfg, use_depth=use_depth)
    model=make_model(cfg, device=device, in_channels=in_channels)
    state=torch.load(a.ckpt, map_location="cpu")
    sd=state["model"] if isinstance(state,dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    best_success=None
    best_any=None
    seen=0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch,dict):
                rgb=batch["rgb"]; grasp=batch["grasp"]; depth=batch.get("depth",None)
            else:
                if len(batch)==2: rgb,grasp=batch; depth=None
                else: rgb,depth,grasp=batch

            x=torch.cat([rgb,depth],dim=1) if use_depth else rgb
            out=_extract_model_output(model(x))
            p_np=out.cpu().numpy()
            g_np=grasp.cpu().numpy()

            for i in range(p_np.shape[0]):
                p,g = align_pred_to_gt(p_np[i], g_np[i], img_size)
                pred_poly=poly_from_rect(p[0],p[1],p[2],p[3],p[4])
                gt_poly  =poly_from_rect(g[0],g[1],g[2],g[3],g[4])
                iou=iou_from_polys((img_size,img_size), gt_poly, pred_poly)
                ang=angle_diff_deg(float(p[4]), float(g[4]))
                success=(iou>=iou_th) and (ang<=ang_th)

                img=to_vis_rgb(rgb[i][:3,:,:])
                cand=(success,iou,ang,img,gt_poly,pred_poly)

                if best_any is None or (iou>best_any[1]) or (iou==best_any[1] and ang<best_any[2]):
                    best_any=cand
                if success:
                    if (best_success is None) or (iou>best_success[1]) or (iou==best_success[1] and ang<best_success[2]):
                        best_success=cand

                seen += 1
                if seen>=a.max_samples:
                    break
            if seen>=a.max_samples:
                break

    pick = best_success if best_success is not None else best_any
    success,iou,ang,img,gt_poly,pred_poly = pick

    outp=Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)

    fig=plt.figure(figsize=(8,6), dpi=220)
    ax=fig.add_subplot(111)
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])

    def draw(poly,color,ls,label):
        xs=[p[0] for p in poly]+[poly[0][0]]
        ys=[p[1] for p in poly]+[poly[0][1]]
        ax.plot(xs,ys,color=color,linestyle=ls,linewidth=2.5,label=label)

    draw(gt_poly,"lime","--","GT")
    draw(pred_poly,"red","-","Pred")

    ax.set_title("Figura 4.4 — Ejemplo visual de “acierto” según Cornell: IoU y Δθ",
                 fontsize=12, fontweight="bold")
    ax.text(0.02,0.02,f"IoU={iou:.3f}   |   Δθ={ang:.2f}°   |   success={success}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] OUT={outp}")
    print(f"[OK] success={success} iou={iou:.4f} ang={ang:.2f} (th_iou={iou_th}, th_ang={ang_th})")
    if best_success is None:
        print("[WARN] No se encontró success=True en el rango; se guardó el mejor IoU disponible.")

if __name__=="__main__":
    main()
