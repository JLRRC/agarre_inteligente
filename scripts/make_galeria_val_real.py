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
    H,W = size_hw
    m = Image.new("L",(W,H),0)
    ImageDraw.Draw(m).polygon([(float(x),float(y)) for x,y in poly], fill=1)
    return np.array(m, dtype=np.uint8)

def iou_from_polys(size_hw, a, b):
    ma = mask_from_poly(size_hw,a); mb = mask_from_poly(size_hw,b)
    inter = np.logical_and(ma,mb).sum()
    union = np.logical_or(ma,mb).sum()
    return float(inter)/(float(union)+1e-9)

def to_vis_rgb(x3chw):
    x = x3chw.detach().cpu().float()
    x = x - x.min()
    x = x / ((x.max()-x.min())+1e-9)
    return np.clip(x.permute(1,2,0).numpy(),0,1)

def draw_poly(ax, poly, color, ls):
    xs=[p[0] for p in poly]+[poly[0][0]]
    ys=[p[1] for p in poly]+[poly[0][1]]
    ax.plot(xs,ys,color=color,linestyle=ls,linewidth=2.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=4000)
    ap.add_argument("--n_success", type=int, default=4)
    ap.add_argument("--n_fail", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config(a.config)
    metrics_cfg = cfg.get("metrics",{})
    iou_th = float(metrics_cfg.get("iou_thresh",0.25))
    ang_th = float(metrics_cfg.get("angle_thresh",30.0))
    data_cfg = cfg.get("data",{})
    use_depth = bool(data_cfg.get("use_depth",False))
    img_size = int(data_cfg.get("img_size",224))
    in_channels = 4 if use_depth else 3

    device = torch.device("cpu")
    _, val_loader = make_dataloaders(cfg, use_depth=use_depth)
    model = make_model(cfg, device=device, in_channels=in_channels)
    state = torch.load(a.ckpt, map_location="cpu")
    sd = state["model"] if isinstance(state,dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    good=[]; bad=[]
    seen=0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch,dict):
                rgb=batch["rgb"]; grasp=batch["grasp"]; depth=batch.get("depth",None)
            else:
                if len(batch)==2: rgb,grasp=batch; depth=None
                else: rgb,depth,grasp=batch

            x = torch.cat([rgb,depth],dim=1) if use_depth else rgb
            out = _extract_model_output(model(x))
            p_np = out.cpu().numpy()
            g_np = grasp.cpu().numpy()

            for i in range(p_np.shape[0]):
                p = p_np[i].astype(np.float64); g=g_np[i].astype(np.float64)
                p[2]=max(abs(p[2]),1e-6); p[3]=max(abs(p[3]),1e-6)

                pred = poly_from_rect(p[0],p[1],p[2],p[3],p[4])
                gt   = poly_from_rect(g[0],g[1],g[2],g[3],g[4])

                iou = iou_from_polys((img_size,img_size), gt, pred)
                ang = angle_diff_deg(float(p[4]), float(g[4]))
                success = (iou>=iou_th) and (ang<=ang_th)

                img = to_vis_rgb(rgb[i][:3,:,:])
                item = (img,gt,pred,iou,ang,success)

                if success and len(good)<a.n_success:
                    good.append(item)
                elif (not success) and len(bad)<a.n_fail:
                    bad.append(item)

                seen += 1
                if seen>=a.max_samples:
                    break
                if len(good)>=a.n_success and len(bad)>=a.n_fail:
                    break
            if seen>=a.max_samples or (len(good)>=a.n_success and len(bad)>=a.n_fail):
                break

    panels = good + bad
    if not panels:
        raise RuntimeError("No encontré ejemplos para la galería.")

    cols=4
    rows=(len(panels)+cols-1)//cols
    fig=plt.figure(figsize=(12, 3*rows), dpi=220)

    for idx,(img,gt,pred,iou,ang,success) in enumerate(panels, start=1):
        ax=fig.add_subplot(rows, cols, idx)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        draw_poly(ax,gt,"lime","--"); draw_poly(ax,pred,"red","-")
        ax.set_title(f"{'OK' if success else 'FAIL'} | IoU={iou:.2f} Δθ={ang:.1f}°", fontsize=10)

    fig.suptitle("Galería cualitativa (val): GT vs Pred (Cornell) — ejemplos OK y FAIL",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] OUT={out} | ok={len(good)} fail={len(bad)} scanned={seen}")

if __name__=="__main__":
    main()
