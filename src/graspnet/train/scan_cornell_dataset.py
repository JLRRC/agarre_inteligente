#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from graspnet.datasets.cornell_dataset import CornellGraspDataset


def load_cfg(cfg_path: str) -> dict:
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe config: {p}")
    return yaml.safe_load(p.read_text())


def load_idx_file(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe index_file: {p}")
    idx = [int(x.strip()) for x in p.read_text().splitlines() if x.strip()]
    if not idx:
        raise ValueError(f"index_file vacío: {p}")
    return idx


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def finite_report(arr: np.ndarray):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    ok = bool(finite.all())
    if ok:
        return ok, 0, 0, float(np.nanmin(arr)), float(np.nanmax(arr))
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    # min/max solo en finitos
    fin_vals = arr[np.isfinite(arr)]
    mn = float(fin_vals.min()) if fin_vals.size else float("nan")
    mx = float(fin_vals.max()) if fin_vals.size else float("nan")
    return ok, n_nan, n_inf, mn, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML de config (ej. config/exp2_simple_rgb_augment.yaml)")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Split a escanear")
    ap.add_argument("--use_train_aug", action="store_true",
                    help="Si se activa, usa augmentations de TRAIN (según YAML) incluso si split=val")
    ap.add_argument("--max_bad_print", type=int, default=20, help="Cuántos casos malos imprimir")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    data = cfg["data"]

    root_dir = str(data["root_dir"])
    img_size = int(data.get("img_size", 224))
    val_split = float(data.get("val_split", 0.2))
    use_depth = bool(data.get("use_depth", False))

    aug_cfg = data.get("augmentation", {}) or {}
    train_aug = {
        "geometric": bool(aug_cfg.get("geometric", False)),
        "photometric": bool(aug_cfg.get("photometric", False)),
    }

    # Para escanear: por defecto NO aug; si pides --use_train_aug, sí.
    augmentation = train_aug if args.use_train_aug else None

    ds_base = CornellGraspDataset(
        root_dir=root_dir,
        split=args.split,
        val_split=val_split,
        img_size=img_size,
        use_depth=use_depth,
        augmentation=augmentation,
    )

    # Subset limpio (Opción B PRO) si aplica
    idx_cfg = (data.get("index_files", {}) or {})
    idx_file = str(idx_cfg.get(args.split, "")).strip()
    if idx_file:
        idx = load_idx_file(idx_file)
        ds = Subset(ds_base, idx)
        subset = True
        print(f"[INFO] Subset {args.split.upper()}: base={len(ds_base)} -> subset={len(ds)} (idx_file={idx_file})")
    else:
        ds = ds_base
        subset = False
        print(f"[INFO] Dataset {args.split.upper()} SIN Subset: len={len(ds)}")

    print(f"[INFO] root_dir={root_dir} | split={args.split} | use_depth={use_depth} | img_size={img_size} | aug={augmentation} | subset={subset}")

    bad = []
    for i in range(len(ds)):
        s = ds[i]

        # Soporta dict o tuple/list
        if isinstance(s, dict):
            rgb = s.get("rgb")
            depth = s.get("depth", None)
            grasp = s.get("grasp")
        else:
            if len(s) == 2:
                rgb, grasp = s
                depth = None
            elif len(s) == 3:
                rgb, depth, grasp = s
            else:
                raise ValueError(f"Muestra con formato raro en i={i}: type={type(s)} len={len(s)}")

        rgb_np = to_np(rgb)
        ok_rgb, nan_rgb, inf_rgb, mn_rgb, mx_rgb = finite_report(rgb_np)

        ok_d, nan_d, inf_d, mn_d, mx_d = True, 0, 0, float("nan"), float("nan")
        if use_depth:
            if depth is None:
                ok_d = False
            else:
                depth_np = to_np(depth)
                ok_d, nan_d, inf_d, mn_d, mx_d = finite_report(depth_np)

        grasp_np = to_np(grasp)
        ok_g, nan_g, inf_g, mn_g, mx_g = finite_report(grasp_np)

        if not (ok_rgb and ok_d and ok_g):
            bad.append(i)
            if len(bad) <= args.max_bad_print:
                print(
                    f"[BAD] i={i} | "
                    f"rgb(ok={ok_rgb}, nan={nan_rgb}, inf={inf_rgb}, min={mn_rgb:.3g}, max={mx_rgb:.3g}) | "
                    f"depth(ok={ok_d}, nan={nan_d}, inf={inf_d}, min={mn_d:.3g}, max={mx_d:.3g}) | "
                    f"grasp(ok={ok_g}, nan={nan_g}, inf={inf_g}, min={mn_g:.3g}, max={mx_g:.3g})"
                )

    print(f"[DONE] total={len(ds)} | total_bad={len(bad)}")
    if bad:
        out = Path("reports/cornell_audit") / f"scan_bad_{args.split}{'_aug' if args.use_train_aug else ''}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(map(str, bad)) + "\n")
        print(f"[SAVE] índices malos guardados en: {out}")


if __name__ == "__main__":
    main()
