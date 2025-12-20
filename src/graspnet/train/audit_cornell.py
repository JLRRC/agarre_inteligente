#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

# Import correcto (NO "src.")
from graspnet.datasets.cornell_dataset import CornellGraspDataset


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe config: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML inválido (no dict).")
    return cfg


def finite_mask_x(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W] -> mask [B]
    return torch.isfinite(x).view(x.size(0), -1).all(dim=1)


def finite_mask_y(y: torch.Tensor) -> torch.Tensor:
    # y: [B,5] -> mask [B]
    return torch.isfinite(y).all(dim=1)


def main():
    ap = argparse.ArgumentParser(description="Auditoría Cornell: detecta NaN/Inf y genera índices limpios.")
    ap.add_argument("--config", required=True, help="YAML de experimento (el mismo que train_cornell.py)")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Split a auditar")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--outdir", default="reports/cornell_audit", help="Carpeta de salida")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    root_dir = data_cfg["root_dir"]
    img_size = int(data_cfg.get("img_size", 224))
    val_split = float(data_cfg.get("val_split", 0.2))
    use_depth = bool(data_cfg.get("use_depth", False))

    # Augmentation SOLO para train si está activado en YAML
    aug_cfg = data_cfg.get("augmentation", {})
    train_aug = {
        "geometric": bool(aug_cfg.get("geometric", False)),
        "photometric": bool(aug_cfg.get("photometric", False)),
    }

    augmentation = train_aug if args.split == "train" else None

    ds = CornellGraspDataset(
        root_dir=root_dir,
        split=args.split,
        val_split=val_split,
        img_size=img_size,
        use_depth=use_depth,
        augmentation=augmentation,
    )

    # Wrapper para guardar el idx original
    class Wrap(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            sample = self.base[i]
            if isinstance(sample, dict):
                sample["_idx"] = i
                return sample
            raise TypeError(f"El dataset no devuelve dict. Tipo: {type(sample)}")

    wds = Wrap(ds)

    dl = DataLoader(
        wds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bad_rows = []
    clean_indices = []
    reasons = Counter()

    for batch_idx, batch in enumerate(dl):
        idxs = batch["_idx"].tolist()
        rgb = batch["rgb"]          # [B,3,H,W]
        grasp = batch["grasp"]      # [B,5]
        if use_depth:
            depth = batch["depth"]  # [B,1,H,W]
            x = torch.cat([rgb, depth], dim=1)
        else:
            depth = None
            x = rgb

        mx = finite_mask_x(x)
        my = finite_mask_y(grasp)

        # checks adicionales útiles
        # (no marcan como "bad" por sí solos, pero ayudan a diagnosticar)
        # w/h no finitos ya caen en my; aquí detectamos w/h <= 0
        wh_bad = (grasp[:, 2] <= 0) | (grasp[:, 3] <= 0)

        for j, idx in enumerate(idxs):
            ok_x = bool(mx[j].item())
            ok_y = bool(my[j].item())
            ok_wh = not bool(wh_bad[j].item())

            if ok_x and ok_y and ok_wh:
                clean_indices.append(int(idx))
                continue

            if not ok_x and not ok_y:
                reason = "nonfinite_x_and_y"
            elif not ok_x:
                reason = "nonfinite_x"
            elif not ok_y:
                reason = "nonfinite_y"
            else:
                reason = "nonpositive_w_or_h"

            reasons[reason] += 1

            row = {
                "idx": int(idx),
                "batch": int(batch_idx),
                "reason": reason,
                "rgb_min": float(torch.nan_to_num(rgb[j]).min().item()),
                "rgb_max": float(torch.nan_to_num(rgb[j]).max().item()),
                "grasp": grasp[j].detach().cpu().numpy().tolist(),
            }
            if use_depth and depth is not None:
                row["depth_min"] = float(torch.nan_to_num(depth[j]).min().item())
                row["depth_max"] = float(torch.nan_to_num(depth[j]).max().item())

            bad_rows.append(row)

    # outputs
    bad_csv = outdir / f"bad_{args.split}.csv"
    clean_txt = outdir / f"clean_idx_{args.split}.txt"
    summary_json = outdir / f"summary_{args.split}.json"

    if bad_rows:
        with bad_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=bad_rows[0].keys())
            writer.writeheader()
            writer.writerows(bad_rows)
    else:
        bad_csv.write_text("NO BAD SAMPLES\n", encoding="utf-8")

    clean_txt.write_text("\n".join(map(str, clean_indices)) + "\n", encoding="utf-8")

    summary = {
        "split": args.split,
        "use_depth": use_depth,
        "total": len(wds),
        "clean": len(clean_indices),
        "bad": len(bad_rows),
        "reasons": dict(reasons),
        "outdir": str(outdir),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Cornell audit DONE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"BAD CSV:  {bad_csv}")
    print(f"CLEAN IDX:{clean_txt}")
    print(f"SUMMARY: {summary_json}")


if __name__ == "__main__":
    main()
