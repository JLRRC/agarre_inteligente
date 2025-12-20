#!/usr/bin/env python3
"""
make_day2_report.py

Genera:
  - Tabla bonita (Markdown) para memoria del trabajo, a partir de experiments/summary_base.csv
  - 2 gráficas por experimento (train/val loss y val_success vs epoch) a partir de metrics.csv

Uso:
  python src/graspnet/metrics/make_day2_report.py --root experiments

Requisitos:
  - experiments/<EXP_NAME>/metrics.csv
  - experiments/summary_base.csv  (si no existe, el script igual genera plots; tabla saldrá vacía)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _read_metrics(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    # Normaliza columnas esperadas
    for c in ["epoch", "train_loss", "val_loss", "val_iou", "val_angle", "val_success"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _find_experiments(root: Path) -> list[Path]:
    exps = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "metrics.csv").exists():
            exps.append(d)
    return exps


def _best_epoch_from_metrics(df: pd.DataFrame) -> int | None:
    if df.empty:
        return None
    if "val_success" in df.columns and df["val_success"].notna().any():
        d2 = df.dropna(subset=["val_success"]).copy()
        if d2.empty:
            return None
        # best = max val_success, tie-break: max val_iou, min val_angle, min val_loss
        for c in ["val_iou", "val_angle", "val_loss"]:
            if c not in d2.columns:
                d2[c] = float("nan")
        d2["val_iou"] = d2["val_iou"].fillna(-1.0)
        d2["val_angle"] = d2["val_angle"].fillna(float("inf"))
        d2["val_loss"] = d2["val_loss"].fillna(float("inf"))

        d2 = d2.sort_values(
            by=["val_success", "val_iou", "val_angle", "val_loss"],
            ascending=[False, False, True, True],
        )
        return int(d2.iloc[0]["epoch"])
    # fallback: max val_iou
    if "val_iou" in df.columns and df["val_iou"].notna().any():
        d2 = df.dropna(subset=["val_iou"]).sort_values("val_iou", ascending=False)
        return int(d2.iloc[0]["epoch"])
    return None


def _plot_experiment(exp_dir: Path, out_dir: Path) -> None:
    metrics_path = exp_dir / "metrics.csv"
    df = _read_metrics(metrics_path)
    if df.empty or "epoch" not in df.columns:
        print(f"[WARN] {exp_dir.name}: metrics.csv vacío o sin 'epoch'")
        return

    # Plot 1: train/val loss
    plt.figure()
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.title(f"{exp_dir.name} — Loss vs Epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{exp_dir.name}__loss.png", dpi=160)
    plt.close()

    # Plot 2: val_success
    if "val_success" in df.columns and df["val_success"].notna().any():
        plt.figure()
        plt.plot(df["epoch"], df["val_success"], label="val_success")
        best_ep = _best_epoch_from_metrics(df)
        if best_ep is not None:
            plt.axvline(best_ep, linestyle="--", label=f"best_epoch={best_ep}")
        plt.title(f"{exp_dir.name} — Val Success vs Epoch")
        plt.xlabel("epoch")
        plt.ylabel("val_success")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{exp_dir.name}__val_success.png", dpi=160)
        plt.close()


def _make_markdown_table(summary_csv: Path) -> str:
    if not summary_csv.exists():
        return "> [WARN] No existe experiments/summary_base.csv (tabla no generada)\n"

    df = pd.read_csv(summary_csv)

    # Columnas típicas de tu summary_base.csv actual
    wanted = [
        "exp_id", "model", "modality", "augment", "best_epoch",
        "val_success", "val_iou", "val_angle", "val_loss",
        "ckpt_best", "ckpt_last",
    ]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()

    # Orden “bonito”: por val_success desc (si existe)
    if "val_success" in df.columns:
        df["val_success"] = pd.to_numeric(df["val_success"], errors="coerce")
        df = df.sort_values("val_success", ascending=False)

    # Redondeos de métricas
    for c in ["val_success", "val_iou", "val_angle", "val_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").map(
                lambda x: f"{x:.6f}" if _is_finite(x) else ""
            )

    # Tabla Markdown
    return df.to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="experiments", help="Carpeta experiments/")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    summary_csv = root / "summary_base.csv"

    out_dir = root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Tabla para memoria
    md = _make_markdown_table(summary_csv)
    md_path = root / "summary_base_pretty.md"
    md_path.write_text(md + "\n", encoding="utf-8")
    print(f"[OK] Tabla Markdown: {md_path}")

    # 2) Plots por experimento
    exp_dirs = _find_experiments(root)
    print(f"[INFO] Experimentos encontrados: {len(exp_dirs)}")
    for exp in exp_dirs:
        _plot_experiment(exp, out_dir)

    print(f"[OK] Plots en: {out_dir}")


if __name__ == "__main__":
    main()
