#!/usr/bin/env python3
"""
summarize_by_seed.py (PRO)

- Lee experiments/summary_base.csv (salida de select_best_epoch.py)
- Agrupa SOLO experimentos con sufijo: _seedN
- Calcula media±std por experimento base para:
    val_success, val_iou, val_angle, val_loss
- Genera:
    experiments/summary_by_seed.csv
    experiments/summary_by_seed.md

Uso:
  python src/graspnet/metrics/summarize_by_seed.py \
    --input experiments/summary_base.csv \
    --out_csv experiments/summary_by_seed.csv \
    --out_md  experiments/summary_by_seed.md
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SEED_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")
METRICS = ["val_success", "val_iou", "val_angle", "val_loss"]


def to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def fmt_pm(m: float, s: float, nd: int) -> str:
    if not np.isfinite(m):
        return ""
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Resumen por experimento base (media±std) agrupando seeds.")
    ap.add_argument("--input", default="experiments/summary_base.csv")
    ap.add_argument("--out_csv", default="experiments/summary_by_seed.csv")
    ap.add_argument("--out_md", default="experiments/summary_by_seed.md")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}")

    rows: List[Dict[str, str]] = []
    with in_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    # Agrupar SOLO _seedN
    groups: Dict[str, Dict[str, List[float]]] = {}
    seeds_seen: Dict[str, set] = {}

    for r in rows:
        exp_id = (r.get("exp_id") or "").strip()
        m = SEED_RE.match(exp_id)
        if not m:
            continue

        base = m.group("base")
        seed = int(m.group("seed"))

        groups.setdefault(base, {k: [] for k in METRICS})
        seeds_seen.setdefault(base, set()).add(seed)

        for k in METRICS:
            v = to_float(r.get(k, ""))
            if v is not None:
                groups[base][k].append(v)

    if not groups:
        raise RuntimeError("No he encontrado experimentos con sufijo _seedN en summary_base.csv")

    # Construye filas de salida
    out_rows: List[Dict[str, str]] = []
    for base in sorted(groups.keys()):
        n = len(seeds_seen.get(base, set()))
        out: Dict[str, str] = {"exp_base": base, "n_seeds": str(n)}

        for k in METRICS:
            m_, s_ = mean_std(groups[base][k])
            out[f"{k}_mean"] = "" if not np.isfinite(m_) else f"{m_:.6f}"
            out[f"{k}_std"] = "" if not np.isfinite(s_) else f"{s_:.6f}"
        out_rows.append(out)

    # CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["exp_base", "n_seeds"] + [f"{k}_{t}" for k in METRICS for t in ("mean", "std")]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # MD bonito
    out_md = Path(args.out_md)
    lines: List[str] = []
    lines.append("| Experimento base | n(seeds) | val_success | val_iou | val_angle | val_loss |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for r in out_rows:
        base = r["exp_base"]
        n = r["n_seeds"]

        def pm(key: str, nd: int) -> str:
            m = to_float(r.get(f"{key}_mean", ""))
            s = to_float(r.get(f"{key}_std", ""))
            if m is None or s is None:
                return ""
            return fmt_pm(m, s, nd=nd)

        lines.append(
            f"| {base} | {n} | "
            f"{pm('val_success', 4)} | {pm('val_iou', 4)} | {pm('val_angle', 2)} | {pm('val_loss', 4)} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] CSV: {out_csv}")
    print(f"[OK] MD : {out_md}")
    print("[INFO] Nota: este resumen usa SOLO carpetas con sufijo _seedN (versión PRO).")


if __name__ == "__main__":
    main()
