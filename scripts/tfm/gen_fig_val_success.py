#!/usr/bin/env python3
"""Genera figura comparativa de val_success desde summary o metrics reales."""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_metrics(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for metrics_path in sorted((root / "experiments").glob("*/metrics.csv")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = list(csv.DictReader(handle))
        if not data:
            continue
        best = None
        best_val = None
        for row in data:
            try:
                val = float(row.get("val_success", "nan"))
            except Exception:
                continue
            if best is None or val > best_val:
                best = row
                best_val = val
        if best is None:
            continue
        rows.append({"exp_id": metrics_path.parent.name, "val_success": best.get("val_success", "")})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary_path = root / "experiments" / "summary_base.csv"
    rows: List[Dict[str, str]] = []

    if summary_path.exists():
        rows = load_summary(summary_path)
    else:
        rows = load_metrics(root)

    if not rows:
        print("ERROR: no hay datos reales para val_success", file=sys.stderr)
        return 2

    data: Dict[str, List[float]] = {}
    for row in rows:
        exp = row.get("exp_id", "")
        try:
            val = float(row.get("val_success", ""))
        except Exception:
            continue
        if exp:
            data.setdefault(exp, []).append(val)

    if not data:
        print("ERROR: datos vacios", file=sys.stderr)
        return 2

    labels = sorted(data.keys())
    values = [sum(data[k]) / len(data[k]) for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Comparativa val_success (media)")
    plt.ylabel("val_success")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"OK: figura generada en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
