#!/usr/bin/env python3
"""Genera tabla resumen A/B desde summary_base.csv o metrics reales."""
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_summary(summary_path: Path) -> List[Dict[str, str]]:
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def load_metrics(exp_dir: Path) -> Tuple[str, Dict[str, str]]:
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        return "", {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "", {}
    best = None
    best_val = None
    for row in rows:
        try:
            val = float(row.get("val_success", "nan"))
        except Exception:
            continue
        if best is None or val > best_val:
            best = row
            best_val = val
    return exp_dir.name, (best or rows[-1])


def extract_seed(exp_id: str) -> str:
    match = re.search(r"_seed(\d+)", exp_id)
    return match.group(1) if match else ""


def rows_to_markdown(rows: List[Dict[str, str]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join([str(row.get(col, "")) for col in header]) + " |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary_path = root / "experiments" / "summary_base.csv"
    rows: List[Dict[str, str]] = []

    if summary_path.exists():
        rows = load_summary(summary_path)
    else:
        for metrics_path in sorted((root / "experiments").glob("*/metrics.csv")):
            exp_id, best = load_metrics(metrics_path.parent)
            if not exp_id or not best:
                continue
            rows.append(
                {
                    "exp_id": exp_id,
                    "seed": extract_seed(exp_id),
                    "best_epoch": best.get("epoch", ""),
                    "val_success": best.get("val_success", ""),
                    "val_loss": best.get("val_loss", ""),
                }
            )

    if not rows:
        print("ERROR: no hay summary_base.csv ni metrics.csv reales", file=sys.stderr)
        return 2

    header = ["exp_id", "seed", "best_epoch", "val_success", "val_loss"]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(rows_to_markdown(rows, header) + "\n", encoding="utf-8")
    print(f"OK: tabla generada en {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
