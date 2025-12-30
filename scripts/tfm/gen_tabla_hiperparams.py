#!/usr/bin/env python3
"""Genera tabla de hiperparametros desde config/*.yaml reales."""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except Exception:
    yaml = None


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

    if yaml is None:
        print("ERROR: PyYAML no disponible", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    cfg_dir = root / "config"
    rows: List[Dict[str, str]] = []

    if cfg_dir.exists():
        for cfg in sorted(cfg_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
            model_cfg = data.get("model", {}) if isinstance(data, dict) else {}
            train_cfg = data.get("train", {}) if isinstance(data, dict) else {}
            data_cfg = data.get("data", {}) if isinstance(data, dict) else {}
            rows.append(
                {
                    "exp": cfg.stem,
                    "model": str(model_cfg.get("name", model_cfg.get("type", ""))),
                    "lr": str(train_cfg.get("lr", train_cfg.get("learning_rate", ""))),
                    "batch_size": str(train_cfg.get("batch_size", "")),
                    "epochs": str(train_cfg.get("num_epochs", train_cfg.get("epochs", ""))),
                    "use_depth": str(bool(data_cfg.get("use_depth", False))),
                }
            )

    if not rows:
        print("ERROR: no hay configs YAML reales", file=sys.stderr)
        return 2

    header = ["exp", "model", "lr", "batch_size", "epochs", "use_depth"]
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
