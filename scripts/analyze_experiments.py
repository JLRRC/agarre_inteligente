# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/scripts/analyze_experiments.py
# Nombre: analyze_experiments.py
# Qué hace: Orquesta la generación de `summary_base` y `summary_by_seed` a partir de los experimentos.

#!/usr/bin/env python3
"""Instrumento PRO para guardar summaries de los experimentos."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_module(script: Path, args: list[str], env: dict[str, str]):
    print(f"[RUN] {script} {' '.join(args)}")
    subprocess.run([sys.executable, str(script), *args], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Genera summary_base + summary_by_seed (más reproducibilidad).")
    parser.add_argument("--root", type=str, default=".", help="Raíz del repo agarre_inteligente.")
    parser.add_argument("--output", type=str, default="experiments/summary_base.csv", help="CSV de salida para summary_base.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    experiments = root / "experiments"
    if not experiments.exists():
        raise FileNotFoundError(f"No existe el directorio experiments en {root}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{env.get('PYTHONPATH', '')}"

    select_script = root / "src" / "graspnet" / "metrics" / "select_best_epoch.py"
    summarize_script = root / "src" / "graspnet" / "metrics" / "summarize_by_seed.py"

    run_module(
        select_script,
        ["--root", str(experiments), "--output", str(root / args.output)],
        env,
    )
    run_module(
        summarize_script,
        [
            "--input",
            str(root / args.output),
            "--out_csv",
            str(experiments / "summary_by_seed.csv"),
            "--out_md",
            str(experiments / "summary_by_seed.md"),
        ],
        env,
    )

    print("[OK] Experimentos analizados y summary_by_seed regenerado.")


if __name__ == "__main__":
    main()
