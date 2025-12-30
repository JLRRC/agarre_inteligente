# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/scripts/export_evidencias.py
# Nombre: export_evidencias.py
# Qué hace: Sincroniza tablas/figuras de experimentos en `reports/tablas_memoria/` y opcionalmente en docs.

#!/usr/bin/env python3
"""Sincroniza evidencias (summary, A/B, latencias, figuras) para la memoria."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


ONLY_CHOICES = ("all", "tables", "figures")


def run_analyze(root: Path):
    script = root / "scripts" / "analyze_experiments.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{env.get('PYTHONPATH', '')}"
    subprocess.run([sys.executable, str(script)], check=True, cwd=root, env=env)


def copy_summary_files(root: Path, target: Path):
    paths = {
        "summary_base.csv": root / "experiments" / "summary_base.csv",
        "summary_base_pretty.md": root / "experiments" / "summary_base_pretty.md",
        "summary_by_seed.csv": root / "experiments" / "summary_by_seed.csv",
        "summary_by_seed.md": root / "experiments" / "summary_by_seed.md",
    }
    for name, src in paths.items():
        if not src.exists():
            print(f"[WARN] No existe {src}, omito copia.")
            continue
        dst = target / name
        dst.write_text(src.read_text(), encoding="utf-8")
        print(f"[COPY] {dst}")


def aggregate_ab_tables(root: Path, target: Path):
    files = sorted(root.glob("experiments/ab_*.csv"))
    if not files:
        print("[AB] No hay archivos ab_*.csv en experiments/. No genero tabla A/B.")
        return

    rows = []
    seen_columns = set()
    for path in files:
        with path.open() as handler:
            reader = csv.DictReader(handler)
            columns = reader.fieldnames or []
            seen_columns.update(columns)
            for line in reader:
                line = {k: v for k, v in line.items()}
                line["source"] = path.name
                rows.append(line)

    fieldnames = ["source"] + sorted(c for c in seen_columns if c not in {"source"})
    csv_path = target / "tabla_ab.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handler:
        writer = csv.DictWriter(handler, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = ["| Fuente | Métrica | Diff abs (p.p.) |", "|---|---|---:|"]
    for row in rows:
        diff = row.get("diff_abs_percent_points", "")
        metric = row.get("metric", "N/A")
        md_lines.append(f"| {row.get('source')} | {metric} | {diff} |")

    md_path = target / "tabla_ab.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[AB] Tabla A/B escrita en {csv_path} y {md_path}")


def measure_latency(dst: Path):
    warmup = 5
    reps = 20
    batch = 1
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None
    results = []

    def run_device(device: torch.device):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        model.eval()
        model.to(device)
        inp = torch.randn(batch, 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inp)
            durations = []
            for _ in range(reps):
                start = time.perf_counter()
                _ = model(inp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
        mean = statistics.mean(durations)
        std = statistics.pstdev(durations)
        results.append((device.type, warmup, reps, batch, mean, std, min(durations)))

    run_device(device_cpu)
    if device_gpu is not None:
        run_device(device_gpu)

    csv_path = dst / "tabla_latency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handler:
        fieldnames = ["device", "warmup", "reps", "batch", "mean_ms", "std_ms", "min_ms"]
        writer = csv.DictWriter(handler, fieldnames=fieldnames)
        writer.writeheader()
        for device, w, r, b, mean, std, min_ in results:
            writer.writerow(
                {
                    "device": device,
                    "warmup": w,
                    "reps": r,
                    "batch": b,
                    "mean_ms": f"{mean:.3f}",
                    "std_ms": f"{std:.3f}",
                    "min_ms": f"{min_:.3f}",
                }
            )

    md_lines = [
        "| Device | Warmup | Reps | Batch | Media (ms) | Std (ms) | Mín (ms) |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for device, w, r, b, mean, std, min_ in results:
        md_lines.append(f"| {device} | {w} | {r} | {b} | {mean:.3f} | {std:.3f} | {min_:.3f} |")

    md_path = dst / "tabla_latency.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[LAT] Latencias grabadas en {csv_path} y {md_path}")


def copy_figures(root: Path, target: Path):
    src_dir = root / "experiments" / "figures_memoria"
    dest_dir = target / "figures_memoria"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(src_dir.glob("*.png")):
        shutil.copy2(png, dest_dir / png.name)
        print(f"[FIG] Copiado {png.name}")


def _copy_files(src_dir: Path, dest_dir: Path, pattern: str):
    for src in sorted(src_dir.glob(pattern)):
        shutil.copy2(src, dest_dir / src.name)
        print(f"[DOCS] Copiado {src.name} -> {dest_dir}")


def mirror_to_docs(target: Path, docs_root: Path, mode: str):
    if not docs_root:
        return
    docs_root.mkdir(parents=True, exist_ok=True)
    if mode in ("all", "tables"):
        table_dest = docs_root / "tablas_memoria"
        table_dest.mkdir(parents=True, exist_ok=True)
        _copy_files(target, table_dest, "*.csv")
        _copy_files(target, table_dest, "*.md")
    if mode in ("all", "figures"):
        figs_src = target / "figures_memoria"
        if figs_src.exists():
            fig_dest = docs_root / "figuras_memoria"
            fig_dest.mkdir(parents=True, exist_ok=True)
            _copy_files(figs_src, fig_dest, "*.png")
        else:
            print(f"[DOCS] No hay figuras en {figs_src} para copiar.")


def main():
    parser = argparse.ArgumentParser(description="Exporta tablas y/o figuras para la memoria.")
    parser.add_argument("--root", default=".", help="Ruta al workspace agarre_inteligente.")
    parser.add_argument(
        "--only",
        choices=ONLY_CHOICES,
        default="all",
        help="Indica si se exportan tablas, figuras o ambos.",
    )
    parser.add_argument(
        "--docs-root",
        default="",
        help="Ruta alternativa (por ejemplo docs/tfm) donde replicar CSV/MD/figuras.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    target = root / "reports" / "tablas_memoria"
    target.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "tables"):
        run_analyze(root)
        copy_summary_files(root, target)
        aggregate_ab_tables(root, target)
        measure_latency(target)
    if args.only in ("all", "figures"):
        copy_figures(root, target)

    if args.docs_root:
        docs_path = Path(args.docs_root).resolve()
        mirror_to_docs(target, docs_path, args.only)

    print("[EXPORT] Evidencias sincronizadas en reports/tablas_memoria/ y experiments/figures_memoria/.")


if __name__ == "__main__":
    main()
