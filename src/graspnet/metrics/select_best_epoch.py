#!/usr/bin/env python
"""
select_best_epoch.py

Script para:
  - Leer los ficheros metrics.csv de cada experimento en experiments/
  - Seleccionar la mejor época según las métricas de validación
  - Generar un resumen en CSV con una fila por experimento

Uso típico:

  # Resumen de TODOS los experimentos en la carpeta experiments/
  python src/graspnet/metrics/select_best_epoch.py --root experiments

  # Resumen de un solo experimento
  python src/graspnet/metrics/select_best_epoch.py \
      --exp experiments/EXP3_RESNET18_RGBD_seed0

Requisitos:
  - Cada experimento tiene una carpeta tipo experiments/EXP... con:
      - metrics.csv  (con columnas epoch, val_loss, val_iou,
                      val_angle y, si existe, val_success)
      - config.yaml  (copiado por el script de entrenamiento)
"""

import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


def load_config(config_path: str) -> Dict:
    """
    Carga el YAML de configuración si existe.
    Devuelve {} si no se puede leer o no está PyYAML.
    """
    if not os.path.exists(config_path):
        return {}
    if yaml is None:
        # PyYAML no está instalado, devolvemos vacío
        return {}

    with open(config_path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception:
            cfg = {}
    return cfg or {}


def infer_seed_from_name(exp_name: str) -> Optional[int]:
    """
    Intenta extraer la seed del nombre del experimento.
    Ejemplos:
      EXP1_SIMPLE_RGB_seed0 -> 0
      EXP3_RESNET18_RGBD_seed42 -> 42
    """
    parts = exp_name.split("_")
    for p in parts:
        if p.startswith("seed"):
            try:
                return int(p.replace("seed", ""))
            except ValueError:
                pass
    return None


def is_finite(x: Optional[float]) -> bool:
    """
    True si x es un número real finito.
    """
    if x is None:
        return False
    return math.isfinite(x)


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    """
    Intenta leer un float de row[key]. Si falla, devuelve None.
    """
    if key not in row:
        return None
    try:
        return float(row[key])
    except (ValueError, TypeError):
        return None


def load_metrics_csv(metrics_path: str) -> List[Dict]:
    """
    Carga metrics.csv como una lista de dicts.
    Cada dict representa una fila (época).
    """
    rows: List[Dict] = []
    if not os.path.exists(metrics_path):
        return rows

    with open(metrics_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def select_best_row(
    rows: List[Dict]
) -> Optional[Dict]:
    """
    Selecciona la mejor época dada una lista de filas de metrics.csv.

    Criterio:
      1) Si existe columna val_success:
           - Maximizar val_success
           - Desempate: mayor val_iou, menor val_angle, menor val_loss
      2) Si NO existe val_success:
           - Maximizar val_iou
           - Desempate: menor val_angle, menor val_loss
    Solo se consideran filas con métricas finitas.
    """
    if not rows:
        return None

    has_val_success = "val_success" in rows[0]

    valid_rows: List[Tuple[float, float, float, float, Dict]] = []

    for row in rows:
        val_loss = parse_float(row, "val_loss")
        val_iou = parse_float(row, "val_iou")
        val_angle = parse_float(row, "val_angle")
        val_success = parse_float(row, "val_success") if has_val_success else None

        # Filtramos filas donde TODO esté roto
        if not (is_finite(val_iou) or is_finite(val_success)):
            continue

        if has_val_success and is_finite(val_success):
            # Usamos val_success como criterio principal
            score_main = val_success
        else:
            # Fallback: val_iou como criterio principal
            score_main = val_iou if is_finite(val_iou) else -1.0

        # Normalizamos valores None
        val_loss = val_loss if is_finite(val_loss) else float("inf")
        val_iou = val_iou if is_finite(val_iou) else -1.0
        val_angle = val_angle if is_finite(val_angle) else float("inf")

        valid_rows.append((score_main, val_iou, -val_angle, -val_loss, row))

    if not valid_rows:
        return None

    # Ordenamos por:
    #  - score_main (descendente)
    #  - val_iou (descendente)
    #  - -val_angle (descendente => ángulo pequeño)
    #  - -val_loss (descendente => loss pequeña)
    valid_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)

    # Devolvemos la mejor fila
    return valid_rows[0][4]


def summarize_experiment(exp_dir: str) -> Optional[Dict]:
    """
    Dado un directorio de experimento (con metrics.csv y config.yaml),
    devuelve un diccionario con el resumen de la mejor época.
    """
    exp_name = os.path.basename(exp_dir.rstrip("/"))
    metrics_path = os.path.join(exp_dir, "metrics.csv")
    config_path = os.path.join(exp_dir, "config.yaml")

    rows = load_metrics_csv(metrics_path)
    if not rows:
        print(f"[WARN] {exp_name}: no se ha podido leer metrics.csv")
        return None

    best_row = select_best_row(rows)
    if best_row is None:
        print(f"[WARN] {exp_name}: no hay filas válidas en metrics.csv")
        return None

    cfg = load_config(config_path)

    model_name = cfg.get("model", {}).get("name", "")
    modality = cfg.get("data", {}).get("modality", "")
    seed = infer_seed_from_name(exp_name)

    # Extraemos valores numéricos con cuidado
    def get_float(row: Dict[str, str], key: str) -> str:
        val = parse_float(row, key)
        if val is None or not math.isfinite(val):
            return ""
        return f"{val:.6f}"

    epoch_str = best_row.get("epoch", "")
    try:
        best_epoch = int(epoch_str)
    except (ValueError, TypeError):
        best_epoch = -1

    summary = {
        "exp_id": exp_name,
        "model": model_name,
        "modality": modality,
        "seed": seed if seed is not None else "",
        "best_epoch": best_epoch,
        "val_loss": get_float(best_row, "val_loss"),
        "val_iou": get_float(best_row, "val_iou"),
        "val_angle": get_float(best_row, "val_angle"),
        "val_success": get_float(best_row, "val_success")
        if "val_success" in best_row
        else "",
    }

    return summary


def find_experiments(root: str) -> List[str]:
    """
    Busca subcarpetas dentro de root que parezcan experimentos
    (contengan metrics.csv).
    """
    exp_dirs: List[str] = []
    if not os.path.exists(root):
        return exp_dirs

    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        metrics_path = os.path.join(full, "metrics.csv")
        if os.path.exists(metrics_path):
            exp_dirs.append(full)

    return exp_dirs


def write_summary_csv(summaries: List[Dict], out_path: str) -> None:
    """
    Escribe la lista de resúmenes en un CSV.
    """
    if not summaries:
        print("[WARN] No hay resúmenes para escribir en el CSV.")
        return

    fieldnames = [
        "exp_id",
        "model",
        "modality",
        "seed",
        "best_epoch",
        "val_loss",
        "val_iou",
        "val_angle",
        "val_success",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"[OK] Resumen escrito en: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Selecciona la mejor época de cada experimento y genera un resumen CSV."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="experiments",
        help="Carpeta raíz donde están las carpetas de experimentos.",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="",
        help="Ruta a un experimento concreto (opcional). Si se indica, solo resume ese.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/summary_base.csv",
        help="Ruta del CSV de salida cuando se procesan varios experimentos.",
    )

    args = parser.parse_args()

    if args.exp:
        # Solo un experimento
        exp_dir = args.exp
        if not os.path.isdir(exp_dir):
            print(f"[ERROR] {exp_dir} no es un directorio válido.")
            return
        summary = summarize_experiment(exp_dir)
        if summary is None:
            print("[ERROR] No se ha podido generar el resumen.")
            return

        print("\nResumen del experimento:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    # Varios experimentos en root
    exp_dirs = find_experiments(args.root)
    if not exp_dirs:
        print(f"[ERROR] No se han encontrado experimentos en {args.root}")
        return

    print(f"[INFO] Encontrados {len(exp_dirs)} experimentos en {args.root}")
    summaries: List[Dict] = []

    for exp_dir in exp_dirs:
        summary = summarize_experiment(exp_dir)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("[ERROR] No se ha podido generar ningún resumen.")
        return

    # Escribimos CSV resumen
    write_summary_csv(summaries, args.output)

    print("\nResumenes generados:")
    for s in summaries:
        print(
            f"  - {s['exp_id']}: epoch={s['best_epoch']}, "
            f"val_success={s['val_success']}, val_iou={s['val_iou']}, "
            f"val_angle={s['val_angle']}"
        )


if __name__ == "__main__":
    main()
