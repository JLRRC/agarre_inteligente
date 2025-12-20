#!/usr/bin/env python3
"""
select_best_epoch.py (PRO)

- Lee metrics.csv de cada experimento dentro de experiments/
- Selecciona la mejor época:
    * si hay val_success -> maximiza val_success
      desempate: mayor val_iou, menor val_angle, menor val_loss
    * si no hay val_success -> maximiza val_iou
      desempate: menor val_angle, menor val_loss
- Genera experiments/summary_base.csv (1 fila por experimento)

Uso:
  python src/graspnet/metrics/select_best_epoch.py --root experiments
  python src/graspnet/metrics/select_best_epoch.py --exp experiments/EXP1_SIMPLE_RGB_seed0
  python src/graspnet/metrics/select_best_epoch.py --root experiments --output experiments/summary_base.csv
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def is_finite(x: Optional[float]) -> bool:
    return (x is not None) and math.isfinite(x)


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_metrics_csv(metrics_path: Path) -> List[Dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_config_from_exp(exp_dir: Path) -> Dict:
    """
    Entrenamiento copia a config_used.yaml (PRO). Mantenemos fallback.
    """
    if yaml is None:
        return {}

    candidates = [
        exp_dir / "config_used.yaml",
        exp_dir / "config_used.yml",
        exp_dir / "config.yaml",
        exp_dir / "config.yml",
    ]
    for p in candidates:
        if p.exists():
            try:
                return yaml.safe_load(p.read_text()) or {}
            except Exception:
                return {}
    return {}


def infer_seed(exp_name: str, cfg: Dict) -> str:
    # 1) por sufijo en nombre: *_seedN
    parts = exp_name.split("_")
    for p in parts:
        if p.startswith("seed"):
            s = p.replace("seed", "").strip()
            if s.isdigit():
                return s

    # 2) por train.seed en YAML
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    seed = train_cfg.get("seed", "")
    return str(seed) if seed != "" else ""


def infer_modality(cfg: Dict) -> str:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    use_depth = bool(data_cfg.get("use_depth", False))
    return "RGBD" if use_depth else "RGB"


def infer_augment(cfg: Dict) -> str:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    aug = data_cfg.get("augmentation", {}) if isinstance(data_cfg, dict) else {}
    if not isinstance(aug, dict):
        return ""
    g = bool(aug.get("geometric", False))
    p = bool(aug.get("photometric", False))
    if g or p:
        return f"geo={int(g)}|photo={int(p)}"
    return ""


def select_best_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not rows:
        return None

    has_val_success = "val_success" in rows[0].keys()
    ranked: List[Tuple[float, float, float, float, Dict[str, str]]] = []

    for row in rows:
        val_success = parse_float(row, "val_success") if has_val_success else None
        val_iou = parse_float(row, "val_iou")
        val_angle = parse_float(row, "val_angle")
        val_loss = parse_float(row, "val_loss")

        if (not is_finite(val_success)) and (not is_finite(val_iou)):
            continue

        if has_val_success and is_finite(val_success):
            score_main = float(val_success)
        else:
            score_main = float(val_iou) if is_finite(val_iou) else -1.0

        iou = float(val_iou) if is_finite(val_iou) else -1.0
        ang = float(val_angle) if is_finite(val_angle) else float("inf")
        loss = float(val_loss) if is_finite(val_loss) else float("inf")

        # Orden: score_main DESC, iou DESC, ang ASC, loss ASC
        ranked.append((score_main, iou, -ang, -loss, row))

    if not ranked:
        return None

    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    return ranked[0][4]


def summarize_experiment(exp_dir: Path) -> Optional[Dict[str, str]]:
    exp_name = exp_dir.name
    metrics_path = exp_dir / "metrics.csv"

    rows = load_metrics_csv(metrics_path)
    if not rows:
        print(f"[WARN] {exp_name}: no se pudo leer metrics.csv")
        return None

    best = select_best_row(rows)
    if best is None:
        print(f"[WARN] {exp_name}: no hay filas válidas en metrics.csv")
        return None

    cfg = load_config_from_exp(exp_dir) if yaml is not None else {}

    model_name = ""
    if isinstance(cfg, dict):
        model_name = str(cfg.get("model", {}).get("name", ""))

    modality = infer_modality(cfg) if cfg else ""
    augment = infer_augment(cfg) if cfg else ""
    seed = infer_seed(exp_name, cfg) if cfg else ""

    def fmt(key: str, nd: int) -> str:
        v = parse_float(best, key)
        if v is None or not math.isfinite(v):
            return ""
        return f"{v:.{nd}f}"

    try:
        best_epoch = int(best.get("epoch", ""))
    except Exception:
        best_epoch = -1

    # ckpt paths (respetando tu estructura)
    ckpt_best = exp_dir / "checkpoints" / "best.pth"
    ckpt_last = exp_dir / "checkpoints" / "last.pth"

    return {
        "exp_id": exp_name,
        "model": model_name,
        "modality": modality,
        "augment": augment,
        "seed": seed,
        "best_epoch": str(best_epoch),
        "val_success": fmt("val_success", 6),
        "val_iou": fmt("val_iou", 6),
        "val_angle": fmt("val_angle", 2),
        "val_loss": fmt("val_loss", 6),
        "ckpt_best": str(ckpt_best) if ckpt_best.exists() else "",
        "ckpt_last": str(ckpt_last) if ckpt_last.exists() else "",
    }


def find_experiments(root: Path) -> List[Path]:
    if not root.exists():
        return []
    exp_dirs: List[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "metrics.csv").exists():
            exp_dirs.append(d)
    return exp_dirs


def write_summary_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        print("[WARN] No hay resúmenes para escribir.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "exp_id",
        "model",
        "modality",
        "augment",
        "seed",
        "best_epoch",
        "val_success",
        "val_iou",
        "val_angle",
        "val_loss",
        "ckpt_best",
        "ckpt_last",
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Resumen escrito en: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="experiments")
    ap.add_argument("--exp", type=str, default="")
    ap.add_argument("--output", type=str, default="experiments/summary_base.csv")
    args = ap.parse_args()

    if args.exp:
        exp_dir = Path(args.exp)
        if not exp_dir.is_dir():
            print(f"[ERROR] {exp_dir} no es un directorio válido.")
            return
        s = summarize_experiment(exp_dir)
        if not s:
            print("[ERROR] No se pudo resumir el experimento.")
            return
        print("\nResumen del experimento:")
        for k, v in s.items():
            print(f"  {k}: {v}")
        return

    root = Path(args.root)
    exps = find_experiments(root)
    if not exps:
        print(f"[ERROR] No se encontraron experimentos en {root}")
        return

    print(f"[INFO] Encontrados {len(exps)} experimentos en {root}")
    summaries: List[Dict[str, str]] = []
    for e in exps:
        s = summarize_experiment(e)
        if s:
            summaries.append(s)

    if not summaries:
        print("[ERROR] No se pudo generar ningún resumen.")
        return

    out_path = Path(args.output)
    write_summary_csv(summaries, out_path)

    print("\nResúmenes:")
    for s in summaries:
        print(
            f"  - {s['exp_id']}: best_epoch={s['best_epoch']}, "
            f"val_success={s['val_success']}, val_iou={s['val_iou']}, val_angle={s['val_angle']}"
        )


if __name__ == "__main__":
    main()
