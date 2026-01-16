"""
build_summary_base.py

Lee los metrics.csv de los experimentos base (EXP1-EXP4 por seed),
busca la época con mejor val_success (ignorando NaN)
y genera experiments/summary_base.csv listo para la memoria.
"""

import argparse
from pathlib import Path

import pandas as pd


def _resolve_exp_dir(base_name: str, seed: int) -> Path:
    exp_dir = Path("experiments") / f"{base_name}_seed{seed}"
    return exp_dir


def main():
    ap = argparse.ArgumentParser(description="Resumen base por seed (estructura EXP*_seedN).")
    ap.add_argument("--seed", type=int, default=0, help="Seed a resumir (default: 0)")
    ap.add_argument("--output", type=str, default="experiments/summary_base.csv", help="CSV de salida")
    args = ap.parse_args()

    experiments = [
        {
            "name": "EXP1_SIMPLE_RGB",
            "architecture": "SimpleGraspCNN",
            "augmentation": "No",
        },
        {
            "name": "EXP2_SIMPLE_RGBD",
            "architecture": "SimpleGraspCNN",
            "augmentation": "No",
        },
        {
            "name": "EXP3_RESNET18_RGB_AUGMENT",
            "architecture": "ResNet18Grasp",
            "augmentation": "Sí",
        },
        {
            "name": "EXP4_RESNET18_RGBD",
            "architecture": "ResNet18Grasp",
            "augmentation": "No",
        },
    ]

    rows = []

    for exp in experiments:
        exp_dir = _resolve_exp_dir(exp["name"], args.seed)
        metrics_path = exp_dir / "metrics.csv"

        if not metrics_path.exists():
            print(f"[ADVERTENCIA] No encuentro {metrics_path}, salto {exp['name']}")
            continue

        print(f"[INFO] Leyendo métricas de {metrics_path}")
        df = pd.read_csv(metrics_path)

        # Nos quedamos solo con las filas donde val_success NO es NaN
        df_valid = df[df["val_success"].notna()]

        if df_valid.empty:
            print(f"[ADVERTENCIA] Todas las val_success son NaN en {exp['name']}")
            continue

        # Índice de la fila con mayor val_success
        idx_best = df_valid["val_success"].idxmax()
        best_row = df_valid.loc[idx_best]

        # Extraemos valores numéricos
        epoch = int(best_row["epoch"])
        val_loss = float(best_row["val_loss"])
        val_iou = float(best_row["val_iou"])
        val_angle = float(best_row["val_angle"])
        val_success = float(best_row["val_success"])
        val_success_percent = val_success * 100.0

        print(
            f"[INFO] {exp['name']}: mejor época = {epoch}, "
            f"val_success = {val_success:.4f} ({val_success_percent:.2f}%)"
        )

        rows.append(
            {
                "experiment": exp["name"],
                "seed": args.seed,
                "architecture": exp["architecture"],
                "augmentation": exp["augmentation"],
                "best_epoch": epoch,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_angle_deg": val_angle,
                "val_success": val_success,
                "val_success_percent": val_success_percent,
            }
        )

    if not rows:
        print("[ERROR] No se ha podido generar ningún resumen (lista de filas vacía).")
        return

    df_out = pd.DataFrame(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(out_path, index=False)
    print(f"\n[OK] Resumen guardado en: {out_path}\n")

    # Lo mostramos por pantalla en formato legible
    print(
        df_out.to_string(
            index=False,
            float_format=lambda x: f"{x:0.4f}",
        )
    )


if __name__ == "__main__":
    main()

