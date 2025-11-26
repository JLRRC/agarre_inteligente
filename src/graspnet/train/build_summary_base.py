"""
build_summary_base.py

Lee los metrics.csv de los experimentos base (EXP1-EXP3),
busca la época con mejor val_success (ignorando NaN)
y genera un fichero experiments/summary_base.csv con los resultados
listos para usar en la Tabla X del TFM.
"""

from pathlib import Path
import pandas as pd


def main():
    # Definimos aquí los experimentos que queremos resumir.
    # Ajusta rutas/nombres si cambian.
    experiments = [
        {
            "name": "EXP1",
            "dir": "experiments/exp1_simple_rgb",
            "architecture": "SimpleGraspCNN",
            "augmentation": "No",
        },
        {
            "name": "EXP2",
            "dir": "experiments/exp2_simple_rgb_augment",
            "architecture": "SimpleGraspCNN",
            "augmentation": "Sí",
        },
        {
            "name": "EXP3",
            "dir": "experiments/exp3_resnet18_rgb_augment",
            "architecture": "ResNet18Grasp",
            "augmentation": "Sí",
        },
    ]

    rows = []

    for exp in experiments:
        metrics_path = Path(exp["dir"]) / "metrics.csv"

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

    out_path = Path("experiments") / "summary_base.csv"
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


