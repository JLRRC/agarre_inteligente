"""
compare_ab.py

Compara dos experimentos (A y B) usando las últimas k épocas
con métricas válidas (sin NaN) y calcula la media de una métrica
(val_success por defecto), así como las diferencias absoluta y relativa.

Uso típico:
    python src/graspnet/train/compare_ab.py \
        --exp-a experiments/exp1_simple_rgb \
        --exp-b experiments/exp2_simple_rgb_augment \
        --metric val_success \
        --k 5 \
        --output experiments/ab_exp1_vs_exp2_val_success.csv
"""

from pathlib import Path
import argparse
import pandas as pd


def mean_last_k_valid(exp_dir: str, metric: str, k: int):
    """
    Lee metrics.csv de un experimento, se queda solo con las filas donde
    la métrica no es NaN, y calcula la media de las últimas k filas válidas.

    Devuelve (media, num_filas_usadas).
    """
    metrics_path = Path(exp_dir) / "metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"No se encuentra {metrics_path}")

    df = pd.read_csv(metrics_path)

    if metric not in df.columns:
        raise ValueError(f"La métrica '{metric}' no existe en {metrics_path}")

    # Filtramos filas con valor válido (no NaN)
    df_valid = df[df[metric].notna()]

    if df_valid.empty:
        raise ValueError(f"Todas las filas tienen NaN en '{metric}' para {exp_dir}")

    # Nos quedamos con las últimas k válidas (si hay menos de k, coge todas)
    df_last = df_valid.tail(k)
    k_used = len(df_last)

    mean_value = df_last[metric].mean()

    return float(mean_value), k_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-a", required=True, help="Ruta carpeta experimento A (contiene metrics.csv)")
    parser.add_argument("--exp-b", required=True, help="Ruta carpeta experimento B (contiene metrics.csv)")
    parser.add_argument("--metric", default="val_success", help="Nombre de la métrica a comparar")
    parser.add_argument("--k", type=int, default=5, help="Número de últimas épocas válidas a promediar")
    parser.add_argument("--output", required=True, help="Ruta del CSV de salida con el resumen A/B")

    args = parser.parse_args()

    print(f"[INFO] Comparando '{args.metric}' entre:")
    print(f"       A = {args.exp_a}")
    print(f"       B = {args.exp_b}")
    print(f"       usando las últimas {args.k} épocas válidas.\n")

    mean_a, k_a = mean_last_k_valid(args.exp_a, args.metric, args.k)
    mean_b, k_b = mean_last_k_valid(args.exp_b, args.metric, args.k)

    print(f"[INFO] A: media = {mean_a:.4f} (a partir de {k_a} épocas válidas)")
    print(f"[INFO] B: media = {mean_b:.4f} (a partir de {k_b} épocas válidas)")

    # Diferencias en valor "crudo" (por ejemplo, 0.05 vs 0.08)
    diff_abs = mean_b - mean_a

    # Si la métrica es un ratio tipo val_success (0.x), lo pasamos a porcentaje
    mean_a_percent = mean_a * 100.0
    mean_b_percent = mean_b * 100.0
    diff_abs_percent_points = diff_abs * 100.0  # puntos porcentuales

    if mean_a != 0.0:
        diff_rel_percent = (diff_abs / mean_a) * 100.0
    else:
        diff_rel_percent = float("nan")

    print("\n[RESULTADO]")
    print(f"  media A       = {mean_a:.4f} ({mean_a_percent:.2f}%)")
    print(f"  media B       = {mean_b:.4f} ({mean_b_percent:.2f}%)")
    print(f"  diff abs (pp) = {diff_abs_percent_points:.2f} puntos porcentuales")
    print(f"  diff rel (%)  = {diff_rel_percent:.2f}%\n")

    # Guardamos todo en un CSV pequeño
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(
        [
            {
                "metric": args.metric,
                "k_requested": args.k,
                "k_used_A": k_a,
                "k_used_B": k_b,
                "mean_A": mean_a,
                "mean_B": mean_b,
                "mean_A_percent": mean_a_percent,
                "mean_B_percent": mean_b_percent,
                "diff_abs_percent_points": diff_abs_percent_points,
                "diff_rel_percent": diff_rel_percent,
            }
        ]
    )

    df_out.to_csv(out_path, index=False)
    print(f"[OK] Resumen A/B guardado en: {out_path}")


if __name__ == "__main__":
    main()


