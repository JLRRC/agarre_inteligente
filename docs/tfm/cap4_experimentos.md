# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/docs/tfm/cap4_experimentos.md
# Nombre: cap4_experimentos.md
# Qué hace: Describe los experimentos realizados, métricas y salidas listas para la memoria.

# Capítulo 4 — Experimentos y métricas

Los experimentos se organizan en `experiments/EXP*` y generan los siguientes artefactos reproducibles:

- `metrics.csv` (por época) dentro de cada experimento.
- `checkpoints/` con los pesos (`best.pth`/`last.pth`).
- `summary_base.csv` resume la mejor época de cada experimento gracias a `scripts/analyze_experiments.py`.
- `summary_by_seed.*` agrupa múltiples semillas y ofrece media ± desviación estándar para `val_success`, `val_iou`, `val_angle` y `val_loss`.
- `experiments/plots/` almacena curvas de pérdida y accuracy, mientras que `experiments/figures_memoria/` guarda las figuras ganadoras que luego se llevan a `reports/tablas_memoria/`.
- Los scripts `scripts/run_experiment_rgb.sh` y `scripts/run_experiment_rgbd.sh` permiten lanzar entrenamientos desde el workspace ROS al preparar `.venv` y ajustar `PYTHONPATH`.

El script `scripts/export_evidencias.py` crea:

1. `reports/tablas_memoria/summary_base_*.{csv,md}` y `summary_by_seed.{csv,md}` listos para pegar en la memoria.
2. `reports/tablas_memoria/tabla_ab.csv` que resume cada comparación A/B (lee `experiments/ab_*.csv`) y añade una tabla Markdown descriptiva.
3. `reports/tablas_memoria/tabla_latency.{csv,md}` siguiendo el protocolo warmup/reps/batch=1 sobre un modelo ligero y CPU/GPU (si hay).
4. Una copia sincronizada de `experiments/figures_memoria/`.

Para replicar el flujo de capítulo 4:

```bash
cd ~/TFM/agarre_inteligente
source .venv/bin/activate
./scripts/run_seeds.sh config/exp2_simple_rgbd.yaml 0 1 2
python scripts/analyze_experiments.py --root experiments --output experiments/summary_base.csv
python scripts/export_evidencias.py --force-latency
```

Todo lo generado está listo en `reports/tablas_memoria/` y en `experiments/figures_memoria/` para su descarga a `docs/tfm/evidencias_index.md`.
