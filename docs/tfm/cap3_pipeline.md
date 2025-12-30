# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/docs/tfm/cap3_pipeline.md
# Nombre: cap3_pipeline.md
# Qué hace: Documenta el pipeline completo desde los datos de Cornell hasta la generación de métricas.

# Capítulo 3 — Pipeline de percepción

1. **Datos**: `data/cornell_raw/` contiene los pares RGB / depth y box annotations originales. `src/graspnet/datasets/cornell_dataset.py` expone el loader reusable para PyTorch.
2. **Configuraciones**: `config/*.yaml` parametriza conjuntos de entrenamientos y los nombres de experimento (por ejemplo `exp3_resnet18_rgbd.yaml`). Cada archivo define:
   - dataset (RGB/RGBD, padding, augmentaciones).
   - modelo (`model.name`) y tamaño.
   - entrenamiento (seed, epochs, lr_scheduler).
3. **Entrenamiento**: `scripts/run_one.sh` (nuevo) ejecuta `src/graspnet/train/train_cornell.py` para una configuración y semilla concretas. `scripts/run_seeds.sh` crea distintas versiones por semilla.
4. **Evaluación / métricas**:
   - `src/graspnet/metrics/select_best_epoch.py` (invoke via `scripts/analyze_experiments.py`) agrupa los `metrics.csv` en `experiments/summary_base.csv`.
   - `src/graspnet/metrics/summarize_by_seed.py` genera medias ± desviación estándar en `experiments/summary_by_seed.*`.
5. **Outputs reproducibles**:
   ```text
   experiments/
   ├── EXP3_RESNET18_RGBD_seed0/
   │   ├── checkpoints/
   │   └── metrics.csv
   ├── summary_base.csv
   ├── summary_base_pretty.md
   ├── summary_by_seed.csv
   ├── figures_memoria/
   └── plots/
   ```
6. **Evidencias**: `scripts/export_evidencias.py` sincroniza estos resúmenes con `reports/tablas_memoria/` (CSV + MD), copia `experiments/figures_memoria/` y genera plantillas de latencia y A/B.

Para rehacer el pipeline completo:

```bash
cd ~/TFM/agarre_inteligente
source .venv/bin/activate
./scripts/run_seeds.sh config/exp3_resnet18_rgbd.yaml 0 1 2
python scripts/analyze_experiments.py
python scripts/export_evidencias.py
```

El script `scripts/check_env.py` comprueba que todas las dependencias (torch, torchvision, pandas, cv2, matplotlib) estén accesibles antes de entrenar.
