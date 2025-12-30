# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/docs/tfm/evidencias_index.md
# Nombre: evidencias_index.md
# Qué hace: Índice de evidencias generadas para capítulo 4+ del TFM.

# Índice de evidencias

## Figuras ganadoras
- `experiments/figures_memoria/winner_loss.png`: curva de pérdida principal.
- `experiments/figures_memoria/winner_val_success.png`: comportamiento de éxito en validación.
- `reports/tablas_memoria/figures_memoria/`: copia básica de las figuras generadas.
- `docs/tfm/figuras_memoria/`: versión lista para la memoria (botón “Exportar gráficas” en el panel).

## Tablas clave
- `reports/tablas_memoria/summary_base.csv` / `summary_base_pretty.md`: métrica principal por experimento.
- `reports/tablas_memoria/summary_by_seed.csv` / `summary_by_seed.md`: medias ± desviación estándar agrupando seeds.
- `reports/tablas_memoria/tabla_ab.csv` / `tabla_ab.md`: comparaciones A/B disponibles en `experiments/ab_*.csv`.
- `reports/tablas_memoria/tabla_latency.csv` / `tabla_latency.md`: plantilla de latencias de inferencia (warmup=5, reps=20, batch=1).
- `docs/tfm/tablas_memoria/`: copia preparada para pegar en capítulos 4 y 5.

## Documentación
- `docs/tfm/cap1_contexto.md`
- `docs/tfm/cap3_pipeline.md`
- `docs/tfm/cap4_experimentos.md`

Recomiendo ejecutar `python scripts/export_evidencias.py --docs-root docs/tfm` o usar el botón correspondiente en el panel ROS2 después de cada ronda de entrenamiento para mantener estos artefactos actualizados y listos para el documento.
