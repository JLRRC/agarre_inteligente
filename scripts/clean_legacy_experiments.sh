#!/usr/bin/env bash
set -euo pipefail

ROOT="experiments"
LEGACY_DIR="${ROOT}/_legacy"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

ts="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LEGACY_DIR"

echo "[INFO] Root: $ROOT"
echo "[INFO] Legacy: $LEGACY_DIR"
echo "[INFO] Dry-run: $DRY_RUN"
echo

shopt -s nullglob

moved=0
kept=0
skipped=0

for d in "$ROOT"/*; do
  [[ -d "$d" ]] || continue
  name="$(basename "$d")"

  # Ignora carpetas auxiliares
  if [[ "$name" == "_legacy" || "$name" == "plots" ]]; then
    continue
  fi

  # Ignora ficheros sueltos tipo summary_*.csv/md
  # (esto ya lo evita el [[ -d ]], pero lo dejo claro)

  # Si ya es seed => no se toca
  if [[ "$name" =~ _seed[0-9]+$ ]]; then
    continue
  fi

  # Solo consideramos "experimentos" si tienen metrics.csv
  if [[ ! -f "$d/metrics.csv" ]]; then
    skipped=$((skipped+1))
    continue
  fi

  # ¿Hay al menos un experimento con seeds para este base?
  if compgen -G "$ROOT/${name}_seed*" > /dev/null; then
    target="${LEGACY_DIR}/${name}__legacy__${ts}"
    echo "[MOVE] $d  ->  $target"
    if [[ $DRY_RUN -eq 0 ]]; then
      mv "$d" "$target"
    fi
    moved=$((moved+1))
  else
    echo "[KEEP] $d (no tiene seeds)"
    kept=$((kept+1))
  fi
done

echo
echo "[OK] moved=$moved kept=$kept skipped=$skipped"
echo "[INFO] Puedes regenerar summary_base.csv después (recomendado)."
