# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/scripts/run_one.sh
# Nombre: run_one.sh
# Qué hace: Ejecuta un único entrenamiento de `train_cornell.py` con config y semilla dadas.

#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-}"
SEED="${2:-0}"

function usage() {
  cat <<EOF
Uso: $0 <config_yaml> [seed]
Ejemplo: $0 config/exp1_simple_rgb.yaml 1
EOF
  exit 1
}

if [[ -z "$CFG" ]]; then
  usage
fi

if [[ ! -f "$REPO_ROOT/$CFG" ]]; then
  echo "[ERROR] No se encuentra el config: $REPO_ROOT/$CFG" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONPATH

echo "[RUN_ONE] Config: $CFG | Seed: $SEED"
.venv/bin/python "src/graspnet/train/train_cornell.py" --config "$CFG" --seed "$SEED"
