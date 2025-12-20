#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Uso: $0 <config_base.yaml> <seed1> [seed2 ...]"
  echo "Ejemplo: $0 config/exp1_simple_rgb.yaml 0 1 2"
  exit 1
fi

CFG_BASE="$1"
shift
SEEDS=("$@")

if [[ ! -f "$CFG_BASE" ]]; then
  echo "[ERROR] No existe el config base: $CFG_BASE"
  exit 1
fi

# Directorio temporal para configs por seed
TMP_DIR="$(mktemp -d /tmp/seed_cfgs.XXXXXX)"
echo "[INFO] TMP_DIR=$TMP_DIR"

for s in "${SEEDS[@]}"; do
  OUT_CFG="$TMP_DIR/$(basename "${CFG_BASE%.yaml}")_seed${s}.yaml"

  python - "$CFG_BASE" "$OUT_CFG" "$s" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_base = Path(sys.argv[1])
out_cfg  = Path(sys.argv[2])
seed     = int(sys.argv[3])

cfg = yaml.safe_load(cfg_base.read_text())

# experiment_name -> añade sufijo _seedN
exp_name = cfg.get("experiment_name", "EXP")
cfg["experiment_name"] = f"{exp_name}_seed{seed}"

# train.seed
cfg.setdefault("train", {})
cfg["train"]["seed"] = seed

out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(out_cfg))
PY

  echo "[INFO] Entrenando seed=$s con $OUT_CFG"
  python src/graspnet/train/train_cornell.py --config "$OUT_CFG" --seed "$s"
done

echo "[OK] Listo. Configs generados en: $TMP_DIR"
echo "[INFO] Recomendado:"
echo "  python src/graspnet/metrics/select_best_epoch.py --root experiments"
echo "  python src/graspnet/metrics/summarize_by_seed.py --input experiments/summary_base.csv --out_csv experiments/summary_by_seed.csv --out_md experiments/summary_by_seed.md"
