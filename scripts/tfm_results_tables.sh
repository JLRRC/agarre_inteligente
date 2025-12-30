#!/usr/bin/env bash
# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/scripts/tfm_results_tables.sh
# Nombre: tfm_results_tables.sh
# Qué hace: genera tablas (por seed y agregadas media±desv por experimento) desde experiments/summary_base.csv en CSV y Markdown.

set -euo pipefail

AI_DIR="${AI_DIR:-$HOME/TFM/agarre_inteligente}"
SUMMARY="${SUMMARY:-$AI_DIR/experiments/summary_base.csv}"
OUT_ROOT="${OUT_ROOT:-$AI_DIR/reports/tfm_results_tables}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/$TS"
mkdir -p "$OUT_DIR"

if [[ ! -f "$SUMMARY" ]]; then
  echo "[ERROR] No existe: $SUMMARY"
  exit 1
fi

python3 - <<'PY' "$SUMMARY" "$OUT_DIR"
import csv, os, sys, math
from collections import defaultdict

summary_csv, out_dir = sys.argv[1], sys.argv[2]

def try_float(x):
    try:
        if x is None: return None
        x=str(x).strip()
        if x=="" or x.lower()=="nan": return None
        return float(x)
    except:
        return None

def mean(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def std_sample(xs):
    xs=[x for x in xs if x is not None]
    n=len(xs)
    if n<=1: return 0.0
    m=mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(n-1))

# Leer summary_base.csv
with open(summary_csv, newline="") as f:
    rows=list(csv.DictReader(f))

if not rows:
    raise SystemExit("[ERROR] summary_base.csv está vacío.")

cols=set().union(*[r.keys() for r in rows])

# Elegir métrica automáticamente (preferencias típicas)
prefs=["best_val_success","val_success","success","best_iou","val_iou","iou","best_f1","val_f1","f1","best_acc","val_acc","acc","val_loss","loss"]
metric=None
for p in prefs:
    if p in cols:
        metric=p
        break
if metric is None:
    # fallback: primera columna numérica que encuentre
    metric = "val_success" if "val_success" in cols else list(cols)[0]

# Tabla por seed
seed_cols=[c for c in ["experiment","seed",metric,"best_epoch","epoch","model","use_depth","augment","config"] if c in cols]
if "experiment" not in seed_cols: seed_cols.insert(0,"experiment")
if "seed" not in seed_cols and "seed" in cols: seed_cols.insert(1,"seed")
if metric not in seed_cols: seed_cols.append(metric)

by_seed=[]
for r in rows:
    rr={k:r.get(k,"") for k in seed_cols}
    v=try_float(r.get(metric))
    if v is not None:
        rr[metric]=f"{v:.6g}"
    by_seed.append(rr)

by_seed.sort(key=lambda r:(r.get("experiment",""), str(r.get("seed",""))))

# Agregada por experimento (media±desv muestral)
vals=defaultdict(list)
meta={}
for r in rows:
    exp=r.get("experiment") or r.get("name") or ""
    if not exp: 
        continue
    v=try_float(r.get(metric))
    if v is None:
        continue
    vals[exp].append(v)
    meta.setdefault(exp,r)

agg=[]
for exp, vlist in sorted(vals.items(), key=lambda kv: kv[0]):
    m=mean(vlist)
    s=std_sample(vlist)
    rr={
        "experiment": exp,
        "n_seeds": len(vlist),
        f"{metric}_mean": f"{m:.6g}" if m is not None else "",
        f"{metric}_std": f"{s:.6g}",
        f"{metric}_mean±std": f"{m:.6g} ± {s:.6g}" if m is not None else "",
    }
    for extra in ["model","use_depth","augment","config"]:
        if extra in cols:
            rr[extra]=meta[exp].get(extra,"")
    agg.append(rr)

agg_cols=["experiment","n_seeds",f"{metric}_mean±std",f"{metric}_mean",f"{metric}_std"]
for extra in ["model","use_depth","augment","config"]:
    if extra in cols:
        agg_cols.append(extra)

def write_csv(path, fieldnames, rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_md(path, fieldnames, rows, title):
    with open(path,"w") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"]*len(fieldnames)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k,"")) for k in fieldnames) + " |\n")

# Guardar
csv_seed=os.path.join(out_dir,"tabla_resultados_por_seed.csv")
md_seed=os.path.join(out_dir,"tabla_resultados_por_seed.md")
write_csv(csv_seed, seed_cols, by_seed)
write_md(md_seed, seed_cols, by_seed, f"Tabla resultados por seed (métrica: {metric})")

csv_agg=os.path.join(out_dir,"tabla_resultados_agregados_media_std.csv")
md_agg=os.path.join(out_dir,"tabla_resultados_agregados_media_std.md")
write_csv(csv_agg, agg_cols, agg)
write_md(md_agg, agg_cols, agg, f"Tabla agregada por experimento (media±desv) — métrica: {metric}")

idx=os.path.join(out_dir,"RESULTS_TABLES_INDEX.md")
with open(idx,"w") as f:
    f.write("# Índice — Tablas de resultados\n\n")
    f.write(f"- Fuente: `{os.path.basename(summary_csv)}`\n")
    f.write(f"- Métrica: **{metric}**\n\n")
    f.write("## Archivos\n")
    f.write("- tabla_resultados_por_seed.(csv/md)\n")
    f.write("- tabla_resultados_agregados_media_std.(csv/md)\n")

print("[OK] OUT_DIR:", out_dir)
PY

echo "[OK] Generado en: $OUT_DIR"
ls -ltr "$OUT_DIR"
