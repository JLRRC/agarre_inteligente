#!/usr/bin/env bash
set -euo pipefail

# === Ajustes mínimos ===
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-config/exp2_simple_rgbd.yaml}"     # usa tu mejor config por defecto
SPLIT="${2:-val}"                            # val recomendado
OUT_DIR="docs/tfm/figuras_memoria"
OUT_PNG="${OUT_DIR}/galeria_cualitativa_val_E1_E4.png"

mkdir -p "$OUT_DIR"

# Usamos el venv del repo si existe; si no, ejecuta con python del sistema
PY="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
fi

export ROOT CFG SPLIT OUT="$OUT_PNG"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] CFG=$CFG SPLIT=$SPLIT"
echo "[INFO] OUT=$OUT_PNG"

# Script inline: selecciona 3 aciertos y 3 fallos, dibuja overlays y compone una 2x3
$PY - <<'PY'
import os, math, random
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib (sin display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(os.environ.get("ROOT", ".")).resolve()
CFG = Path(os.environ.get("CFG", str(ROOT/"config/exp2_simple_rgbd.yaml")))
SPLIT = os.environ.get("SPLIT", "val")
OUT = Path(os.environ.get("OUT", str(ROOT/"docs/tfm/figuras_memoria/galeria_cualitativa_val_E1_E4.png")))

# --- Helpers geométricos ---
def rect_to_poly(cx, cy, w, h, theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    dx = w/2.0; dy = h/2.0
    pts = np.array([[-dx,-dy],[ dx,-dy],[ dx, dy],[-dx, dy]], dtype=np.float32)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    pts = pts @ R.T
    pts[:,0] += cx; pts[:,1] += cy
    return pts

def draw_rect(ax, poly, color, lw=2, alpha=0.9, label=None):
    ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=color, linewidth=lw, alpha=alpha))
    if label:
        ax.text(poly[0,0], poly[0,1], label, color=color, fontsize=9, weight="bold")

# --- Localización de artefactos reales ---
# Intentamos usar un CSV de predicciones si existiera; si no, marcamos no disponible con mensaje claro.
# Buscamos por patrones típicos en reports/ o experiments/.
cand_csv = list((ROOT/"reports").rglob("*qual*csv")) + list((ROOT/"experiments").rglob("*pred*csv"))
cand_csv = [p for p in cand_csv if p.is_file()]

if not cand_csv:
    # No hay artefacto de cualitativo/predicciones listo -> generamos una imagen "placeholder" honesta.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12,6))
    plt.axis("off")
    msg = (
        "GALERÍA CUALITATIVA NO DISPONIBLE (artefacto no encontrado)\n\n"
        "No se ha localizado en el repositorio un CSV/artefacto con predicciones por muestra\n"
        "para generar automáticamente overlays pred vs GT.\n\n"
        "Acción: ejecutar el script de generación cualitativa del pipeline y guardar el PNG en:\n"
        f"  docs/tfm/figuras_memoria/galeria_cualitativa_val_E1_E4.png\n"
    )
    plt.text(0.02, 0.95, msg, va="top", ha="left", fontsize=12)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"[WARN] No se encontraron artefactos cualitativos. Se creó placeholder: {OUT}")
    raise SystemExit(0)

# Si existe algún CSV, tomamos el primero (mejorable si luego quieres afinar)
pred_csv = cand_csv[0]
print(f"[INFO] Usando CSV: {pred_csv}")

df = pd.read_csv(pred_csv)
# Heurística: necesitamos columnas de imagen + pred + gt.
# Si el formato no coincide, generamos placeholder explicando qué faltó.
need_any = ["cx","cy","w","h","theta"]
if not any(col in df.columns for col in need_any):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12,6))
    plt.axis("off")
    plt.text(0.02, 0.95,
             "GALERÍA CUALITATIVA NO DISPONIBLE (formato inesperado)\n\n"
             f"CSV encontrado: {pred_csv}\n"
             f"Columnas: {list(df.columns)[:30]}...\n\n"
             "Acción: adaptar el generador para este formato o exportar pred/gt por muestra.\n",
             va="top", ha="left", fontsize=11)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"[WARN] CSV no tiene columnas esperadas. Placeholder: {OUT}")
    raise SystemExit(0)

# Si llegamos aquí, intentamos construir la galería con lo que haya (best-effort)
# Selección dummy: 3 filas “success==1” y 3 “success==0” si existe esa columna.
success_col = None
for c in ["success","val_success","is_success","ok"]:
    if c in df.columns:
        success_col = c
        break

if success_col:
    ok = df[df[success_col] == 1]
    bad = df[df[success_col] == 0]
else:
    ok = df.iloc[:0]
    bad = df

# Si no hay suficientes, hacemos muestreo simple
ok_rows = ok.sample(min(3, len(ok)), random_state=0) if len(ok) else df.sample(min(3, len(df)), random_state=0)
bad_rows = bad.sample(min(3, len(bad)), random_state=1) if len(bad) else df.sample(min(3, len(df)), random_state=1)

rows = list(ok_rows.to_dict("records")) + list(bad_rows.to_dict("records"))
titles = (["ACIERTO"]*len(ok_rows)) + (["FALLO"]*len(bad_rows))

# Composición 2x3
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, r, t in zip(axes.flat, rows, titles):
    ax.axis("off")
    ax.set_title(t, fontsize=11, weight="bold")
    # Intento cargar imagen si hay ruta
    img = None
    for key in ["image_path","img_path","path","file"]:
        if key in r:
            p = Path(str(r[key]))
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                import imageio.v2 as imageio
                img = imageio.imread(p)
            break
    if img is None:
        # Fondo vacío
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    ax.imshow(img)

    # Pred
    if all(k in r for k in ["pred_cx","pred_cy","pred_w","pred_h","pred_theta"]):
        pcx,pcy,pw,ph,pt = r["pred_cx"],r["pred_cy"],r["pred_w"],r["pred_h"],r["pred_theta"]
        poly = rect_to_poly(pcx,pcy,pw,ph,math.radians(pt) if abs(pt) > 2*math.pi else pt)
        draw_rect(ax, poly, "lime", label="PRED")
    # GT (si existe)
    if all(k in r for k in ["gt_cx","gt_cy","gt_w","gt_h","gt_theta"]):
        gcx,gcy,gw,gh,gt = r["gt_cx"],r["gt_cy"],r["gt_w"],r["gt_h"],r["gt_theta"]
        poly = rect_to_poly(gcx,gcy,gw,gh,math.radians(gt) if abs(gt) > 2*math.pi else gt)
        draw_rect(ax, poly, "deepskyblue", label="GT")

    # Tipología (best-effort)
    err = r.get("error_type", None) or r.get("E", None) or ""
    if err:
        ax.text(5, 20, str(err), color="yellow", fontsize=10, weight="bold")

# Si faltan paneles (por falta de filas), apagamos
for ax in axes.flat[len(rows):]:
    ax.axis("off")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"[OK] Galería guardada en: {OUT}")
PY
