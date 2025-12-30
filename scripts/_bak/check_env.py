# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/scripts/check_env.py
# Nombre: check_env.py
# Qué hace: Verifica que el entorno Python (.venv) esté listo y muestra versiones clave.

#!/usr/bin/env python3
"""Verifica imports críticos y muestra el estado del entorno.

Uso: python scripts/check_env.py
"""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


def try_import(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover
        return (False, str(exc))

    version = getattr(module, "__version__", "<desconocido>")
    return (True, version)


def pip_freeze() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Comprueba el entorno .venv y dependencias.")
    parser.add_argument(
        "--venv",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / ".venv"),
        help="Ruta al entorno virtual (.venv por defecto).",
    )
    args = parser.parse_args()

    venv_path = Path(args.venv)
    print(f"Python: {sys.executable} ({sys.version.splitlines()[0]})")
    print(f"Entorno virtual esperado: {venv_path}")
    print("PATH:", os.environ.get("PATH", ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

    packages = ["torch", "torchvision", "numpy", "pandas", "cv2", "matplotlib", "yaml", "tqdm"]
    print("\nDependencias clave:")
    for pkg in packages:
        ok, msg = try_import(pkg)
        status = "OK" if ok else "FALLO"
        print(f"  - {pkg:12} [{status}] -> {msg}")

    try:
        import torch  # noqa: F401

        cuda_avail = torch.cuda.is_available()
        print(f"\nCUDA disponible: {cuda_avail}")
        if cuda_avail:
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Device name : {torch.cuda.get_device_name(0)}")
            print(f"  CuDNN       : {torch.backends.cudnn.enabled}")
    except ImportError:
        print("\nTorch no disponible: saltando chequeos CUDA.")

    print("\nSalida de pip freeze (sólo primeras 10 líneas):")
    frozen = pip_freeze().splitlines()
    for line in frozen[:10]:
        print("  ", line)
    if len(frozen) > 10:
        print("  ...")

    print("\nChecklist:")
    print(" - Activa .venv con `source .venv/bin/activate` antes de entrenar.")
    print(" - Revisa `requirements.lock` y llama a `python scripts/check_env.py --venv .venv` si hay dudas.")


if __name__ == "__main__":
    main()
