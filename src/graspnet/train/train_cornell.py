import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Any, Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import yaml

from graspnet.datasets.cornell_dataset import CornellGraspDataset
from graspnet.models import build_model
from graspnet.utils.metrics import (
    angle_diff_deg,
    grasp_iou,
)

# =============================================================================
# Args / Config
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento en Cornell Grasping Dataset (RGB / RGB-D)"
    )
    parser.add_argument("--config", type=str, required=True, help="Ruta al YAML de config")
    parser.add_argument("--seed", type=int, default=0, help="Semilla aleatoria")
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de config: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"El YAML no parece un dict válido: {cfg_path}")
    return cfg


# =============================================================================
# Reproducibilidad (con manejo CuBLAS)
# =============================================================================


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        # Menos estricto (más rápido)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def ensure_cublas_determinism_or_disable(cfg: dict, device: torch.device) -> None:
    """
    Si determinismo está activado y usamos CUDA, CuBLAS necesita CUBLAS_WORKSPACE_CONFIG.
    - Si no está, NO crasheamos: avisamos y desactivamos determinismo.
    - Si quieres forzar error en vez de auto-desactivar, pon:
        train:
          deterministic_strict: true
    """
    train_cfg = cfg.get("train", {})
    deterministic = bool(train_cfg.get("deterministic", True))
    strict = bool(train_cfg.get("deterministic_strict", False))

    if device.type != "cuda":
        return
    if not deterministic:
        return

    # PyTorch exige esto para algunas ops (CuBLAS >= 10.2)
    env = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "").strip()
    if env in (":4096:8", ":16:8"):
        return

    msg = (
        "[WARN] Determinismo activado pero falta CUBLAS_WORKSPACE_CONFIG.\n"
        "       Exporta antes de ejecutar:\n"
        "         export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
        "       (o :16:8)\n"
    )

    if strict:
        raise RuntimeError(msg + "       deterministic_strict=true => abortando.")
    else:
        print(msg + "       Continuo DESACTIVANDO determinismo para evitar crash.\n")
        # Desactivamos determinismo para que no explote en backward
        seed_everything(int(train_cfg.get("seed_effective", 0)), deterministic=False)


# =============================================================================
# Helpers
# =============================================================================


def _finite_rows(t: torch.Tensor) -> torch.Tensor:
    b = t.size(0)
    return torch.isfinite(t.view(b, -1)).all(dim=1)


class _WithIndex(Dataset):
    """Wrap dataset to add _idx for traceability in DataLoader batches."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        sample = self.base[i]
        if not isinstance(sample, dict):
            return sample
        if isinstance(self.base, Subset):
            base_idx = int(self.base.indices[i])
        else:
            base_idx = int(i)
        out = dict(sample)
        out["_idx"] = base_idx
        return out


def _load_indices_strict(path: str) -> List[int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"index_file no existe: {p}")
    idx: List[int] = []
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                idx.append(int(line))
    if len(idx) == 0:
        raise ValueError(f"index_file vacío: {p}")
    return idx


def _extract_model_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        out = outputs
    elif isinstance(outputs, dict):
        for k in ("pred", "out", "outputs", "y", "logits"):
            if k in outputs:
                out = outputs[k]
                break
        else:
            raise ValueError(f"Salida dict sin clave reconocida: {list(outputs.keys())}")
        if not isinstance(out, torch.Tensor):
            raise ValueError("La clave encontrada en la salida dict no contiene un Tensor.")
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        out = outputs[0]
        if not isinstance(out, torch.Tensor):
            raise ValueError("La salida del modelo (tuple/list) no contiene Tensor en la posición 0.")
    else:
        raise ValueError(f"Tipo de salida del modelo no soportado: {type(outputs)}")

    if out.ndim != 2 or out.size(-1) != 5:
        raise ValueError(f"Salida del modelo con shape inesperada: {tuple(out.shape)} (esperado [B,5])")
    return out


def _get_batch_tensors(
    batch: Any,
    device: torch.device,
    use_depth: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if isinstance(batch, dict):
        if "rgb" not in batch or "grasp" not in batch:
            raise KeyError(f"Batch dict sin 'rgb'/'grasp'. Claves: {list(batch.keys())}")
        rgb = batch["rgb"]
        grasp = batch["grasp"]
        depth = batch.get("depth", None)
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            rgb, grasp = batch
            depth = None
        elif len(batch) == 3:
            rgb, depth, grasp = batch
        else:
            raise ValueError(f"Batch tuple/list con longitud no soportada: {len(batch)}")
    else:
        raise TypeError(f"Tipo de batch no soportado: {type(batch)}")

    nb = torch.cuda.is_available()
    rgb = rgb.to(device, non_blocking=nb)
    grasp = grasp.to(device, non_blocking=nb)

    if use_depth:
        if depth is None:
            raise KeyError("use_depth=True pero el batch no trae 'depth'. Revisa dataset/config.")
        depth = depth.to(device, non_blocking=nb)
    else:
        depth = None

    return rgb, depth, grasp


def _get_batch_indices(batch: Any) -> Optional[torch.Tensor]:
    if isinstance(batch, dict) and "_idx" in batch:
        idxs = batch["_idx"]
        if torch.is_tensor(idxs):
            return idxs
    return None


def _sanitize_params_np(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32, copy=True)
    p[2] = max(float(abs(p[2])), 1e-6)  # w
    p[3] = max(float(abs(p[3])), 1e-6)  # h
    return p


def _unwrap_subset(ds):
    """Devuelve (base_ds, subset_flag)."""
    if isinstance(ds, _WithIndex):
        ds = ds.base
    if isinstance(ds, Subset):
        base = ds.dataset
        if isinstance(base, _WithIndex):
            base = base.base
        return base, True
    return ds, False


def _dataset_debug_info(name: str, ds, cfg_root: str):
    base, is_subset = _unwrap_subset(ds)
    print(f"[INFO] Cornell root_dir (cfg) = {cfg_root}")
    print(f"[INFO] Dataset {name}: type={type(ds).__name__}, len={len(ds)}, subset={is_subset}")
    # Intentamos sacar atributos típicos del dataset
    root_dir = getattr(base, "root_dir", None)
    split = getattr(base, "split", None)
    use_depth = getattr(base, "use_depth", None)
    img_size = getattr(base, "img_size", None)
    print(
        f"[INFO] Dataset {name}: base_type={type(base).__name__}, "
        f"root_dir={root_dir}, split={split}, use_depth={use_depth}, img_size={img_size}"
    )


def _append_bad_indices(path: Optional[Path], split: str, batch_idx: int, reason: str, idxs: Optional[torch.Tensor]):
    if path is None or idxs is None:
        return
    idx_list = [int(x) for x in idxs.detach().cpu().tolist()]
    if not idx_list:
        return
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["split", "batch", "reason", "idx"])
        for idx in idx_list:
            writer.writerow([split, batch_idx, reason, idx])


# =============================================================================
# DataLoaders (Opción B PRO)
# =============================================================================


def make_dataloaders(cfg: dict, use_depth: bool):
    if "data" not in cfg:
        raise KeyError("Falta la sección 'data' en el YAML.")
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})

    root_dir = str(data_cfg["root_dir"]).strip()
    img_size = int(data_cfg.get("img_size", 224))
    val_split = float(data_cfg.get("val_split", 0.2))

    # batch_size/num_workers: primero train:, fallback a data:
    batch_size = int(train_cfg.get("batch_size", data_cfg.get("batch_size", 16)))
    num_workers = int(train_cfg.get("num_workers", data_cfg.get("num_workers", 4)))

    # Comprobación de ruta (si no existe, avisamos fuerte)
    if not Path(root_dir).exists():
        print(f"[WARN] root_dir del YAML NO existe: {root_dir}")
        # No abortamos aquí porque tu CornellGraspDataset puede redirigir internamente,
        # pero lo dejamos MUY visible.
        # Si quieres abortar: pon data.require_root_exists: true
        if bool(data_cfg.get("require_root_exists", False)):
            raise FileNotFoundError(f"root_dir no existe y require_root_exists=true: {root_dir}")

    aug_cfg = data_cfg.get("augmentation", {})
    train_aug = {
        "geometric": bool(aug_cfg.get("geometric", False)),
        "photometric": bool(aug_cfg.get("photometric", False)),
    }

    # 1) Datasets base
    train_dataset = CornellGraspDataset(
        root_dir=root_dir,
        split="train",
        val_split=val_split,
        img_size=img_size,
        use_depth=use_depth,
        include_depth=use_depth,
        augmentation=train_aug,
    )
    val_dataset = CornellGraspDataset(
        root_dir=root_dir,
        split="val",
        val_split=val_split,
        img_size=img_size,
        use_depth=use_depth,
        include_depth=use_depth,
        random_grasp=False,
        augmentation=None,
    )

    # 2) Subset PRO por índices limpios (si está configurado)
    index_cfg = data_cfg.get("index_files", None)
    if isinstance(index_cfg, dict):
        train_index_file = str(index_cfg.get("train", "")).strip()
        val_index_file = str(index_cfg.get("val", "")).strip()

        if train_index_file or val_index_file:
            print(f"[INFO] index_files.train = {train_index_file}")
            print(f"[INFO] index_files.val   = {val_index_file}")

        if train_index_file:
            idx_train = _load_indices_strict(train_index_file)
            old = len(train_dataset)
            train_dataset = Subset(train_dataset, idx_train)
            print(f"[INFO] Subset TRAIN por índices limpios: {old} -> {len(train_dataset)} (n_idx={len(idx_train)})")

        if val_index_file:
            idx_val = _load_indices_strict(val_index_file)
            old = len(val_dataset)
            val_dataset = Subset(val_dataset, idx_val)
            print(f"[INFO] Subset VAL por índices limpios: {old} -> {len(val_dataset)} (n_idx={len(idx_val)})")

    # 3) DataLoaders
    train_dataset = _WithIndex(train_dataset)
    val_dataset = _WithIndex(val_dataset)

    pin_memory = bool(data_cfg.get("pin_memory", False)) and torch.cuda.is_available()
    persistent_workers = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Debug: confirmar dataset real
    _dataset_debug_info("TRAIN", train_dataset, cfg_root=root_dir)
    _dataset_debug_info("VAL", val_dataset, cfg_root=root_dir)

    return train_loader, val_loader


# =============================================================================
# Modelo + salidas
# =============================================================================


def make_model(cfg: dict, device: torch.device, in_channels: int) -> nn.Module:
    if "model" not in cfg:
        raise KeyError("Falta la sección 'model' en el YAML.")
    model_cfg = cfg["model"]
    model_name = str(model_cfg["name"]).strip()
    pretrained = bool(model_cfg.get("pretrained", False))
    data_cfg = cfg.get("data", {})
    img_size = int(data_cfg.get("img_size", 224))

    cfg_in = model_cfg.get("in_channels", None)
    if cfg_in is not None and int(cfg_in) != int(in_channels):
        print(f"[WARN] model.in_channels={cfg_in} pero data.use_depth implica in_channels={in_channels}. Usaré {in_channels}.")

    model = build_model(
        model_name,
        in_channels=in_channels,
        pretrained=pretrained,
        img_size=img_size,
    )
    model.to(device)
    return model


def ensure_dirs(cfg: dict):
    if "logging" not in cfg:
        raise KeyError("Falta la sección 'logging' en el YAML.")
    if "experiment_name" not in cfg:
        raise KeyError("Falta 'experiment_name' en el YAML.")

    log_cfg = cfg["logging"]
    output_root = Path(log_cfg.get("output_dir", log_cfg.get("base_dir", "experiments")))
    exp_name = cfg["experiment_name"]

    ckpt_dirname = str(log_cfg.get("ckpt_dirname", "checkpoints"))
    metrics_filename = str(log_cfg.get("metrics_filename", "metrics.csv"))

    base_dir = output_root / exp_name
    ckpt_dir = base_dir / ckpt_dirname
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = base_dir / metrics_filename
    return base_dir, ckpt_dir, metrics_path


# =============================================================================
# Logging
# =============================================================================


def append_metrics_row(metrics_path: Path, metrics_dict: dict):
    file_exists = metrics_path.exists()
    fieldnames = ["epoch", "train_loss", "val_loss", "val_iou", "val_angle", "val_success"]

    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)


def save_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics_dict: dict,
    is_best: bool,
):
    epoch = int(metrics_dict["epoch"])
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics_dict,
    }

    torch.save(state, ckpt_dir / "last.pth")
    if is_best:
        torch.save(state, ckpt_dir / "best.pth")


# =============================================================================
# Train / Validate
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_depth: bool,
    bad_indices_path: Optional[Path] = None,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        rgb, depth, grasp = _get_batch_tensors(batch, device=device, use_depth=use_depth)
        idxs = _get_batch_indices(batch)
        x = torch.cat([rgb, depth], dim=1) if use_depth else rgb

        # Red de seguridad: filtra por muestra si algo no-finito se cuela
        finite = _finite_rows(x) & torch.isfinite(grasp).all(dim=1)
        if not finite.all():
            bad = (~finite).sum().item()
            total = grasp.size(0)
            print(f"[WARN] NaN/Inf en TRAIN (x/grasp): {bad}/{total} (batch {batch_idx}). Se filtran.")
            _append_bad_indices(bad_indices_path, "train", batch_idx, "nonfinite_input", idxs[~finite] if idxs is not None else None)
            x = x[finite]
            grasp = grasp[finite]
            if idxs is not None:
                idxs = idxs[finite]
            if grasp.size(0) == 0:
                continue

        optimizer.zero_grad(set_to_none=True)

        outputs_raw = model(x)
        outputs = _extract_model_output(outputs_raw)

        finite_out = torch.isfinite(outputs).all(dim=1)
        if not finite_out.all():
            bad = (~finite_out).sum().item()
            total = outputs.size(0)
            print(f"[WARN] NaN/Inf en outputs TRAIN: {bad}/{total} (batch {batch_idx}). Se filtran.")
            _append_bad_indices(bad_indices_path, "train", batch_idx, "nonfinite_output", idxs[~finite_out] if idxs is not None else None)
            outputs = outputs[finite_out]
            grasp = grasp[finite_out]
            if idxs is not None:
                idxs = idxs[finite_out]
            if grasp.size(0) == 0:
                continue

        loss = criterion(outputs, grasp)
        if not torch.isfinite(loss).item():
            print(f"[WARN] NaN/Inf en loss (TRAIN, batch {batch_idx}), se salta el batch.")
            _append_bad_indices(bad_indices_path, "train", batch_idx, "nonfinite_loss", idxs)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = grasp.size(0)
        running_loss += float(loss.item()) * bs
        total_samples += bs

    if total_samples == 0:
        print("[WARN] Ningún batch válido en train_one_epoch (total_samples=0). Devuelvo train_loss=0.0")
        return 0.0

    return running_loss / total_samples


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    use_depth: bool,
    bad_indices_path: Optional[Path] = None,
):
    model.eval()

    if "metrics" not in cfg:
        raise KeyError("Falta la sección 'metrics' en el YAML.")
    iou_thresh = float(cfg["metrics"]["iou_thresh"])
    angle_thresh = float(cfg["metrics"]["angle_thresh"])

    total_loss = 0.0
    total_samples = 0

    sum_iou = 0.0
    sum_angle = 0.0
    n_eval = 0
    success_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            rgb, depth, grasp = _get_batch_tensors(batch, device=device, use_depth=use_depth)
            idxs = _get_batch_indices(batch)
            x = torch.cat([rgb, depth], dim=1) if use_depth else rgb

            finite = _finite_rows(x) & torch.isfinite(grasp).all(dim=1)
            if not finite.all():
                bad = (~finite).sum().item()
                total = grasp.size(0)
                print(f"[WARN] NaN/Inf en VAL (x/grasp): {bad}/{total} (batch {batch_idx}). Se filtran.")
                _append_bad_indices(bad_indices_path, "val", batch_idx, "nonfinite_input", idxs[~finite] if idxs is not None else None)
                x = x[finite]
                grasp = grasp[finite]
                if idxs is not None:
                    idxs = idxs[finite]
                if grasp.size(0) == 0:
                    continue

            outputs_raw = model(x)
            outputs = _extract_model_output(outputs_raw)

            finite_out = torch.isfinite(outputs).all(dim=1)
            if not finite_out.all():
                bad = (~finite_out).sum().item()
                total = outputs.size(0)
                print(f"[WARN] NaN/Inf en outputs VAL: {bad}/{total} (batch {batch_idx}). Se filtran.")
                _append_bad_indices(bad_indices_path, "val", batch_idx, "nonfinite_output", idxs[~finite_out] if idxs is not None else None)
                outputs = outputs[finite_out]
                grasp = grasp[finite_out]
                if idxs is not None:
                    idxs = idxs[finite_out]
                if grasp.size(0) == 0:
                    continue

            loss = criterion(outputs, grasp)
            if not torch.isfinite(loss).item():
                print(f"[WARN] NaN/Inf en loss (VAL, batch {batch_idx}), se salta el batch.")
                _append_bad_indices(bad_indices_path, "val", batch_idx, "nonfinite_loss", idxs)
                continue

            # ✅ bs después de filtrar
            bs = grasp.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            preds_np = outputs.detach().cpu().numpy()
            grasp_np = grasp.detach().cpu().numpy()

            for p, g in zip(preds_np, grasp_np):
                p = _sanitize_params_np(p)
                g = _sanitize_params_np(g)

                iou = grasp_iou(p, g)
                ang = angle_diff_deg(float(p[4]), float(g[4]))

                if np.isnan(iou) or np.isnan(ang):
                    continue

                success = (iou >= iou_thresh) and (ang <= angle_thresh)

                sum_iou += float(iou)
                sum_angle += float(ang)
                n_eval += 1
                if success:
                    success_count += 1

    if total_samples == 0:
        print("[WARN] Ningún batch válido en validate() (total_samples=0). Devuelvo métricas neutras.")
        return 0.0, 0.0, 0.0, 0.0

    mean_loss = total_loss / total_samples
    denom = max(n_eval, 1)
    mean_iou = sum_iou / denom
    mean_angle = sum_angle / denom
    val_success = success_count / denom

    return mean_loss, mean_iou, mean_angle, val_success


# =============================================================================
# main
# =============================================================================


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ------------------------------------------------------------
    # EXP DIR por seed (auto): EXP_NAME_seed{N}
    # ------------------------------------------------------------
    if "experiment_name" not in cfg or not str(cfg["experiment_name"]).strip():
        raise KeyError("Falta 'experiment_name' en el YAML.")

    base_exp_name = str(cfg["experiment_name"]).strip()

    # Si ya viene con _seedX, no lo duplicamos
    if "_seed" not in base_exp_name:
        cfg["experiment_name"] = f"{base_exp_name}_seed{args.seed}"
    else:
        cfg["experiment_name"] = base_exp_name


    # guardo seed efectivo para logs/funciones
    cfg.setdefault("train", {})
    cfg["train"]["seed_effective"] = int(args.seed)

    # Determinismo: por defecto True (como estabas), pero manejamos CuBLAS
    deterministic = bool(cfg.get("train", {}).get("deterministic", True))
    seed_everything(args.seed, deterministic=deterministic)

    train_cfg = cfg.get("train", {})
    dev_req = str(train_cfg.get("device", "auto")).lower().strip()

    if dev_req == "cpu":
        device = torch.device("cpu")
    elif dev_req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se ha pedido device=cuda pero torch.cuda.is_available()=False")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Usando dispositivo:", device)

    # Si determinismo+CUDA, CuBLAS necesita env var; si no está, auto-desactivamos (o abortamos si strict)
    ensure_cublas_determinism_or_disable(cfg, device)

    data_cfg = cfg.get("data", {})
    use_depth = bool(data_cfg.get("use_depth", False))
    in_channels = 4 if use_depth else 3

    train_loader, val_loader = make_dataloaders(cfg, use_depth=use_depth)
    model = make_model(cfg, device, in_channels=in_channels)

    # Obligatorios
    for k in ("lr", "weight_decay", "num_epochs"):
        if k not in train_cfg:
            raise KeyError(f"En 'train' falta clave obligatoria: {k}")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    base_dir, ckpt_dir, metrics_path = ensure_dirs(cfg)
    bad_train_path = base_dir / "bad_samples_train.csv"
    bad_val_path = base_dir / "bad_samples_val.csv"

    # Copia de config usada
    config_path = Path(args.config)
    try:
        shutil.copy2(config_path, base_dir / "config_used.yaml")
    except Exception as e:
        print(f"[WARN] No se pudo copiar config_used.yaml: {e}")

    num_epochs = int(train_cfg["num_epochs"])
    log_cfg = cfg.get("logging", {})
    metric_name = str(log_cfg.get("save_best_by", "val_success")).strip()

    valid_metric_names = {"val_loss", "val_iou", "val_angle", "val_success", "train_loss"}
    if metric_name not in valid_metric_names:
        print(
            f"[WARN] save_best_by='{metric_name}' no es válido. "
            f"Usaré 'val_success'. Opciones: {sorted(valid_metric_names)}"
        )
        metric_name = "val_success"

    best_metric = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_depth=use_depth,
            bad_indices_path=bad_train_path,
        )
        val_loss, val_iou, val_angle, val_success = validate(
            model,
            val_loader,
            criterion,
            device,
            cfg,
            use_depth=use_depth,
            bad_indices_path=bad_val_path,
        )

        metrics_dict = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_iou": float(val_iou),
            "val_angle": float(val_angle),
            "val_success": float(val_success),
        }

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_angle={val_angle:.2f} | "
            f"val_success={val_success:.4f}"
        )

        append_metrics_row(metrics_path, metrics_dict)

        current = metrics_dict[metric_name]
        if best_metric is None:
            best_metric = current
            is_best = True
        else:
            if metric_name in ("val_loss", "val_angle"):
                is_best = current < best_metric
            else:
                is_best = current > best_metric
            if is_best:
                best_metric = current

        save_checkpoint(ckpt_dir, model, optimizer, metrics_dict, is_best=is_best)

    print(f"Entrenamiento terminado. Resultados en: {base_dir}")


if __name__ == "__main__":
    main()
