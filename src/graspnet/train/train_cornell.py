import argparse
import csv
import random
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from graspnet.datasets.cornell_dataset import CornellGraspDataset
from graspnet.models import build_model
from graspnet.utils.metrics import (
    params_to_rect,
    rect_to_bbox,
    bbox_iou,
    angle_diff_deg,
    compute_grasp_success,
)

# -------------------------------------------------------------------------
#  Argumentos y configuración
# -------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento en Cornell Grasping Dataset (RGB / RGB-D)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al fichero de configuración YAML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semilla aleatoria",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Carga un fichero YAML de configuración y lo devuelve como dict."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de config: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def seed_everything(seed: int):
    """Fija semillas para Python, NumPy y PyTorch para mejorar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
#  Data: CornellGraspDataset + DataLoaders
# -------------------------------------------------------------------------


def make_dataloaders(cfg: dict, use_depth: bool):
    """
    Crea los DataLoaders de entrenamiento y validación a partir del YAML.

    El CornellGraspDataset devuelve diccionarios con:
      - "rgb": tensor [B, 3, H, W]
      - "depth": tensor [B, 1, H, W]
      - "grasp": tensor [B, 5] con (cx, cy, w, h, angle)
    """
    data_cfg = cfg["data"]
    root_dir = data_cfg["root_dir"]
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 4))
    img_size = int(data_cfg.get("img_size", 224))
    val_split = float(data_cfg.get("val_split", 0.2))

    aug_cfg = data_cfg.get("augmentation", {})
    train_aug = {
        "geometric": bool(aug_cfg.get("geometric", False)),
        "photometric": bool(aug_cfg.get("photometric", False)),
    }

    # Dataset de entrenamiento con augmentation
    train_dataset = CornellGraspDataset(
        root_dir=root_dir,
        split="train",
        val_split=val_split,
        img_size=img_size,
        use_depth=use_depth,
        augmentation=train_aug,
    )

    # Dataset de validación sin augmentation
    val_dataset = CornellGraspDataset(
        root_dir=root_dir,
        split="val",
        val_split=val_split,
        img_size=img_size,
        use_depth=use_depth,
        augmentation=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# -------------------------------------------------------------------------
#  Modelo + directorios de salida
# -------------------------------------------------------------------------


def make_model(cfg: dict, device: torch.device, in_channels: int) -> nn.Module:
    """
    Construye el modelo a partir del YAML.
    Espera en cfg["model"]["name"] algo tipo:
        - "simple_cnn"
        - "resnet18"

    Se pasa in_channels para soportar:
      - 3 canales (RGB)
      - 4 canales (RGB + Depth)
    """
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    pretrained = bool(model_cfg.get("pretrained", False))

    model = build_model(
        model_name,
        in_channels=in_channels,
        pretrained=pretrained,
    )
    model.to(device)
    return model


def ensure_dirs(cfg: dict):
    """
    Crea la carpeta de experimento, la subcarpeta de checkpoints y
    devuelve:
      - base_dir: ruta a experiments/<experiment_name>
      - ckpt_dir: ruta a experiments/<experiment_name>/checkpoints
      - metrics_path: ruta al metrics.csv
    """
    log_cfg = cfg["logging"]
    output_root = Path(log_cfg.get("output_dir", "experiments"))
    exp_name = cfg["experiment_name"]

    base_dir = output_root / exp_name
    ckpt_dir = base_dir / "checkpoints"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = base_dir / "metrics.csv"
    return base_dir, ckpt_dir, metrics_path


# -------------------------------------------------------------------------
#  Utilidades de logging: metrics.csv y checkpoints
# -------------------------------------------------------------------------


def append_metrics_row(metrics_path: Path, metrics_dict: dict):
    """
    Añade una fila al metrics.csv. Si no existe, escribe primero la cabecera.
    Columnas esperadas (en este orden):
        epoch, train_loss, val_loss, val_iou, val_angle, val_success
    """
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
    """
    Guarda checkpoints:
      - always: last.pth
      - si is_best: best.pth
    """
    epoch = metrics_dict["epoch"]
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics_dict,
    }

    last_path = ckpt_dir / "last.pth"
    torch.save(state, last_path)

    if is_best:
        best_path = ckpt_dir / "best.pth"
        torch.save(state, best_path)


# -------------------------------------------------------------------------
#  Bucle de entrenamiento + validación
# ------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_depth: bool,
) -> float:
    """
    Entrena una época sobre el dataloader y devuelve la pérdida media.

    El dataloader devuelve batches como diccionarios con keys:
      - "rgb": tensor [B, 3, H, W]
      - "depth": tensor [B, 1, H, W]
      - "grasp": tensor [B, 5]

    Si use_depth=False:
      - Se usa solo rgb -> [B, 3, H, W]
    Si use_depth=True:
      - Se concatena rgb y depth -> [B, 4, H, W]
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device)      # [B, 3, H, W]
        depth = batch["depth"].to(device)  # [B, 1, H, W]
        grasp = batch["grasp"].to(device)  # [B, 5]

        if use_depth:
            x = torch.cat([rgb, depth], dim=1)  # [B, 4, H, W]
        else:
            x = rgb  # [B, 3, H, W]

        # Chequeos de sanity antes del forward
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARN] NaN/Inf en entrada (batch {batch_idx}), se salta el batch.")
            continue
        if torch.isnan(grasp).any() or torch.isinf(grasp).any():
            print(f"[WARN] NaN/Inf en etiquetas grasp (batch {batch_idx}), se salta el batch.")
            continue

        optimizer.zero_grad()
        outputs = model(x)

        # Chequeo de outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"[WARN] NaN/Inf en outputs del modelo (batch {batch_idx}), se salta el batch.")
            continue

        loss = criterion(outputs, grasp)

        # Chequeo de la loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] NaN/Inf en loss (batch {batch_idx}), se salta el batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = grasp.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        # Si TODOS los batches han sido inválidos, devolvemos 0.0 para no propagar NaNs
        print("[WARN] Ningún batch válido en train_one_epoch (total_samples=0). Devuelvo train_loss=0.0")
        return 0.0

    mean_loss = running_loss / total_samples
    return mean_loss

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    use_depth: bool,
):
    """
    Evalúa el modelo en validación y devuelve:
      - val_loss
      - val_iou medio
      - val_angle medio (diferencia angular en grados)
      - val_success (proporción de aciertos según Cornell)
    """
    model.eval()
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
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            grasp = batch["grasp"].to(device)

            if use_depth:
                x = torch.cat([rgb, depth], dim=1)  # [B, 4, H, W]
            else:
                x = rgb  # [B, 3, H, W]

            # Chequeos de sanity
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[WARN] NaN/Inf en entrada (VAL, batch {batch_idx}), se salta el batch.")
                continue
            if torch.isnan(grasp).any() or torch.isinf(grasp).any():
                print(f"[WARN] NaN/Inf en etiquetas grasp (VAL, batch {batch_idx}), se salta el batch.")
                continue

            outputs = model(x)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"[WARN] NaN/Inf en outputs (VAL, batch {batch_idx}), se salta el batch.")
                continue

            loss = criterion(outputs, grasp)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] NaN/Inf en loss (VAL, batch {batch_idx}), se salta el batch.")
                continue

            batch_size = grasp.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Métricas tipo Cornell
            preds_np = outputs.cpu().numpy()
            grasp_np = grasp.cpu().numpy()

            for p, g in zip(preds_np, grasp_np):
                cx_p, cy_p, w_p, h_p, ang_p = p
                cx_g, cy_g, w_g, h_g, ang_g = g

                rect_p = params_to_rect(cx_p, cy_p, w_p, h_p, ang_p)
                rect_g = params_to_rect(cx_g, cy_g, w_g, h_g, ang_g)

                bbox_p = rect_to_bbox(rect_p)
                bbox_g = rect_to_bbox(rect_g)

                iou = bbox_iou(bbox_p, bbox_g)
                angle_diff = angle_diff_deg(ang_p, ang_g)

                success = compute_grasp_success(p, g, iou_thresh, angle_thresh)

                # iou o angle_diff no deberían ser NaN, pero por si acaso:
                if np.isnan(iou) or np.isnan(angle_diff):
                    continue

                sum_iou += iou
                sum_angle += angle_diff
                n_eval += 1
                if success:
                    success_count += 1

    if total_samples == 0:
        print("[WARN] Ningún batch válido en validate() (total_samples=0). Devuelvo métricas neutras.")
        return 0.0, 0.0, 0.0, 0.0

    mean_loss = total_loss / total_samples
    mean_iou = sum_iou / max(n_eval, 1)
    mean_angle = sum_angle / max(n_eval, 1)
    val_success = success_count / max(n_eval, 1)

    return mean_loss, mean_iou, mean_angle, val_success


# -------------------------------------------------------------------------
#  main()
# -------------------------------------------------------------------------


def main():
    # 1) Argumentos cli
    args = parse_args()

    # 2) Semillas
    seed_everything(args.seed)

    # 3) Config
    cfg = load_config(args.config)

    # ⚠️ En el MeLE NO hay GPU NVIDIA: forzamos SIEMPRE CPU
    device = torch.device("cpu")
    print("Usando dispositivo:", device)

    # 4) Config de datos: decidir si usamos profundidad
    data_cfg = cfg["data"]
    use_depth = bool(data_cfg.get("use_depth", False))
    in_channels = 4 if use_depth else 3

    # 5) DataLoaders
    train_loader, val_loader = make_dataloaders(cfg, use_depth=use_depth)

    # 6) Modelo
    model = make_model(cfg, device, in_channels=in_channels)

    # 7) Loss y optimizador
    train_cfg = cfg["train"]
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # 8) Directorios y metrics.csv
    base_dir, ckpt_dir, metrics_path = ensure_dirs(cfg)

    # Guardar copia de la config usada (para reproducibilidad del experimento)
    config_path = Path(args.config)
    shutil.copy2(config_path, base_dir / "config_used.yaml")

    # 9) Bucle de entrenamiento
    num_epochs = int(train_cfg["num_epochs"])
    log_cfg = cfg["logging"]
    metric_name = log_cfg.get("save_best_by", "val_success")
    best_metric = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_depth=use_depth
        )
        val_loss, val_iou, val_angle, val_success = validate(
            model, val_loader, criterion, device, cfg, use_depth=use_depth
        )

        metrics_dict = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_iou": float(val_iou),
            "val_angle": float(val_angle),
            "val_success": float(val_success),
        }

        # Log a consola
        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_angle={val_angle:.2f} | "
            f"val_success={val_success:.4f}"
        )

        # Guardar en metrics.csv
        append_metrics_row(metrics_path, metrics_dict)

        # Selección de mejor modelo según metric_name
        current = metrics_dict[metric_name]
        if best_metric is None:
            best_metric = current
            is_best = True
        else:
            if metric_name == "val_loss":
                is_best = current < best_metric
            else:
                is_best = current > best_metric
            if is_best:
                best_metric = current

        # Guardar checkpoints
        save_checkpoint(ckpt_dir, model, optimizer, metrics_dict, is_best=is_best)

    print(f"Entrenamiento terminado. Resultados en: {base_dir}")


if __name__ == "__main__":
    main()
