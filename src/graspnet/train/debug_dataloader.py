import torch
from torch.utils.data import DataLoader

from graspnet.data.cornell_dataset import CornellGraspDataset


def main():
    """
    Script de debug rápido para comprobar que el DataLoader
    devuelve tensores con las shapes esperadas.

    Esperado:
      - RGB:   [B, 3, H, W]
      - Depth: [B, 1, H, W]
      - Grasp: [B, 5]
    """
    dataset = CornellGraspDataset(
        root_dir="data/cornell",  # ajusta si usas otra ruta
        split="train",
        val_split=0.2,
        img_size=224,
        use_depth=True,           # activamos carga de profundidad
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,            # 0 para debug, luego puedes subirlo
    )

    batch = next(iter(loader))

    rgb = batch["rgb"]
    depth = batch["depth"]
    grasp = batch["grasp"]

    print("RGB:", rgb.shape)
    print("Depth:", depth.shape)
    print("Grasp:", grasp.shape)

    # Comprobaciones extra opcionales
    print("RGB contains NaN?   ", torch.isnan(rgb).any().item())
    print("Depth contains NaN? ", torch.isnan(depth).any().item())
    print("Grasp contains NaN?", torch.isnan(grasp).any().item())


if __name__ == "__main__":
    main()
