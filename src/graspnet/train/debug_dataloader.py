import torch
from torch.utils.data import DataLoader

from graspnet.datasets.cornell_dataset import CornellGraspDataset


def main():
    dataset = CornellGraspDataset(
        root_dir="data/cornell",
        split="train",
        use_depth=True,   # <- MUY IMPORTANTE: queremos cargar D
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    rgb = batch["rgb"]      # [B, 3, H, W]
    depth = batch["depth"]  # [B, 1, H, W]
    grasp = batch["grasp"]  # [B, 5]

    print("RGB:", rgb.shape)
    print("Depth:", depth.shape)
    print("Grasp:", grasp.shape)


if __name__ == "__main__":
    main()

