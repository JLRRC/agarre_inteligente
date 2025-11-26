import torch
from torch.utils.data import DataLoader
from graspnet.datasets.cornell_dataset import CornellGraspDataset

def main():
    dataset = CornellGraspDataset(
        root_dir="data/cornell_raw/cornell_grasp",
        split="train",  # o lo que uses
        # añade aquí transforms si ya los tienes
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    batch = next(iter(loader))
    rgb = batch["rgb"]
    depth = batch["depth"]
    grasp = batch["grasp"]

    print("RGB:", rgb.shape)
    print("Depth:", depth.shape)
    print("Grasp:", grasp.shape)

if __name__ == "__main__":
    main()
