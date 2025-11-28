import torch
import torch.nn as nn


class SimpleGraspCNN(nn.Module):
    """
    CNN sencilla para regresión de 5 parámetros de agarre:
    (cx, cy, w, h, angle).

    in_channels puede ser:
      - 3  -> RGB
      - 4  -> RGB-D (RGB + Depth concatenado)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Forzamos salida a tamaño fijo 7x7
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5),  # (cx, cy, w, h, angle)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x
