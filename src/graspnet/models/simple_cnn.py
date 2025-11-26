import torch
import torch.nn as nn
from typing import Optional


class SimpleGraspCNN(nn.Module):
    """
    CNN sencilla para regresión de parámetros de agarre tipo Cornell.

    Entrada:
        x: tensor [B, C, H, W]
           - C = 3 si solo RGB
           - C = 4 si concatenas RGB + depth

    Salida:
        out: tensor [B, 5]
             (cx, cy, w, h, angle_deg)
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Bloques convolucionales sencillos
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/2 x W/2

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/4 x W/4

            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/8 x W/8

            # Bloque 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/16 x W/16
        )

        # Calcular automáticamente el tamaño después de las convoluciones
        feat_dim = self._get_feature_dim(in_channels, img_size)

        fc_layers = [
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            fc_layers.append(nn.Dropout(p=dropout))

        # Capa final: 5 parámetros (cx, cy, w, h, angle_deg)
        fc_layers.append(nn.Linear(hidden_dim, 5))

        self.regressor = nn.Sequential(*fc_layers)

    def _get_feature_dim(self, in_channels: int, img_size: int) -> int:
        """Pasa un tensor dummy para calcular el nº de features tras self.features."""
        with torch.no_grad():
            x = torch.zeros(1, in_channels, img_size, img_size)
            y = self.features(x)
            return y.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, 5]
        """
        feats = self.features(x)               # [B, C', H', W']
        feats = feats.view(feats.size(0), -1)  # [B, F]
        out = self.regressor(feats)            # [B, 5]
        return out
