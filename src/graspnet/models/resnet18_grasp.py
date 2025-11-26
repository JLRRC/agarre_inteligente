import torch
import torch.nn as nn

from torchvision.models import resnet18
try:
    # Torchvision >= 0.13
    from torchvision.models import ResNet18_Weights
except ImportError:
    ResNet18_Weights = None


class ResNet18Grasp(nn.Module):
    """
    Wrapper de ResNet-18 para regresión de parámetros de agarre tipo Cornell.

    Entrada:
        x: tensor [B, C, H, W]
           - C = 3 si solo RGB
           - C = 4 si concatenas RGB + depth (u otros canales)

    Salida:
        out: tensor [B, 5]
             (cx, cy, w, h, angle_deg)
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        # 1) Cargar backbone ResNet-18
        if pretrained and ResNet18_Weights is not None:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = resnet18(weights=weights)

        # 2) Adaptar la primera conv si C != 3
        if in_channels != 3:
            old_conv1 = self.backbone.conv1
            new_conv1 = nn.Conv2d(
                in_channels,
                old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias is not None,
            )

            # Inicialización razonable si venía preentrenada
            with torch.no_grad():
                if in_channels >= 3:
                    new_conv1.weight[:, :3, :, :] = old_conv1.weight
                    if in_channels > 3:
                        new_conv1.weight[:, 3:, :, :].zero_()
                else:
                    new_conv1.weight[:, :in_channels, :, :] = \
                        old_conv1.weight[:, :in_channels, :, :]

            self.backbone.conv1 = new_conv1

        # 3) Reemplazar la fc final por una de 5
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, 5]  (cx, cy, w, h, angle_deg)
        """
        out = self.backbone(x)
        return out
