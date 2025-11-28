import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Grasp(nn.Module):
    """
    ResNet-18 adaptada para regresión de parámetros de agarre:
      - Entrada: [B, C, H, W]  con C = 3 (RGB) o C = 4 (RGB-D).
      - Salida:  [B, 5]        (cx, cy, w, h, angle_deg)
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()

        # Cargar backbone con pesos o sin ellos
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

        # Adaptar la primera convolución si no son 3 canales
        if in_channels != 3:
            old_conv1 = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=False,
            )
            # Inicializamos de nuevo conv1 (no usamos los pesos preentrenados aquí)
            nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Cambiar la capa fully-connected final para sacar 5 parámetros
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 5)

        self.backbone = backbone

    def forward(self, x):
        """
        x: [B, C, H, W]  con C = 3 o 4
        """
        return self.backbone(x)
