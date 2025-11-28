from .simple_cnn import SimpleGraspCNN
from .resnet18_grasp import ResNet18Grasp


def build_model(name: str, in_channels: int = 3, pretrained: bool = False):
    """
    Crea el modelo a partir de un nombre y de los canales de entrada.

    name:
      - "simple_cnn"  -> SimpleGraspCNN
      - "resnet18"    -> ResNet18Grasp
    """
    name = name.lower()

    if name in ("simple", "simple_cnn", "simplegraspcnn", "simplegrasp_cnn"):
        return SimpleGraspCNN(in_channels=in_channels)

    elif name in ("resnet18", "resnet18grasp", "resnet18_grasp"):
        return ResNet18Grasp(in_channels=in_channels, pretrained=pretrained)

    else:
        raise ValueError(f"Modelo no reconocido en build_model: {name}")
