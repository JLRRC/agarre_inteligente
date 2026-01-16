from .simple_cnn import SimpleGraspCNN


def build_model(
    name: str,
    in_channels: int = 3,
    pretrained: bool = False,
    img_size: int = 224,
):
    """
    Crea el modelo a partir de un nombre y de los canales de entrada.

    name:
      - "simple_cnn"  -> SimpleGraspCNN
      - "resnet18"    -> ResNet18Grasp
    """
    name = name.lower()

    if name in ("simple", "simple_cnn", "simplegraspcnn", "simplegrasp_cnn"):
        return SimpleGraspCNN(in_channels=in_channels, img_size=img_size)

    elif name in ("resnet18", "resnet18grasp", "resnet18_grasp"):
        try:
            from .resnet18_grasp import ResNet18Grasp
        except Exception:
            raise RuntimeError("ResNet18Grasp no disponible (torchvision no importable).")
        return ResNet18Grasp(in_channels=in_channels, pretrained=pretrained)

    else:
        raise ValueError(f"Modelo no reconocido en build_model: {name}")
