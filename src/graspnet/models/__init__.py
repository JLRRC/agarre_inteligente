from typing import Any

from .simple_cnn import SimpleGraspCNN
from .resnet18_grasp import ResNet18Grasp


def build_model(
    name: str,
    in_channels: int = 3,
    img_size: int = 224,
    pretrained: bool = False,
    **kwargs: Any,
):
    """
    Crea y devuelve el modelo pedido en el YAML.

    Args:
        name: nombre del modelo, p.ej. "simple_cnn" o "resnet18".
        in_channels: nº de canales de entrada (3 = RGB, 4 = RGB+depth, ...).
        img_size: tamaño de la imagen (solo lo usa SimpleGraspCNN).
        pretrained: si True y está disponible, carga ResNet18 con pesos ImageNet.
        **kwargs: parámetros extra que quieras pasar al constructor.

    Returns:
        Instancia de nn.Module (SimpleGraspCNN o ResNet18Grasp).
    """
    name = name.lower()

    # Modelo CNN sencillo para baseline
    if name in ("simple_cnn", "simple", "baseline"):
        return SimpleGraspCNN(
            in_channels=in_channels,
            img_size=img_size,
            **kwargs,
        )

    # Modelo basado en ResNet18
    if name in ("resnet18", "resnet18_grasp", "resnet"):
        return ResNet18Grasp(
            in_channels=in_channels,
            pretrained=pretrained,
            **kwargs,
        )

    # Si el nombre no coincide con ninguno de los anteriores, error claro
    raise ValueError(f"Modelo no reconocido: {name}")
