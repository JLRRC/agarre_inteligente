import os
import glob
import math
import random
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CornellGraspDataset(Dataset):
    """
    Dataset para el Cornell Grasping Dataset.

    Cada sample contiene:
      - rgb: imagen RGB [3, H, W]
      - depth: imagen de profundidad [1, H, W]
      - grasp: vector (cx, cy, w, h, angle_deg)

    Args:
        root_dir: Carpeta raíz donde está Cornell (carpetas 01, 02, ... o pcdXXXX*).
                  En tu estructura, puede ser data/cornell, data/cornell_processed o
                  data/cornell_raw; el dataset intenta resolverlo automáticamente.
        split: "train", "val" o "all".
        val_split: Proporción de samples para validación (ej: 0.2 = 80/20).
        img_size: Tamaño final (img_size x img_size).
        include_depth: De momento no se usa para recortar nada; siempre se carga depth.
        random_grasp: Si True elige un grasp aleatorio del cpos; si False, el primero.
        use_depth: Bandera para indicar si el modelo va a usar profundidad (pipeline RGB-D).
        augment: Atajo rápido: si True -> augmentation={"geometric": True, "photometric": True}.
        augmentation: Diccionario opcional con flags de augmentation, por ejemplo:
                      {"geometric": True, "photometric": True}.
                      Si se pasa, tiene prioridad sobre augment.
        transform_rgb: Transform opcional para RGB (TorchVision, etc.).
        transform_depth: Transform opcional para depth.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_split: float = 0.2,
        img_size: int = 224,
        include_depth: bool = True,
        random_grasp: bool = True,
        use_depth: bool = False,
        augment: bool = False,
        augmentation: Optional[Dict[str, bool]] = None,
        transform_rgb=None,
        transform_depth=None,
    ):
        assert split in ("train", "val", "all"), f"split inválido: {split}"

        # Guardamos la ruta tal cual y luego la resolvemos
        self.root_dir = root_dir
        self.split = split
        self.val_split = val_split
        self.img_size = img_size
        self.include_depth = include_depth
        self.random_grasp = random_grasp

        # Para el pipeline: saber si se piensa usar profundidad en el modelo
        self.use_depth = use_depth

        # Manejo de augmentation: primero `augmentation` explícito, si no, `augment`
        if augmentation is not None:
            self.augmentation = augmentation
        else:
            self.augmentation = {"geometric": True, "photometric": True} if augment else {}

        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth

        # Resolver automáticamente la carpeta real (cornell / cornell_processed / cornell_raw)
        self.root_dir = self._resolve_root_dir(self.root_dir)

        # Construimos la lista de samples
        self.samples: List[Dict[str, str]] = self._collect_samples()

        if split != "all":
            self._apply_split()

    # --------------------------------------------------------------------- #
    #   Resolución de root_dir
    # --------------------------------------------------------------------- #
    @staticmethod
    def _resolve_root_dir(root_dir: str) -> str:
        """
        Intenta encontrar una carpeta válida a partir de root_dir.
        Soporta estructuras como:
          - data/cornell
          - data/cornell_processed
          - data/cornell_raw

        Prioridad:
          1) Lo que se pasa si existe tal cual.
          2) cornell_raw
          3) cornell_processed
        """
        # Si ya existe tal cual, perfecto
        if os.path.isdir(root_dir):
            return root_dir

        base_name = os.path.basename(root_dir)
        parent = os.path.dirname(root_dir)

        candidates = []

        # Caso típico: root_dir = "data/cornell"
        if base_name == "cornell":
            # Preferimos usar el dataset RAW original
            candidates.append(os.path.join(parent, "cornell_raw"))
            # Y en segundo lugar el processed
            candidates.append(os.path.join(parent, "cornell_processed"))

        # También probamos sufijos por si acaso
        candidates.append(root_dir + "_raw")
        candidates.append(root_dir + "_processed")

        for c in candidates:
            if c and os.path.isdir(c):
                return c

        # Si nada ha funcionado, error explícito
        raise FileNotFoundError(
            f"Directorio raíz no encontrado: {root_dir}. "
            f"Probadas alternativas: {', '.join(candidates)}"
        )

    # --------------------------------------------------------------------- #
    #   Construcción de la lista de samples
    # --------------------------------------------------------------------- #
    def _collect_samples(self) -> List[Dict[str, str]]:
        """
        Recorre root_dir de forma RECURSIVA buscando ficheros *cpos.txt
        y, a partir de ahí, intenta encontrar la RGB y la depth correspondientes.

        Es más robusto que asumir siempre pcdXXXXr.png + pcdXXXXd.tiff
        y que todo esté solo a un nivel de profundidad.
        """
        samples: List[Dict[str, str]] = []

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Directorio raíz no encontrado: {self.root_dir}")

        # Recorremos recursivamente todos los subdirectorios
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Filtramos todos los ficheros que terminen en "cpos.txt"
            cpos_files = [f for f in filenames if f.endswith("cpos.txt")]
            for cpos_name in cpos_files:
                cpos_path = os.path.join(dirpath, cpos_name)

                # Quitamos el sufijo "cpos.txt" para construir el "base"
                base = cpos_path[:-8]  # quita 'cpos.txt'

                # Candidatos para RGB
                rgb_candidates = [
                    base + "r.png",
                    base + "r.jpg",
                    base + "r.jpeg",
                ]

                # Candidatos para depth
                depth_candidates = [
                    base + "d.tiff",
                    base + "d.tif",
                    base + "d.png",
                    base + "d.jpg",
                ]

                rgb_path = None
                depth_path = None

                for rp in rgb_candidates:
                    if os.path.exists(rp):
                        rgb_path = rp
                        break

                for dp in depth_candidates:
                    if os.path.exists(dp):
                        depth_path = dp
                        break

                # Si falta alguno de los dos, saltamos este sample
                if rgb_path is None or depth_path is None:
                    continue

                samples.append(
                    {
                        "rgb": rgb_path,
                        "depth": depth_path,
                        "cpos": cpos_path,
                    }
                )

        if not samples:
            raise RuntimeError(
                f"No se han encontrado samples válidos en {self.root_dir}. "
                f"¿Existen ficheros *cpos.txt y sus imágenes RGB/Depth asociadas "
                f"(r.png/jpg y d.tif/tiff/png/jpg)?"
            )

        return samples

    def _apply_split(self):
        """Aplica split train/val sobre self.samples."""
        n_total = len(self.samples)
        n_val = int(round(n_total * self.val_split))
        n_train = n_total - n_val

        # División determinista (sin aleatorio) para reproducibilidad simple.
        if self.split == "train":
            self.samples = self.samples[:n_train]
        elif self.split == "val":
            self.samples = self.samples[n_train:]

    # --------------------------------------------------------------------- #
    #   Utilidades de lectura y conversión de grasp
    # --------------------------------------------------------------------- #
    @staticmethod
    def _load_grasp_rects(cpos_path: str) -> np.ndarray:
        """
        Lee un fichero cpos.txt y devuelve un array [N, 4, 2]:
            N = nº de grasp rects,
            4 = nº de vértices,
            2 = (x, y).
        Cornell suele tener 4 líneas por grasp, cada línea "x y".
        """
        with open(cpos_path, "r") as f:
            lines = f.readlines()

        coords: List[Tuple[float, float]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            coords.append((x, y))

        if len(coords) == 0 or len(coords) % 4 != 0:
            raise RuntimeError(
                f"Formato inesperado en {cpos_path}: "
                f"{len(coords)} coordenadas, debería ser múltiplo de 4."
            )

        coords_np = np.array(coords, dtype=np.float32).reshape(-1, 4, 2)
        return coords_np  # [N, 4, 2]

    @staticmethod
    def _rect_to_params(rect: np.ndarray) -> np.ndarray:
        """
        Convierte un rectángulo (4 x 2) -> (cx, cy, w, h, angle_deg)
        """
        if rect.shape != (4, 2):
            raise ValueError(f"rect shape inválido: {rect.shape}, esperado (4, 2)")

        p0, p1, p2, p3 = rect

        # Centro (media de los 4 puntos)
        cx = rect[:, 0].mean()
        cy = rect[:, 1].mean()

        # Largo y ancho
        dx_w = p1[0] - p0[0]
        dy_w = p1[1] - p0[1]
        width = math.hypot(dx_w, dy_w)

        dx_h = p2[0] - p1[0]
        dy_h = p2[1] - p1[1]
        height = math.hypot(dx_h, dy_h)

        # Ángulo en radianes y luego grados
        angle_rad = math.atan2(dy_w, dx_w)
        angle_deg = math.degrees(angle_rad)

        return np.array([cx, cy, width, height, angle_deg], dtype=np.float32)

    def _load_grasp_params(
        self, cpos_path: str, orig_w: int, orig_h: int
    ) -> np.ndarray:
        """
        Carga todos los grasp rects, elige uno y lo convierte a (cx, cy, w, h, angle).
        """
        rects = self._load_grasp_rects(cpos_path)  # [N, 4, 2]
        n_rects = rects.shape[0]

        if self.random_grasp:
            idx = random.randint(0, n_rects - 1)
        else:
            idx = 0

        rect = rects[idx]  # [4, 2]
        params = self._rect_to_params(rect)  # (cx, cy, w, h, angle_deg)

        # Sanity check simple
        cx, cy, w, h, _ = params
        if not (0 <= cx <= orig_w and 0 <= cy <= orig_h):
            # Podrías hacer raise aquí si quieres ser estricto
            pass

        return params

    # --------------------------------------------------------------------- #
    #   Métodos del Dataset
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        rgb_path = sample["rgb"]
        depth_path = sample["depth"]
        cpos_path = sample["cpos"]

        # ---------------------------
        # 1) Cargar RGB (BGR -> RGB)
        # ---------------------------
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"No se pudo leer la imagen RGB: {rgb_path}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = rgb.shape

        # ---------------------------
        # 2) Cargar depth
        # ---------------------------
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"No se pudo leer la imagen de profundidad: {depth_path}")

        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        # ---------------------------
        # 3) Cargar grasp params (en coords originales)
        # ---------------------------
        grasp_params = self._load_grasp_params(cpos_path, orig_w, orig_h)
        cx, cy, w, h, angle_deg = grasp_params

        # ---------------------------
        # 4) Redimensionar imágenes a img_size x img_size
        # ---------------------------
        if self.img_size is not None:
            new_size = (self.img_size, self.img_size)

            rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)

            # Escalar coordenadas de grasp
            scale_x = self.img_size / float(orig_w)
            scale_y = self.img_size / float(orig_h)

            cx *= scale_x
            cy *= scale_y
            w *= scale_x
            h *= scale_y

        # ---------------------------
        # 5) Normalizar y pasar a Tensor
        # ---------------------------

        # RGB: [H, W, 3] uint8 -> float32 [0,1] -> [3, H, W]
        rgb = rgb.astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))  # CHW

        rgb_tensor = torch.from_numpy(rgb)  # [3, H, W]
        if self.transform_rgb is not None:
            rgb_tensor = self.transform_rgb(rgb_tensor)

        # Depth: [H, W] -> float32 [0,1] -> [1, H, W]
        depth = depth.astype(np.float32)
        max_depth = float(depth.max())
        if max_depth > 0.0:
            depth = depth / max_depth

        depth = np.expand_dims(depth, axis=0)  # [1, H, W]
        depth_tensor = torch.from_numpy(depth)  # [1, H, W]
        if self.transform_depth is not None:
            depth_tensor = self.transform_depth(depth_tensor)

        grasp_tensor = torch.tensor(
            [cx, cy, w, h, angle_deg], dtype=torch.float32
        )  # [5]

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "grasp": grasp_tensor,
        }
