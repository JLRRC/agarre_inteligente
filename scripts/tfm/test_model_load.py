#!/usr/bin/env python3
"""Quick check for loading a SimpleGraspCNN checkpoint via GraspModel."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

try:
    import torch  # type: ignore
except Exception as exc:
    print(f"[WARN] torch no disponible, test omitido: {exc}")
    raise SystemExit(0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    root = _repo_root()
    vision_dir = root / "agarre_inteligente"
    ros_src = root / "agarre_ros2_ws" / "src"

    os.environ["VISION_DIR"] = str(vision_dir)
    sys.path.insert(0, str(ros_src))
    sys.path.insert(0, str(vision_dir / "src"))

    from graspnet.models.simple_cnn import SimpleGraspCNN
    from tfm_grasping.model import GraspModel

    model = SimpleGraspCNN(in_channels=3, img_size=224)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "simple_cnn.pth"
        torch.save(model.state_dict(), ckpt_path)
        wrapper = GraspModel(model_path=str(ckpt_path), img_size=224)
        ok = wrapper.load()
        if not ok:
            print(f"[FAIL] no cargo: {wrapper.last_error()}")
            return 1
        if wrapper.info.model_name != "simple_cnn":
            print(f"[FAIL] modelo detectado: {wrapper.info.model_name}")
            return 1
    print("[OK] model load quick test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
