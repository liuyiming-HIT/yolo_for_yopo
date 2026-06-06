"""Lightweight structure check for the YOLO11s-P2 model."""

from pathlib import Path

import torch

from ultralytics import YOLO

MODEL_CFG = Path("ultralytics/cfg/models/11/yolo11s-p2.yaml")


def main() -> None:
    """Load YOLO11s-P2, print model info, and run one synthetic forward pass."""
    model = YOLO(MODEL_CFG)
    model.info()

    x = torch.zeros(1, 3, 960, 960)
    model.model.eval()
    with torch.no_grad():
        y = model.model(x)

    if isinstance(y, (list, tuple)):
        print(f"Forward OK. Output type: {type(y).__name__}, length: {len(y)}")
    else:
        print(f"Forward OK. Output type: {type(y).__name__}")


if __name__ == "__main__":
    main()
