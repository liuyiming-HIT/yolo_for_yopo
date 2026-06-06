"""Train YOLO11s for UAV small-object detection."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

MODEL_CFG = Path("ultralytics/cfg/models/11/yolo11s.yaml")
PRETRAINED = Path("yolo11s.pt")
DEFAULT_DATA = Path("datasets/drone_dataset2_augmented/drone_dataset2_augmented.yaml")


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Train YOLO11s on a UAV small-object dataset.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--device", default="1", help="CUDA device, e.g. '1', '0,1', or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--project", default=None, help="Optional project subdirectory under runs/detect.")
    parser.add_argument("--name", help="Run name. Defaults to a timestamped YOLO11s experiment name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing to an existing project/name directory.")
    parser.add_argument("--weights", default=PRETRAINED, help="Pretrained YOLO11s weights path or model name.")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Train from scratch without loading pretrained weights."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume training. Use '--resume' for latest run or '--resume path/to/last.pt' for a checkpoint.",
    )
    return parser.parse_args()


def build_run_name(args) -> str:
    """Create a distinctive run name when the user does not provide one."""
    if args.name:
        return args.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"uav_yolo11s_{stamp}"


def main() -> None:
    """Load YOLO11s weights and configure training."""
    args = parse_args()
    run_name = build_run_name(args)

    if args.resume:
        model = YOLO(args.resume if isinstance(args.resume, str) else args.weights)
    else:
        if args.no_pretrained:
            model = YOLO(MODEL_CFG)
            print("Training from scratch: pretrained weights are disabled.")
        else:
            try:
                model = YOLO(args.weights)
            except ConnectionError as e:
                raise SystemExit(
                    f"\nFailed to load pretrained weights: {args.weights}\n"
                    "The download usually comes from GitHub releases, which is failing on this network.\n"
                    "Put yolo11s.pt in this project directory and rerun, pass a local path with --weights, "
                    "or use --no-pretrained to train from scratch.\n"
                ) from e

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=args.exist_ok,
        close_mosaic=10,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
