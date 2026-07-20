"""Evaluate a trained YOLO11s-P2 checkpoint on the UAV test split."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("runs/detect/uav_yolo11s_p2_20260521_221210/weights/best.pt")
DEFAULT_DATA = Path("datasets/drone_dataset2_augmented/drone_dataset2_augmented.yaml")


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Evaluate YOLO11s-P2 on the UAV dataset test split.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to trained weights.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=960, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. '0', '1', or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for evaluation.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--project", default=None, help="Optional project subdirectory under runs/detect.")
    parser.add_argument("--name", default="uav_yolo11s_p2_test_eval", help="Evaluation run name.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style JSON predictions if supported.")
    parser.add_argument("--save-txt", action="store_true", help="Save predictions as YOLO-format txt files.")
    return parser.parse_args()


def main() -> None:
    """Run test-split evaluation."""
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        save_txt=args.save_txt,
    )

    print("\nTest metrics:")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print(f"mAP50:    {metrics.box.map50:.6f}")
    print(f"mAP75:    {metrics.box.map75:.6f}")


if __name__ == "__main__":
    main()
