#!/usr/bin/env python3
"""Pre-compute YOLO detections for a video, saving to CSV.

Usage:
    uv run python3 scripts/extract_detections.py videos/helicopter3.mp4 --output videos/helicopter3.csv
"""

import argparse
import csv

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Extract YOLO detections to CSV")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--model", default="videos/yolov8n.pt",
                        help="YOLO model path")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: video path + .csv)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--target-classes", nargs="+", type=int,
                        default=[4, 14],
                        help="COCO class IDs to track (4=airplane, 14=bird)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.video.rsplit(".", 1)[0] + ".csv"

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: could not open video {args.video}")
        return 1

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "cx", "cy"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, conf=args.conf)
            best_center = None
            best_conf = 0

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls in args.target_classes and conf > best_conf:
                        best_conf = conf
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        best_center = (cx, cy)

            if best_center:
                writer.writerow([frame_idx, best_center[0], best_center[1]])

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")

    cap.release()
    print(f"Wrote {args.output} ({frame_idx} frames)")


if __name__ == "__main__":
    main()
