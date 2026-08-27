#!/usr/bin/env python3
"""Live human detection over a webcam using YOLO.

Quick prototype for the ground-based CV detector path in
AUAV_ground_command_center (OBS-06 in that project's requirements
workbook) - proves live detection with bounding boxes on ordinary laptop
hardware before anything gets wired into the geotagging pipeline. Person
class only (COCO class 0); nothing here talks to that project yet.

Usage:
    python src/detect.py                        # laptop webcam, live window
    python src/detect.py --source 1              # a different camera index
    python src/detect.py --source path/to.jpg    # single image, headless
    python src/detect.py --source path/to.mp4    # video file
    python src/detect.py --save outputs/demo.mp4 # also record annotated output
    python src/detect.py --list-cameras           # probe camera indices

Press q or Esc to quit the live window. The first run downloads the model
weights (~6 MB) from Ultralytics; they are cached locally afterward and are
not checked into git (see .gitignore).

Author: Worawis Sribunma
© Copyright Aero UAViation 2026
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class id
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_cameras(max_index: int = 8) -> None:
    print(f"probing camera indices 0..{max_index - 1}")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            print(f"  {i}: OK  {frame.shape[1]}x{frame.shape[0]}" if ok else f"  {i}: opens but no frame")
        cap.release()


def run(source: str, model_name: str, conf: float, save_path: str | None) -> int:
    model = YOLO(model_name)
    # A digit-only source string means a camera index; anything else is a
    # file path (image or video) - same convention OpenCV itself uses.
    cam_source = int(source) if source.isdigit() else source
    is_single_image = isinstance(cam_source, str) and cam_source.lower().endswith(IMAGE_EXTS)

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {source!r}")

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # H.264-in-MP4 container
        # Fall back to a common default if the source can't report its own
        # size (some virtual/loopback cameras return 0 for these).
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))

    prev_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, classes=[PERSON_CLASS], conf=conf, verbose=False)
            annotated = results[0].plot()

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            cv2.putText(annotated, f"{fps:.1f} fps  {len(results[0].boxes)} person(s)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(annotated)

            if is_single_image:
                os.makedirs("outputs", exist_ok=True)
                out_path = os.path.join("outputs", "annotated.jpg")
                cv2.imwrite(out_path, annotated)
                print(f"wrote {out_path} ({len(results[0].boxes)} person(s) detected)")
                break

            cv2.imshow("AUAV CV prototype - person detection (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # 27 == Esc
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, image path, or video path (default: 0)")
    ap.add_argument("--model", default="yolov8n.pt", help="ultralytics model name or path (default: yolov8n.pt)")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold (default: 0.4)")
    ap.add_argument("--save", default=None, help="also write annotated output to this video path")
    ap.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return 0

    return run(args.source, args.model, args.conf, args.save)


if __name__ == "__main__":
    raise SystemExit(main())
