#!/usr/bin/env python3
"""Live human detection over a webcam using YOLO.

Quick prototype for the ground-based CV detector path in
AUAV_ground_command_center (OBS-06 in that project's requirements
workbook) - proves live detection with bounding boxes on ordinary laptop
hardware before anything gets wired into the geotagging pipeline. Person
class only (COCO class 0).

`detect_people()` is the reusable half: pixels in, structured `Detection`
objects out (box, confidence, class, capture time) - no window, no file I/O.
That's what AUAV_ground_command_center's app/sources/cv_detection.py imports
to turn a detection into a geotagged Observation. `run()` below is this
repo's own CLI on top of it - same function, drawing its own boxes from the
same `Detection` objects rather than a second, divergent path through
`results[0].plot()`.

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
import datetime as dt
import os
import time
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class id
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Detection:
    """One detected box, in the frame's own pixel coordinates - (0, 0) at
    top-left, x right, y down, same convention OpenCV and Ultralytics both
    use. `frame_width`/`frame_height` ride along so a caller can turn a box
    into a normalized offset from frame center without also having to thread
    the frame's shape through separately."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    captured_at: dt.datetime
    frame_width: int
    frame_height: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def detect_people(model: YOLO, frame, conf: float = 0.4) -> list[Detection]:
    """Run `model` on one BGR frame (as `cv2.VideoCapture.read()` returns),
    return every person detection scoring at least `conf`. Pure function of
    its inputs plus wall-clock time for `captured_at` - no window, no camera
    access, no file I/O, so it's callable from a background thread or a unit
    test with a static image just as well as from `run()`'s live loop below.
    """
    results = model.predict(frame, classes=[PERSON_CLASS], conf=conf, verbose=False)
    now = dt.datetime.now(dt.timezone.utc)
    height, width = frame.shape[:2]
    boxes = results[0].boxes
    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
        class_id = int(boxes.cls[i])
        detections.append(
            Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(boxes.conf[i]),
                class_id=class_id,
                class_name=model.names[class_id],
                captured_at=now,
                frame_width=width,
                frame_height=height,
            )
        )
    return detections


def _draw_detections(frame, detections: list[Detection]):
    for det in detections:
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(frame, label, (p1[0], max(p1[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


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

            detections = detect_people(model, frame, conf=conf)
            annotated = _draw_detections(frame, detections)

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            cv2.putText(annotated, f"{fps:.1f} fps  {len(detections)} person(s)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(annotated)

            if is_single_image:
                os.makedirs("outputs", exist_ok=True)
                out_path = os.path.join("outputs", "annotated.jpg")
                cv2.imwrite(out_path, annotated)
                print(f"wrote {out_path} ({len(detections)} person(s) detected)")
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
