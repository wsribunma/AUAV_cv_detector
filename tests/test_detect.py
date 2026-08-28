"""Tests for detect_people() and Detection - the structured half of this
repo's detector, as opposed to run()'s CLI/display loop.

Uses the person photo Ultralytics itself bundles as a package asset
(zidane.jpg) rather than a fixture checked into this repo, so there's no
image binary to maintain here - and it means these tests exercise a real
YOLO model on a real photo, not a mock.

Author: Worawis Sribunma
© Copyright Aero UAViation 2026
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import cv2
import pytest
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detect import Detection, detect_people

ZIDANE_JPG = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg")


@pytest.fixture(scope="module")
def model() -> YOLO:
    return YOLO("yolov8n.pt")


def test_detect_people_finds_people_in_a_real_photo(model: YOLO) -> None:
    """zidane.jpg (Ultralytics' own bundled sample) has two people in it.

    Expected output: at least one Detection, all of them class "person"
    with confidence in (0, 1], and captured_at is a timezone-aware UTC
    timestamp from just now.
    """
    frame = cv2.imread(ZIDANE_JPG)
    before = dt.datetime.now(dt.timezone.utc)

    detections = detect_people(model, frame, conf=0.4)

    assert len(detections) >= 1
    for det in detections:
        assert det.class_name == "person"
        assert det.class_id == 0
        assert 0.0 < det.confidence <= 1.0
        assert det.frame_width == frame.shape[1]
        assert det.frame_height == frame.shape[0]
        assert det.captured_at.tzinfo is not None
        assert det.captured_at >= before


def test_higher_confidence_threshold_yields_no_more_detections(model: YOLO) -> None:
    """Raising --conf can only narrow the result, never widen it - a
    detection that clears a stricter bar always cleared a looser one too.

    Expected output: the count at conf=0.8 is less than or equal to the
    count at conf=0.4 on the same frame.
    """
    frame = cv2.imread(ZIDANE_JPG)

    loose = detect_people(model, frame, conf=0.4)
    strict = detect_people(model, frame, conf=0.8)

    assert len(strict) <= len(loose)


def test_detection_center_is_the_bounding_box_midpoint() -> None:
    """Pure math on the dataclass - no model or image needed.

    Expected output: center is exactly the average of the two corners.
    """
    det = Detection(
        x1=100.0, y1=200.0, x2=300.0, y2=400.0,
        confidence=0.9, class_id=0, class_name="person",
        captured_at=dt.datetime(2026, 8, 28, tzinfo=dt.timezone.utc),
        frame_width=640, frame_height=480,
    )

    assert det.center == (200.0, 300.0)
