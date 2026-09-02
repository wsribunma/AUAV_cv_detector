# AUAV_cv_detector

Live human detection over a webcam using YOLO. Built to prove out the
ground-based CV detector path (`OBS-06` in
[AUAV_ground_command_center](../AUAV_ground_command_center)'s requirements
workbook) - an elevated, downward-looking camera detecting people for
search-and-rescue / surveillance, ahead of an eventual onboard port once an
airframe is funded.

This repo has no geotagging or `Observation` schema of its own - it stays
camera in, structured detections out - but it's no longer a dead end:
`detect_people()` in `src/detect.py` is a reusable function (box, confidence,
class, capture time), not just the CLI's own display loop, and it's what
AUAV_ground_command_center's `app/sources/cv_detection.py` imports (via the
`external/AUAV_cv_detector` submodule this repo already is there) to turn a
live detection into a geotagged `Observation` against whatever aircraft pose
AUAV SITL is currently reporting. `run()` below is still this repo's own CLI
on top of the same function, for previewing detection on its own.

## Setup

`ultralytics` and `opencv-python` only - install them however you like.

```bash
pip install -r requirements.txt
```

A venv is optional, not required - use one if you want this prototype's
dependency versions isolated from the rest of your system, skip it if you'd
rather install straight into your normal environment:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

The first run downloads YOLOv8n weights (~6 MB) from Ultralytics and caches
them locally; they are not committed (see `.gitignore`).

## Usage

```bash
python src/detect.py                        # laptop webcam, live window, q to quit
python src/detect.py --list-cameras          # find the right camera index first
python src/detect.py --source 1              # a different camera index
python src/detect.py --source path/to.jpg    # single image, headless - writes outputs/annotated.jpg
python src/detect.py --source path/to.mp4    # video file
python src/detect.py --save outputs/demo.mp4 # also record the annotated live feed
python src/detect.py --conf 0.6              # raise the confidence threshold
```

`outputs/` is gitignored on purpose - annotated frames/video may contain
real people and are regenerable, not source.

## As a library

```python
from ultralytics import YOLO
from detect import detect_people   # src/ on your path

model = YOLO("yolov8n.pt")
detections = detect_people(model, frame, conf=0.4)   # frame: a cv2.VideoCapture().read() BGR array
for d in detections:
    print(d.class_name, d.confidence, d.center)       # center in the frame's own pixel coords
```

No window, no file I/O - safe to call from a background thread or a test.
`tests/test_detect.py` runs it against the person photo Ultralytics itself
bundles as a package asset, so there's no image fixture to maintain here:

```bash
pip install -r requirements.txt   # now includes pytest
pytest -q
```

## Notes

- Person class only (COCO class 0) - other object classes are detected by
  the model but filtered out before drawing.
- `yolov8n.pt` (nano) is the default for prototype speed on CPU. Swap
  `--model yolov8s.pt` or larger for better accuracy once this moves past
  "does it work on my laptop."
- No accuracy claims are made by this script itself. Once this is wired
  into the real pipeline, detection performance (precision/recall) may be
  measured and published alongside geolocation accuracy - geolocation
  remains the primary, harder-to-fake claim to lead with, but detector
  numbers don't need to be scrubbed out of the pitch.
