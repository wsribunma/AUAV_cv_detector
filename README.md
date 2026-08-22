# AUAV_cv_detector

Quick prototype: live human detection over a webcam using YOLO, bounding boxes
drawn in real time. Built to prove out the ground-based CV detector path
(`OBS-06` in [AUAV_ground_command_center](../AUAV_ground_command_center)'s
requirements workbook) - an elevated, downward-looking camera detecting
people for search-and-rescue / surveillance, ahead of an eventual onboard
port once an airframe is funded.

This repo is standalone and doesn't talk to AUAV_ground_command_center yet -
no geotagging, no `Observation` schema, just camera in, bounding boxes out.
The intent is to fold it in as a git submodule once the detector side is
worth versioning separately from the platform repo.

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

## Notes

- Person class only (COCO class 0) - other object classes are detected by
  the model but filtered out before drawing.
- `yolov8n.pt` (nano) is the default for prototype speed on CPU. Swap
  `--model yolov8s.pt` or larger for better accuracy once this moves past
  "does it work on my laptop."
- No accuracy claims are made by this script itself - see
  AUAV_ground_command_center's `CLAUDE.md` (decision 3) for the project's
  position on publishing detector vs. geolocation numbers once this is wired
  into the real pipeline.
