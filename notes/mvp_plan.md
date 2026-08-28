# CV framework MVP plan

Scope for this repo before integration work shifts to AUAV_ground_command_center.
Feeds OBS-06 (ground CV detector) and DEM-07 (AI detection demo scenario) there.

## Features to build here

- Structured detection output. `detect.py` currently only draws boxes to a
  window or file. Needs a function that returns typed detections (bbox,
  confidence, class, timestamp) so a caller can consume results without a
  display.
- Pixel-to-angle conversion. Turn a detection's bounding-box center into an
  azimuth/depression offset from boresight, using field of view and pixel
  resolution. Pure math, testable with synthetic bounding boxes, no camera
  needed.
- Measured camera intrinsics. Replace the placeholder FOV values with a real
  measured horizontal/vertical FOV for the camera actually used on the rig.
- Unit tests for the pixel-to-angle math. No pytest coverage exists yet; CI
  only lints and does an import smoke check.

## Actions outside code

- Mount the camera elevated and downward-looking, even improvised. The
  detector has only been run level, off a laptop lid.
- Log a rough fixed position for the test rig: lat/lon (phone GPS is enough
  for now) and height above ground. Not a survey, just enough to run an
  end-to-end test with approximate numbers.

## Explicitly out of scope here

- Calling into `Projector`/`Observation` on the command-center side, or
  deciding what source class / assurance level a ground rig should carry.
  That decision and the wiring both live in AUAV_ground_command_center.
- Deleting or promoting `app/api/dev_camera.py`.
- Any ingestion or persistence path into the live picture — SYS-02
  (persistence) isn't built yet on the other side.
- The formal calibration procedure (OBS-03) and the surveyed-truth CEP
  harness (OBS-04). Both need the integration in place first.
- Swapping off yolov8n. Nano is fine until detection itself is the
  bottleneck, not before.

## Path to demo

1. Structured output + pixel-to-angle math + measured FOV, tested standalone
   in this repo.
2. Camera up on an elevated, downward-looking mount with a logged rough
   position.
3. Hand off to AUAV_ground_command_center: wire detections through
   `Projector.project()` into real `Observation`s, decide source class /
   assurance, replace or remove `dev_camera.py`.
4. Calibration procedure and surveyed-truth CEP pass once the pipeline is
   live (OBS-03, OBS-04).
5. Rehearse the detection-and-geotagging scenario end to end (DEM-07).

Blocked on OBS-01 and OBS-02 landing on the command-center side first —
`Projector.project()` isn't stable until then. Everything above this repo's
own boundary can proceed in parallel; the handoff step cannot start early.
