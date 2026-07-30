"""Object detectors for targeted enhancement (license plates, faces).

Two backends:
  - "haar"  : OpenCV Haar cascades. Bundled with opencv-python, zero download,
              CPU-fast. Good enough to locate plates / frontal faces.
  - "yolo"  : Ultralytics YOLO. Far more accurate; needs `pip install
              ultralytics` and a weights file (a plate/vehicle model, or a
              general yolov8n.pt). Enable with --detector yolo --detector-weights.

All detectors return a list of (x, y, w, h) boxes in pixel coordinates on the
input RGB frame.
"""
import os

import cv2

_HAAR_DIR = cv2.data.haarcascades
_HAAR_FILES = {
    "plate": "haarcascade_russian_plate_number.xml",
    "face": "haarcascade_frontalface_default.xml",
}


class HaarDetector:
    def __init__(self, kind="plate", scale_factor=1.1, min_neighbors=4,
                 min_size=(30, 15)):
        fname = _HAAR_FILES[kind]
        self.cascade = cv2.CascadeClassifier(os.path.join(_HAAR_DIR, fname))
        if self.cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade: {fname}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]


class YoloDetector:
    def __init__(self, weights, conf=0.25, classes=None):
        from ultralytics import YOLO
        if not weights:
            raise ValueError("YOLO backend needs --detector-weights <model.pt>")
        self.model = YOLO(weights)
        self.conf = conf
        self.classes = classes  # optional list of class ids to keep

    def detect(self, img_rgb):
        # ultralytics accepts RGB numpy directly
        results = self.model.predict(img_rgb, conf=self.conf, verbose=False,
                                     classes=self.classes)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for xyxy in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = xyxy[:4]
                boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes


def build_detector(kind="plate", backend="haar", weights=None, conf=0.25):
    """Factory. Falls back to Haar if YOLO can't be set up."""
    if backend == "yolo":
        try:
            return YoloDetector(weights, conf=conf)
        except Exception as e:  # noqa: BLE001
            print(f"  ! [detect] YOLO unavailable ({e}); falling back to Haar")
    return HaarDetector(kind)
