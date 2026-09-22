"""
Stage 03 - dynamic-object masking (ROADMAP.md P4).

Runs YOLOv8 segmentation on each image and writes a COLMAP-style mask that hides people, vehicles and
animals, so moving objects do not become false 3D geometry. COLMAP mask convention: a mask named
"<image>.png" in a mask folder, pixel value 0 = ignore that pixel, 255 = keep. The same masks are
reused by the dense-fusion stage to drop dynamic pixels.

Segmentation (yolov8n-seg) gives pixel-accurate masks; a small dilation adds a safety margin.
"""
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# COCO class ids that move and should never contribute to static structure.
DYNAMIC_COCO = {
    0,   # person
    1, 2, 3, 5, 7,        # bicycle, car, motorcycle, bus, truck
    4, 6, 8,              # airplane, train, boat
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,   # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
}

_MODEL_CACHE: Dict[str, Any] = {}


def _load(model_path: str):
    if model_path not in _MODEL_CACHE:
        from ultralytics import YOLO
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def generate_masks(images_dir: str, out_dir: str, model_path: str = "yolov8n-seg.pt",
                   conf: float = 0.25, dilate_px: int = 9, classes: Optional[set] = None,
                   progress=None) -> Dict[str, Any]:
    """Write <out_dir>/<image>.png masks (0 = dynamic/ignore). Returns per-image dynamic-object counts."""
    os.makedirs(out_dir, exist_ok=True)
    model = _load(model_path)
    keep_out = DYNAMIC_COCO if classes is None else set(classes)
    names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    per_image: List[Dict[str, Any]] = []
    with_dynamic = 0
    for n, name in enumerate(names, 1):
        path = os.path.join(images_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.full((h, w), 255, np.uint8)          # keep everything by default
        res = model.predict(path, conf=conf, verbose=False, device="cpu")[0]
        count = 0
        boxes = getattr(res, "boxes", None)
        segs = getattr(res, "masks", None)
        if boxes is not None and len(boxes):
            cls = boxes.cls.cpu().numpy().astype(int)
            if segs is not None and segs.data is not None:      # pixel-accurate segmentation
                md = segs.data.cpu().numpy()
                for i, c in enumerate(cls):
                    if c in keep_out:
                        m = cv2.resize(md[i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                        mask[m] = 0
                        count += 1
            else:                                                # fallback: box regions
                for (x1, y1, x2, y2), c in zip(boxes.xyxy.cpu().numpy(), cls):
                    if c in keep_out:
                        mask[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = 0
                        count += 1
        if dilate_px and count:
            hole = cv2.dilate((mask == 0).astype(np.uint8) * 255, np.ones((dilate_px, dilate_px), np.uint8))
            mask[hole > 0] = 0
        cv2.imwrite(os.path.join(out_dir, name + ".png"), mask)
        if count:
            with_dynamic += 1
        per_image.append({"name": name, "dynamic_objects": count})
        if progress:
            progress("mask", f"{n}/{len(names)} images")
    return {"images": len(names), "images_with_dynamic": with_dynamic, "per_image": per_image}
