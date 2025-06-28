import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import pytesseract
from typing import List, Dict, Any

MODEL_PT = './model/best.pt'
model = YOLO(MODEL_PT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def remove_shadows(img: np.ndarray) -> np.ndarray:
    """Preprocess image to remove shadows."""
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX)
        result_norm_planes.append(norm_img)
    return cv2.merge(result_norm_planes)

def detect_objects(
    img_bgr: np.ndarray,
    conf: float = 0.4,
    imgsz: int = 640
) -> List[Dict[str, Any]]:
    """Run YOLO detection and return bounding box info."""
    img_bgr = remove_shadows(img_bgr)
    results = model.predict(
        source=img_bgr,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=False,
        show=False
    )
    r = results[0]
    detections: List[Dict[str, Any]] = []
    for det in r.boxes:
        cls_id = int(det.cls.cpu().numpy())
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        conf_score = float(det.conf.cpu().numpy())
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf_score
        })
    return detections

def ocr_on_objects(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    psm: int = 6,
    lang: str = 'eng'
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform OCR on each detected object crop and return extracted text.

    Returns a dict mapping class_name to a list of dicts with bbox and text.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    custom_config = f'--oem 3 --psm {psm}'

    for det in detections:
        cls_name = det["class_name"]
        x1, y1, x2, y2 = det["bbox"]
        crop = img_bgr[y1:y2, x1:x2]

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray_crop, lang=lang, config=custom_config).strip()

        entry = {
            "bbox": [x1, y1, x2, y2],
            "text": text
        }
        results.setdefault(cls_name, []).append(entry)

    return results