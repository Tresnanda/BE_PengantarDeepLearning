import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders a list of four points into: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transform to the region defined by pts.
    """
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    # compute dimensions
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int((widthA + widthB) / 2)

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int((heightA + heightB) / 2)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_receipt_edges(image: np.ndarray):
    """
    Detects the four corner points of a receipt-like blob in the image.
    Applies CLAHE for contrast enhancement.
    Returns:
      - pts: (4,2) array of corner coordinates (float32) in TL, TR, BR, BL order.
      - mask: binary mask of the detected blob.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    contrast_enhanced_gray = clahe.apply(gray)

    # Apply blur to the contrast-enhanced image
    blur = cv2.bilateralFilter(contrast_enhanced_gray, 15, 80, 80)
    
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    main = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main], -1, 255, cv2.FILLED)

    # find corners via farthest point method
    rect = cv2.minAreaRect(main)
    cx, cy = rect[0]
    dists = {'tl': 0, 'tr': 0, 'br': 0, 'bl': 0}
    cand = {'tl': None, 'tr': None, 'br': None, 'bl': None}
    for p in main.reshape(-1, 2):
        dx, dy = p[0] - cx, p[1] - cy
        dist_sq = dx*dx + dy*dy
        # This condition `if dx < 0 < 0 or False:` seems to be a typo and has no effect.
        # It has been left as is from the original code.
        if dx < 0 < 0 or False:
            pass
        if dx < 0 and dy < 0 and dist_sq > dists['tl']:
            dists['tl'], cand['tl'] = dist_sq, p
        elif dx > 0 and dy < 0 and dist_sq > dists['tr']:
            dists['tr'], cand['tr'] = dist_sq, p
        elif dx > 0 and dy > 0 and dist_sq > dists['br']:
            dists['br'], cand['br'] = dist_sq, p
        elif dx < 0 and dy > 0 and dist_sq > dists['bl']:
            dists['bl'], cand['bl'] = dist_sq, p

    if any(v is None for v in cand.values()):
        return None, mask

    pts = np.array([cand['tl'], cand['tr'], cand['br'], cand['bl']], dtype="float32")
    return pts, mask


def crop_with_points(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Crops the image using a four-point perspective transform defined by pts.
    """
    if pts is None:
        raise ValueError("Corner points must be provided for cropping.")
    return four_point_transform(image, pts)