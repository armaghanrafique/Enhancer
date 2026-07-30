"""Plate perspective de-skew.

Angled CCTV / gate plates are captured as a skewed quadrilateral, which wrecks
OCR. `deskew_plate` finds the plate's four corners and warps it to a flat
rectangle (bird's-eye view). If it can't find a reliable quad it falls back to a
rotation-only correction, and finally to returning the crop unchanged.
"""
import cv2
import numpy as np


def _order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]        # bottom-right (largest x+y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # top-right (smallest y-x)
    rect[3] = pts[np.argmax(diff)]     # bottom-left
    return rect


def _four_point_warp(img, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if max_w < 12 or max_h < 6:
        return None
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1],
                    [0, max_h - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, m, (max_w, max_h), flags=cv2.INTER_CUBIC)


def deskew_plate(plate_rgb):
    """Return a perspective-corrected plate, or the input if none is found."""
    h, w = plate_rgb.shape[:2]
    if h < 12 or w < 24:
        return plate_rgb

    gray = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.25 * w * h   # plate quad should fill much of the crop
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area, best = area, approx.reshape(4, 2).astype("float32")

    if best is not None:
        warped = _four_point_warp(plate_rgb, best)
        if warped is not None:
            return warped

    # fallback: rotation-only deskew from the dominant edge orientation
    ys, xs = np.where(edges > 0)
    if len(xs) > 30:
        rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle += 90
        if 1.5 < abs(angle) < 45:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            return cv2.warpAffine(plate_rgb, m, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    return plate_rgb
