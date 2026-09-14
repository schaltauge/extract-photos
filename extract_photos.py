#!/usr/bin/env python3
"""
extract_photos.py

Durchsucht einen Ordner nach eingescannten Fotoalbum-Seiten (JPEG/TIFF), erkennt
die darauf aufgeklebten Einzelfotos (auch leicht schief geklebte), schneidet
sie freigestellt und entzerrt aus und speichert sie einzeln im Unterordner
"einzelbilder".

Verwendung:
    python extract_photos.py [ordner]

Wird kein Ordner angegeben, wird das aktuelle Verzeichnis verwendet.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# ---------- Einstellungen ----------
MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.95
PADDING = 0
CONTRAST_MARGIN = 2.0
CONTRAST_FLOOR = 8.0
CLOSE_KERNEL = 9
DEBUG = False


def order_points(pts: np.ndarray) -> np.ndarray:
    """Bringt 4 Punkte in Reihenfolge: oben-links, oben-rechts, unten-rechts, unten-links."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Entzerrt/dreht ein leicht schiefes Viereck auf ein gerades Rechteck."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(max(width_a, width_b)), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(max(height_a, height_b)), 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def shrink_box(box: np.ndarray, pad: float) -> np.ndarray:
    """Zieht die Eckpunkte um pad Pixel zum Zentrum. Bei PADDING=0 unverändert."""
    if pad <= 0:
        return box
    center = box.mean(axis=0)
    shrunk = []
    for p in box:
        v = center - p
        norm = np.linalg.norm(v)
        if norm == 0:
            shrunk.append(p)
        else:
            shrunk.append(p + v / norm * pad)
    return np.array(shrunk, dtype="float32")


def estimate_background_stats(image: np.ndarray, seg_size: int = 60, border: int = 18):
    """Schätzt Seitenfarbe und Rauschlevel robust aus ruhigen Randsegmenten."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    segments = []

    for y0 in (0, max(h - border, 0)):
        for x0 in range(0, w, seg_size):
            x1 = min(x0 + seg_size, w)
            patch = image[y0:y0 + border, x0:x1]
            if patch.size:
                segments.append((patch, float(np.std(gray[y0:y0 + border, x0:x1]))))

    for x0 in (0, max(w - border, 0)):
        for y0 in range(border, max(h - border, border), seg_size):
            y1 = min(y0 + seg_size, h)
            patch = image[y0:y1, x0:x0 + border]
            if patch.size:
                segments.append((patch, float(np.std(gray[y0:y1, x0:x0 + border]))))

    segments.sort(key=lambda s: s[1])
    keep_n = max(1, len(segments) // 2)
    candidates = segments[:keep_n]

    seg_colors = np.array([
        np.median(p.reshape(-1, 3), axis=0)
        for p, _ in candidates
    ])
    consensus = np.median(seg_colors, axis=0)
    color_dist = np.linalg.norm(seg_colors - consensus, axis=1)
    mad = np.median(np.abs(color_dist - np.median(color_dist))) + 1e-6
    keep_mask = color_dist < max(6 * mad, 15.0)

    if not keep_mask.any():
        keep_mask[:] = True

    clean_pixels = np.concatenate([
        p.reshape(-1, 3)
        for (p, _), keep in zip(candidates, keep_mask)
        if keep
    ], axis=0)

    bg_color = np.median(clean_pixels, axis=0)
    diffs = np.linalg.norm(clean_pixels.astype(np.int16) - bg_color, axis=1)
    noise_level = float(np.percentile(diffs, 99))

    return bg_color, noise_level


def compute_photo_mask(diff: np.ndarray, edges: np.ndarray, threshold: float) -> np.ndarray:
    """Baut aus Farbabstand und Kanten eine bereinigte, geschlossene Maske."""
    color_mask = (diff > threshold).astype(np.uint8) * 255
    despeckle = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, despeckle)

    combined = cv2.bitwise_or(edges, color_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL, CLOSE_KERNEL))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    return closed


def boxes_from_mask(mask: np.ndarray, page_area: float):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = page_area * MIN_AREA_RATIO
    max_area = page_area * MAX_AREA_RATIO

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype("float32")
        boxes.append(box)
    return boxes


def find_photo_boxes(image: np.ndarray):
    """Findet die ggf. gedrehten Rechtecke der aufgeklebten Fotos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    smooth_color = cv2.GaussianBlur(image, (9, 9), 0)
    bg_color, noise_level = estimate_background_stats(smooth_color)
    diff = np.linalg.norm(smooth_color.astype(np.int16) - bg_color, axis=2)

    h, w = gray.shape
    page_area = float(h * w)

    threshold = max(noise_level * CONTRAST_MARGIN, CONTRAST_FLOOR)
    mask = compute_photo_mask(diff, edges, threshold)
    return boxes_from_mask(mask, page_area)


def process_image(path: Path, out_dir: Path) -> int:
    image = cv2.imread(str(path))
    if image is None:
        print(f"  Konnte {path.name} nicht lesen, uebersprungen.")
        return 0

    boxes = find_photo_boxes(image)
    if not boxes:
        print(f"  Keine Fotos auf {path.name} gefunden.")
        return 0

    if DEBUG:
        debug_img = image.copy()
        for box in boxes:
            cv2.drawContours(debug_img, [box.astype(int)], 0, (0, 0, 255), 3)
        cv2.imwrite(str(out_dir / f"{path.stem}_debug.jpg"), debug_img)

    boxes.sort(key=lambda b: (round(b[:, 1].min() / 50), b[:, 0].min()))

    count = 0
    for i, box in enumerate(boxes, start=1):
        box = shrink_box(box, PADDING)
        cropped = four_point_transform(image, box)
        out_name = f"{path.stem}_{i:02d}.jpg"
        cv2.imwrite(str(out_dir / out_name), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    print(f"  {path.name}: {count} Foto(s) extrahiert.")
    return count


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if not folder.is_dir():
        print(f"Ordner nicht gefunden: {folder}")
        sys.exit(1)

    out_dir = folder / "einzelbilder"
    out_dir.mkdir(exist_ok=True)

    image_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff")
    )

    if not image_files:
        print("Keine JPEG- oder TIFF-Dateien im Ordner gefunden.")
        return

    print(f"Gefunden: {len(image_files)} Album-Seite(n) in {folder}\n")

    total = 0
    for path in image_files:
        total += process_image(path, out_dir)

    print(f"\nFertig. Insgesamt {total} Einzelbild(er) in '{out_dir}' gespeichert.")


if __name__ == "__main__":
    main()
