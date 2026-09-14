# Photo Album Extractor

Automatically detect, straighten, crop, and export individual
photographs from scanned photo album pages.

The script is designed for scanned album pages containing multiple
physical photos, including slightly rotated photos. It combines edge
detection with adaptive background and color analysis to distinguish
photographs from the album page.

## Features

-   Detects multiple photos on a scanned album page
-   Handles slightly rotated photographs
-   Straightens and perspective-corrects detected photos
-   Supports JPEG and TIFF input
-   Exports individual photos as high-quality JPEG files
-   Uses adaptive background detection for light, aged, or slightly
    uneven album pages
-   Can create debug images showing detected photo boundaries
-   Batch-processes all supported images in a folder

## Requirements

-   Python 3
-   NumPy
-   OpenCV

A virtual environment is recommended.

## Installation

Create and activate a virtual environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

``` bash
pip install numpy opencv-python-headless
```

## Usage

Process scans in the current directory:

``` bash
python extract_photos.py .
```

Or specify another directory:

``` bash
python extract_photos.py /path/to/scans
```

If no directory is specified, the current directory is used.

Supported input formats are `.jpg`, `.jpeg`, `.tif`, and `.tiff`.

Extracted photographs are written to:

``` text
einzelbilder/
```

Output images are JPEG files saved at quality 95.

## Configuration

The main settings are near the top of `extract_photos.py`:

``` python
MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.95
PADDING = 0
CONTRAST_MARGIN = 2.0
CONTRAST_FLOOR = 8.0
CLOSE_KERNEL = 9
DEBUG = True
```

### `MIN_AREA_RATIO`

Minimum detected photo area relative to the complete scanned page. Lower
it if small photographs are missed; increase it to ignore small false
detections.

### `MAX_AREA_RATIO`

Maximum detected photo area relative to the scanned page. This helps
prevent almost the entire album page from being interpreted as one
photograph.

### `PADDING`

Pixels removed from the detected boundary before extraction. Use `0` to
preserve the complete detected photograph. Increase it slightly if
shadows, glue marks, or album-page borders remain.

### `CONTRAST_MARGIN`

Controls how strongly an area must differ from the estimated album-page
background. Lower values make detection more sensitive to faded or
low-contrast photographs; higher values can reduce false detections on
textured or stained pages.

### `CONTRAST_FLOOR`

Minimum absolute threshold used for background separation. Lowering it
may help with very faded photographs but can increase false positives.

### `CLOSE_KERNEL`

Controls how aggressively gaps in detected regions are closed. Reduce it
if photographs placed close together are incorrectly merged.

### `DEBUG`

Set `DEBUG = True` to create additional images showing detected photo
boundaries in red. Set it to `False` when debug images are no longer
needed.

## How it works

For each scanned album page, the script:

1.  Estimates the album-page background color and its natural variation
    from relatively quiet areas around the page border.
2.  Detects visible edges using OpenCV's Canny edge detector.
3.  Measures color differences between the scan and the estimated page
    background.
4.  Combines edge and color information into a photo mask.
5.  Finds sufficiently large connected regions and calculates rotated
    bounding rectangles.
6.  Applies a perspective transform to straighten each detected
    photograph.
7.  Saves each photograph as a separate JPEG file.

This approach can work better than edge detection alone when a
photograph contains bright areas, such as sky, with little contrast
against a light album page.

## Notes

Automatic detection cannot be perfect for every album. Results depend on
the album background, shadows, overlapping photographs, decorative
elements, faded prints, and scan quality.

For a new album, enable `DEBUG`, test a few representative pages, and
adjust the configuration before processing a large collection.

The original scan files are never modified.

## License

No license is included by default. Add a license file appropriate for
how you want others to use the project.
