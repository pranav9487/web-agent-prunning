"""
pruning.core — Core pruning functions.

Pipeline:
    1. Grayscale + binary threshold → content mask
    2. Contour detection → bounding boxes around content regions
    3. Crop + stitch → dense collage with coordinate mapping

Tuning parameters:
    THRESHOLD_VALUE (240): Binary threshold for separating content from
        background. Values above this are treated as background (white/light).
        Lowering catches more subtle content; raising is more aggressive.
        240 works well for typical white-background web pages.

    MIN_CONTOUR_AREA (500): Minimum pixel area for a contour to count as
        a content region. Filters out noise, specks, and single-pixel
        artifacts. 500px is roughly a 22x22 pixel square, small enough
        to keep text blocks but large enough to drop noise.

    PADDING (5): Extra pixels around each bounding box when cropping.
        Prevents clipping text at the edges of detected regions.

    MERGE_DISTANCE (10): If two bounding boxes are within this many
        pixels of each other (vertically), they get merged into one
        region. Prevents fragmentation of content blocks that have
        small gaps (line spacing, thin dividers).
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional
import os
import csv
import datetime

# ── Tuning parameters ───────────────────────────────────────────────────────
THRESHOLD_VALUE = 240       # above this → background
MIN_CONTOUR_AREA = 500      # ignore contours smaller than this (px^2)
PADDING = 5                 # extra pixels around each crop box
MERGE_DISTANCE = 10         # merge boxes within this vertical distance
# ────────────────────────────────────────────────────────────────────────────


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Args:
        image: BGR image (as loaded by cv2.imread).

    Returns:
        Single channel grayscale image.
    """
    if len(image.shape) == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshold_image(
    gray: np.ndarray,
    threshold_value: int = THRESHOLD_VALUE,
) -> np.ndarray:
    """Apply binary thresholding to separate content from background.

    Pixels below threshold_value become white (content), pixels above
    become black (background). The result is then inverted so that
    content regions are white (255) in the mask.

    Args:
        gray: Grayscale image.
        threshold_value: Cutoff value for content vs background.

    Returns:
        Binary mask where white (255) = content, black (0) = background.
    """
    # THRESH_BINARY_INV: pixels below threshold become 255 (content)
    _, binary = cv2.threshold(
        gray, threshold_value, 255, cv2.THRESH_BINARY_INV
    )
    return binary


def find_content_boxes(
    binary_mask: np.ndarray,
    min_area: int = MIN_CONTOUR_AREA,
) -> list[tuple[int, int, int, int]]:
    """Detect content regions via contour detection.

    Finds external contours in the binary mask, filters by minimum area,
    and returns bounding boxes sorted top to bottom.

    Args:
        binary_mask: Binary image (white = content).
        min_area: Minimum contour area in pixels.

    Returns:
        List of (x, y, w, h) bounding boxes, sorted by y position.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    # Sort top to bottom, then left to right
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]],
    merge_distance: int = MERGE_DISTANCE,
) -> list[tuple[int, int, int, int]]:
    """Merge bounding boxes that are close together vertically.

    Prevents fragmentation of content blocks separated by small gaps
    (line spacing, thin dividers, etc).

    Args:
        boxes: List of (x, y, w, h) bounding boxes, sorted by y.
        merge_distance: Maximum vertical gap to trigger a merge.

    Returns:
        Merged list of (x, y, w, h) bounding boxes.
    """
    if not boxes:
        return []

    merged = [list(boxes[0])]  # [x, y, w, h]

    for x, y, w, h in boxes[1:]:
        last = merged[-1]
        last_bottom = last[1] + last[3]

        # Check if this box overlaps or is close to the last merged box
        if y <= last_bottom + merge_distance:
            # Merge: expand the last box to include this one
            new_x = min(last[0], x)
            new_y = min(last[1], y)
            new_right = max(last[0] + last[2], x + w)
            new_bottom = max(last_bottom, y + h)
            merged[-1] = [new_x, new_y, new_right - new_x, new_bottom - new_y]
        else:
            merged.append([x, y, w, h])

    return [tuple(b) for b in merged]


def draw_debug_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on a copy of the image for visual inspection.

    Args:
        image: Original BGR image.
        boxes: List of (x, y, w, h) bounding boxes.
        color: BGR color for the boxes.
        thickness: Line thickness in pixels.

    Returns:
        Copy of the image with boxes drawn on it.
    """
    debug_img = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, thickness)
    return debug_img


def crop_and_stitch(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    padding: int = PADDING,
) -> tuple[Image.Image, dict]:
    """Crop content regions and stitch them into a dense vertical collage.

    Each bounding box is cropped from the original image (with padding),
    and the crops are stacked vertically into a single image. A coordinate
    mapping is built so that positions in the pruned image can be mapped
    back to positions in the original.

    Args:
        image: Original BGR image.
        boxes: List of (x, y, w, h) bounding boxes.
        padding: Extra pixels around each crop.

    Returns:
        Tuple of:
            - PIL Image: the stitched dense collage
            - dict: coordinate mapping with keys:
                "blocks": list of {
                    "original_box": (x, y, w, h) in original image,
                    "pruned_y_start": int, y offset in pruned image,
                    "pruned_y_end": int,
                    "x_offset": int, x shift from crop padding
                }
    """
    h_orig, w_orig = image.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    crops = []
    coord_map = {"blocks": []}
    current_y = 0

    for x, y, w, h in boxes:
        # Apply padding, clamped to image bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_orig, x + w + padding)
        y2 = min(h_orig, y + h + padding)

        crop = pil_image.crop((x1, y1, x2, y2))
        crop_h = y2 - y1
        crop_w = x2 - x1

        crops.append(crop)
        coord_map["blocks"].append({
            "original_box": (x, y, w, h),
            "original_crop": (x1, y1, x2, y2),
            "pruned_y_start": current_y,
            "pruned_y_end": current_y + crop_h,
            "pruned_width": crop_w,
            "x_offset": x1,
        })
        current_y += crop_h

    if not crops:
        # Edge case: no content detected, return original
        return pil_image, {"blocks": []}

    # Stitch vertically: use the max width across all crops
    max_width = max(c.width for c in crops)
    total_height = sum(c.height for c in crops)

    stitched = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    y_pos = 0
    for crop in crops:
        stitched.paste(crop, (0, y_pos))
        y_pos += crop.height

    return stitched, coord_map


def prune(
    image_path: str,
    output_path: Optional[str] = None,
    threshold_value: int = THRESHOLD_VALUE,
    min_contour_area: int = MIN_CONTOUR_AREA,
    padding: int = PADDING,
    merge_distance: int = MERGE_DISTANCE,
    save_debug: bool = False,
    debug_dir: str = "debug_output",
) -> dict:
    """Main pruning function: full pipeline from screenshot to pruned image.

    Takes a screenshot file path, detects content regions, crops and
    stitches them into a dense collage, and returns reduction metrics
    plus a coordinate mapping for click translation.

    Args:
        image_path: Path to the input screenshot PNG.
        output_path: Where to save the pruned image. Defaults to
            "<input_name>_pruned.png" in the same directory.
        threshold_value: Binary threshold for content detection.
        min_contour_area: Minimum contour area to count as content.
        padding: Extra pixels around each crop box.
        merge_distance: Max vertical gap to merge nearby boxes.
        save_debug: If True, save intermediate images (grayscale,
            threshold mask, bounding box overlay) to debug_dir.
        debug_dir: Directory for debug images.

    Returns:
        Dictionary with:
            "pruned_image": PIL Image object (the pruned result)
            "output_path": str, where the pruned image was saved
            "coord_map": dict, coordinate mapping for click translation
            "metrics": dict with reduction statistics:
                "original_pixels": int
                "pruned_pixels": int
                "pixel_reduction_pct": float
                "original_file_kb": float
                "pruned_file_kb": float
                "file_reduction_pct": float
                "num_content_blocks": int
    """
    # Load
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h_orig, w_orig = image.shape[:2]
    original_file_kb = os.path.getsize(image_path) / 1024

    # Step 1: Grayscale
    gray = to_grayscale(image)

    # Step 2: Threshold
    binary = threshold_image(gray, threshold_value)

    # Step 3: Find content boxes
    boxes = find_content_boxes(binary, min_contour_area)
    boxes = merge_nearby_boxes(boxes, merge_distance)

    # Step 4: Crop and stitch
    pruned_image, coord_map = crop_and_stitch(image, boxes, padding)

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_pruned{ext}"

    # Save pruned image
    pruned_image.save(output_path)
    pruned_file_kb = os.path.getsize(output_path) / 1024

    # Calculate metrics
    original_pixels = h_orig * w_orig
    pruned_pixels = pruned_image.width * pruned_image.height
    pixel_reduction = (1 - pruned_pixels / original_pixels) * 100 if original_pixels > 0 else 0
    file_reduction = (1 - pruned_file_kb / original_file_kb) * 100 if original_file_kb > 0 else 0

    metrics = {
        "original_pixels": original_pixels,
        "pruned_pixels": pruned_pixels,
        "pixel_reduction_pct": round(pixel_reduction, 2),
        "original_file_kb": round(original_file_kb, 2),
        "pruned_file_kb": round(pruned_file_kb, 2),
        "file_reduction_pct": round(file_reduction, 2),
        "num_content_blocks": len(boxes),
        "original_dimensions": f"{w_orig}x{h_orig}",
        "pruned_dimensions": f"{pruned_image.width}x{pruned_image.height}",
    }

    # Save debug images if requested
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_grayscale.png"), gray)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_threshold.png"), binary)

        debug_boxes = draw_debug_boxes(image, boxes)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_boxes.png"), debug_boxes)

    return {
        "pruned_image": pruned_image,
        "output_path": output_path,
        "coord_map": coord_map,
        "metrics": metrics,
    }


def log_metrics_csv(
    metrics_list: list[dict],
    csv_path: str = "pruning_metrics.csv",
) -> None:
    """Write a list of pruning metrics to a CSV file.

    Args:
        metrics_list: List of metric dicts (from prune()["metrics"]).
        csv_path: Output CSV file path.
    """
    if not metrics_list:
        return

    fieldnames = [
        "image_name", "original_dimensions", "pruned_dimensions",
        "original_pixels", "pruned_pixels", "pixel_reduction_pct",
        "original_file_kb", "pruned_file_kb", "file_reduction_pct",
        "num_content_blocks", "timestamp",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            row = {k: m.get(k, "") for k in fieldnames}
            if "timestamp" not in m:
                row["timestamp"] = datetime.datetime.now().isoformat()
            writer.writerow(row)
