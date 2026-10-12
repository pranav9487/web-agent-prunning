"""
pruning.core — Core pruning functions (v2 rewrite).

Pipeline:
    1. Estimate background color from image borders (works for any theme)
    2. Build content mask via abs-difference from background
    3. Compute row/column projection profiles
    4. Collapse empty runs longer than MIN_GAP down to GAP_KEEP
    5. Reassemble pruned image preserving spatial layout
    6. Build bidirectional coordinate map (pruned ↔ original)

Tuning parameters:
    BG_TOLERANCE (15): Max abs-difference from background color before
        a pixel counts as "content". Range 12-20 works well.
        Lower = more aggressive content detection (may flag compression
        artifacts); Higher = only high-contrast content.

    MIN_GAP (24): Minimum length of an empty run (in pixels) before
        it gets collapsed. Runs shorter than this are kept as-is
        to preserve tight spacing between nearby content.

    GAP_KEEP (8): When an empty run IS collapsed, it's kept at this
        many pixels (not removed entirely). This preserves some
        visual breathing room between content blocks.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, NamedTuple
import os
import csv
import datetime


# ── Tuning parameters ───────────────────────────────────────────────────────
BG_TOLERANCE = 15       # abs-diff from background to count as content
MIN_GAP = 24            # empty run must be this long to collapse
GAP_KEEP = 8            # collapsed gap is kept at this many pixels
# ────────────────────────────────────────────────────────────────────────────


class Segment(NamedTuple):
    """A segment mapping between pruned and original coordinates.

    pruned_start: where this segment begins in the pruned image
    original_start: where this segment begins in the original image
    length: how many pixels this segment spans
    """
    pruned_start: int
    original_start: int
    length: int


class CoordinateMap:
    """Bidirectional coordinate mapping between pruned and original images.

    Stores axis-aligned segment lists for X and Y, plus helper
    functions for point-to-point mapping in both directions.
    """

    def __init__(
        self,
        x_segments: list[Segment],
        y_segments: list[Segment],
        original_width: int,
        original_height: int,
        pruned_width: int,
        pruned_height: int,
    ):
        self.x_segments = x_segments
        self.y_segments = y_segments
        self.original_width = original_width
        self.original_height = original_height
        self.pruned_width = pruned_width
        self.pruned_height = pruned_height

    def _map_axis(self, coord: int, segments: list[Segment], pruned_to_orig: bool) -> int:
        """Map a single coordinate along one axis.

        Args:
            coord: The coordinate to map.
            segments: List of Segment for this axis.
            pruned_to_orig: If True, map pruned→original. Else original→pruned.

        Returns:
            The mapped coordinate.
        """
        for seg in segments:
            if pruned_to_orig:
                src_start = seg.pruned_start
            else:
                src_start = seg.original_start

            if src_start <= coord < src_start + seg.length:
                offset = coord - src_start
                if pruned_to_orig:
                    return seg.original_start + offset
                else:
                    return seg.pruned_start + offset

        # Fallback: clamp to last segment end
        if segments:
            last = segments[-1]
            if pruned_to_orig:
                return last.original_start + last.length - 1
            else:
                return last.pruned_start + last.length - 1
        return coord

    def pruned_to_original(self, x: int, y: int) -> tuple[int, int]:
        """Map a point from pruned image coordinates to original image coordinates.

        Args:
            x: X coordinate in the pruned image.
            y: Y coordinate in the pruned image.

        Returns:
            (orig_x, orig_y) in the original image.
        """
        orig_x = self._map_axis(x, self.x_segments, pruned_to_orig=True)
        orig_y = self._map_axis(y, self.y_segments, pruned_to_orig=True)
        return (orig_x, orig_y)

    def original_to_pruned(self, x: int, y: int) -> tuple[int, int]:
        """Map a point from original image coordinates to pruned image coordinates.

        Args:
            x: X coordinate in the original image.
            y: Y coordinate in the original image.

        Returns:
            (pruned_x, pruned_y) in the pruned image.
        """
        pruned_x = self._map_axis(x, self.x_segments, pruned_to_orig=False)
        pruned_y = self._map_axis(y, self.y_segments, pruned_to_orig=False)
        return (pruned_x, pruned_y)

    def to_dict(self) -> dict:
        """Serialize the coordinate map to a plain dict."""
        return {
            "x_segments": [(s.pruned_start, s.original_start, s.length) for s in self.x_segments],
            "y_segments": [(s.pruned_start, s.original_start, s.length) for s in self.y_segments],
            "original_size": (self.original_width, self.original_height),
            "pruned_size": (self.pruned_width, self.pruned_height),
        }


# ── Background Detection ───────────────────────────────────────────────────

def estimate_background_color(image: np.ndarray) -> np.ndarray:
    """Estimate the background color as the most common color along image borders.

    Collects pixels from all four edges (top row, bottom row, left column,
    right column), converts to grayscale-equivalent, and returns the mode.
    This works for both light and dark themes — no white assumption.

    Args:
        image: BGR or grayscale image.

    Returns:
        Background color as a scalar (grayscale) or 1D array (BGR).
    """
    h, w = image.shape[:2]

    if len(image.shape) == 2:
        # Grayscale
        border_pixels = np.concatenate([
            image[0, :],           # top row
            image[h - 1, :],       # bottom row
            image[1:h - 1, 0],     # left column (excluding corners)
            image[1:h - 1, w - 1], # right column (excluding corners)
        ])
        # Find mode via histogram (0-255 range)
        hist = np.bincount(border_pixels.ravel(), minlength=256)
        bg_color = np.argmax(hist).astype(np.uint8)
        return bg_color
    else:
        # Color image: convert border pixels to grayscale for mode detection,
        # but return the BGR color at the mode grayscale value
        border_pixels = np.concatenate([
            image[0, :],
            image[h - 1, :],
            image[1:h - 1, 0],
            image[1:h - 1, w - 1],
        ], axis=0)  # shape: (N, 3)

        # Convert to grayscale for binning
        gray_vals = np.round(
            0.299 * border_pixels[:, 2] +
            0.587 * border_pixels[:, 1] +
            0.114 * border_pixels[:, 0]
        ).astype(np.uint8)

        hist = np.bincount(gray_vals, minlength=256)
        mode_gray = np.argmax(hist)

        # Return the mean BGR color of pixels at that grayscale value
        mask = gray_vals == mode_gray
        if mask.any():
            bg_color = border_pixels[mask].mean(axis=0).astype(np.uint8)
        else:
            bg_color = border_pixels.mean(axis=0).astype(np.uint8)

        return bg_color


# ── Content Mask ────────────────────────────────────────────────────────────

def build_content_mask(
    image: np.ndarray,
    bg_color: np.ndarray,
    tolerance: int = BG_TOLERANCE,
) -> np.ndarray:
    """Build a binary mask: content pixels = 255, background = 0.

    A pixel is "content" if its absolute difference from the background
    color exceeds the tolerance. Works for both light and dark themes.

    Args:
        image: BGR or grayscale image.
        bg_color: Estimated background color (from estimate_background_color).
        tolerance: Max abs-difference to still be considered background.

    Returns:
        Binary mask (uint8): 255 = content, 0 = background.
    """
    if len(image.shape) == 2:
        # Grayscale
        diff = np.abs(image.astype(np.int16) - int(bg_color))
        mask = (diff > tolerance).astype(np.uint8) * 255
    else:
        # Color: compute per-channel abs-diff, take the max across channels
        diff = np.abs(image.astype(np.int16) - bg_color.astype(np.int16))
        max_diff = np.max(diff, axis=2)
        mask = (max_diff > tolerance).astype(np.uint8) * 255

    return mask


# ── Projection Profiles ────────────────────────────────────────────────────

def compute_projection_profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute row and column projection profiles from a content mask.

    The row profile counts content pixels per row (used for vertical trimming).
    The column profile counts content pixels per column (horizontal trimming).

    Args:
        mask: Binary mask (255 = content, 0 = background).

    Returns:
        Tuple of (row_profile, col_profile), each a 1D int array.
        row_profile[i] = number of content pixels in row i.
        col_profile[j] = number of content pixels in column j.
    """
    binary = (mask > 0).astype(np.int32)
    row_profile = binary.sum(axis=1)   # sum across columns → one value per row
    col_profile = binary.sum(axis=0)   # sum across rows → one value per column
    return row_profile, col_profile


# ── Whitespace Collapse ────────────────────────────────────────────────────

def _collapse_axis(
    profile: np.ndarray,
    min_gap: int = MIN_GAP,
    gap_keep: int = GAP_KEEP,
) -> list[Segment]:
    """Collapse empty runs on one axis, building the segment mapping.

    Scans the profile to find runs of zero (no content pixels).
    - Runs shorter than min_gap: kept entirely (no collapse)
    - Runs >= min_gap: collapsed down to gap_keep pixels

    Content runs are always kept in full.

    Args:
        profile: 1D array, one entry per row or column.
        min_gap: Minimum empty-run length to trigger collapse.
        gap_keep: How many pixels a collapsed gap is reduced to.

    Returns:
        List of Segment(pruned_start, original_start, length) describing
        the mapping from pruned axis to original axis.
    """
    n = len(profile)
    segments: list[Segment] = []
    pruned_pos = 0
    i = 0

    while i < n:
        # Detect whether current position is content or empty
        is_content = profile[i] > 0

        # Find the end of this run
        j = i + 1
        while j < n and (profile[j] > 0) == is_content:
            j += 1
        run_length = j - i

        if is_content:
            # Content run: keep entirely
            segments.append(Segment(pruned_pos, i, run_length))
            pruned_pos += run_length
        else:
            # Empty run
            if run_length >= min_gap:
                # Collapse: keep only gap_keep pixels (centered)
                keep = min(gap_keep, run_length)
                # Map the kept gap to the middle of the original gap
                orig_offset = (run_length - keep) // 2
                segments.append(Segment(pruned_pos, i + orig_offset, keep))
                pruned_pos += keep
            else:
                # Short gap: keep entirely
                segments.append(Segment(pruned_pos, i, run_length))
                pruned_pos += run_length

        i = j

    return segments


def _apply_segments_to_image(
    image: np.ndarray,
    row_segments: list[Segment],
    col_segments: list[Segment],
) -> np.ndarray:
    """Reassemble the pruned image by extracting and placing segments.

    First collapses rows (vertical), then collapses columns (horizontal)
    on the intermediate result.

    Args:
        image: Original BGR or grayscale image.
        row_segments: Segment list for Y axis.
        col_segments: Segment list for X axis.

    Returns:
        Pruned image with whitespace collapsed.
    """
    # Step 1: Collapse rows (vertical)
    if row_segments:
        pruned_height = sum(s.length for s in row_segments)
    else:
        pruned_height = image.shape[0]

    if len(image.shape) == 3:
        row_collapsed = np.zeros((pruned_height, image.shape[1], image.shape[2]), dtype=image.dtype)
    else:
        row_collapsed = np.zeros((pruned_height, image.shape[1]), dtype=image.dtype)

    for seg in row_segments:
        row_collapsed[seg.pruned_start:seg.pruned_start + seg.length] = \
            image[seg.original_start:seg.original_start + seg.length]

    # Step 2: Collapse columns (horizontal)
    if col_segments:
        pruned_width = sum(s.length for s in col_segments)
    else:
        pruned_width = row_collapsed.shape[1]

    if len(row_collapsed.shape) == 3:
        result = np.zeros((pruned_height, pruned_width, row_collapsed.shape[2]), dtype=image.dtype)
    else:
        result = np.zeros((pruned_height, pruned_width), dtype=image.dtype)

    for seg in col_segments:
        result[:, seg.pruned_start:seg.pruned_start + seg.length] = \
            row_collapsed[:, seg.original_start:seg.original_start + seg.length]

    return result


# ── Debug Visualization ─────────────────────────────────────────────────────

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


# ── Main Prune Function ────────────────────────────────────────────────────

def prune(
    image_path: str,
    output_path: Optional[str] = None,
    tolerance: int = BG_TOLERANCE,
    min_gap: int = MIN_GAP,
    gap_keep: int = GAP_KEEP,
    save_debug: bool = False,
    debug_dir: str = "debug_output",
) -> dict:
    """Main pruning function: full pipeline from screenshot to pruned image.

    Takes a screenshot file path, detects content regions using
    background-aware masking, collapses whitespace via projection
    profiles, and returns reduction metrics plus a bidirectional
    coordinate mapping.

    This version:
    - Estimates background color from image borders (not hardcoded white)
    - Works on light AND dark themes
    - Preserves spatial layout (no block collage)
    - Never discards content regions regardless of size
    - Returns a CoordinateMap with pruned_to_original / original_to_pruned

    Args:
        image_path: Path to the input screenshot PNG.
        output_path: Where to save the pruned image. Defaults to
            "<input_name>_pruned.png" in the same directory.
        tolerance: Abs-diff threshold for content vs background.
        min_gap: Minimum empty-run length to collapse.
        gap_keep: Collapsed gap is reduced to this many pixels.
        save_debug: If True, save intermediate images to debug_dir.
        debug_dir: Directory for debug images.

    Returns:
        Dictionary with:
            "pruned_image": PIL Image object (the pruned result)
            "output_path": str, where the pruned image was saved
            "coord_map": CoordinateMap with bidirectional mapping
            "metrics": dict with reduction statistics:
                "original_pixels": int
                "pruned_pixels": int
                "pixel_reduction_pct": float
                "original_file_kb": float
                "pruned_file_kb": float
                "file_reduction_pct": float
                "original_dimensions": str
                "pruned_dimensions": str
                "ink_retention_pct": float
    """
    # Load
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h_orig, w_orig = image.shape[:2]
    original_file_kb = os.path.getsize(image_path) / 1024

    # Step 1: Estimate background color from borders
    bg_color = estimate_background_color(image)

    # Step 2: Build content mask
    content_mask = build_content_mask(image, bg_color, tolerance)

    # Count content pixels in original (for ink retention metric)
    original_ink_count = int(np.sum(content_mask > 0))

    # Step 3: Compute projection profiles
    row_profile, col_profile = compute_projection_profiles(content_mask)

    # Step 4: Collapse whitespace
    row_segments = _collapse_axis(row_profile, min_gap, gap_keep)
    col_segments = _collapse_axis(col_profile, min_gap, gap_keep)

    # Step 5: Reassemble pruned image
    pruned_bgr = _apply_segments_to_image(image, row_segments, col_segments)

    # Also apply segments to the content mask to count retained ink
    pruned_mask = _apply_segments_to_image(content_mask, row_segments, col_segments)
    pruned_ink_count = int(np.sum(pruned_mask > 0))

    # Build coordinate map
    h_pruned, w_pruned = pruned_bgr.shape[:2]
    coord_map = CoordinateMap(
        x_segments=col_segments,
        y_segments=row_segments,
        original_width=w_orig,
        original_height=h_orig,
        pruned_width=w_pruned,
        pruned_height=h_pruned,
    )

    # Convert to PIL
    pruned_rgb = cv2.cvtColor(pruned_bgr, cv2.COLOR_BGR2RGB)
    pruned_image = Image.fromarray(pruned_rgb)

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_pruned{ext}"

    # Save pruned image
    pruned_image.save(output_path)
    pruned_file_kb = os.path.getsize(output_path) / 1024

    # Calculate metrics
    original_pixels = h_orig * w_orig
    pruned_pixels = w_pruned * h_pruned
    pixel_reduction = (1 - pruned_pixels / original_pixels) * 100 if original_pixels > 0 else 0
    file_reduction = (1 - pruned_file_kb / original_file_kb) * 100 if original_file_kb > 0 else 0
    ink_retention = (pruned_ink_count / original_ink_count * 100) if original_ink_count > 0 else 100.0

    metrics = {
        "original_pixels": original_pixels,
        "pruned_pixels": pruned_pixels,
        "pixel_reduction_pct": round(pixel_reduction, 2),
        "original_file_kb": round(original_file_kb, 2),
        "pruned_file_kb": round(pruned_file_kb, 2),
        "file_reduction_pct": round(file_reduction, 2),
        "original_dimensions": f"{w_orig}x{h_orig}",
        "pruned_dimensions": f"{w_pruned}x{h_pruned}",
        "ink_retention_pct": round(ink_retention, 2),
        "original_ink_pixels": original_ink_count,
        "pruned_ink_pixels": pruned_ink_count,
    }

    # Save debug images if requested
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_grayscale.png"), gray)

        # Content mask
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_content_mask.png"), content_mask)

        # Pruned image
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_pruned_debug.png"), pruned_bgr)

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
        "ink_retention_pct", "timestamp",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            row = {k: m.get(k, "") for k in fieldnames}
            if "timestamp" not in m:
                row["timestamp"] = datetime.datetime.now().isoformat()
            writer.writerow(row)
