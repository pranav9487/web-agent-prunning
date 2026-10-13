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
from typing import Optional, NamedTuple, Callable
import os
import csv
import datetime


# ── Tuning parameters ───────────────────────────────────────────────────────
BG_TOLERANCE = 15       # abs-diff from background to count as content
MIN_GAP = 24            # empty run must be this long to collapse
GAP_KEEP = 8            # collapsed gap is kept at this many pixels
MIN_BLOCK_AREA = 500    # minimum pixel area to keep a block in repack mode
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


# ── Repack Helpers ──────────────────────────────────────────────────────────

def _find_content_blocks(
    mask: np.ndarray,
    min_gap: int = MIN_GAP,
) -> list[tuple[int, int, int, int]]:
    """Find rectangular content blocks via XY-cut on the content mask.

    Uses the row and column projection profiles to split the mask into
    independent rectangular regions, each representing a visual block.

    Args:
        mask: Binary content mask (255 = content, 0 = background).
        min_gap: Minimum gap size to cut on.

    Returns:
        List of (x, y, w, h) bounding boxes in original image coordinates.
    """
    row_profile, col_profile = compute_projection_profiles(mask)

    # Find row bands with content
    def _find_content_bands(profile: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
        """Return list of (start, end) for content runs."""
        bands = []
        in_content = False
        gap_count = 0
        start = 0
        n = len(profile)
        for i in range(n):
            if profile[i] > 0:
                if not in_content:
                    in_content = True
                    start = max(0, i - gap_count)
                gap_count = 0
            else:
                if in_content:
                    gap_count += 1
                    if gap_count >= min_gap:
                        bands.append((start, i - gap_count + 1))
                        in_content = False
                        gap_count = 0
        if in_content:
            bands.append((start, n))
        return bands

    row_bands = _find_content_bands(row_profile, min_gap)
    if not row_bands:
        return []

    h, w = mask.shape[:2]
    blocks = []
    for (r_start, r_end) in row_bands:
        row_mask = mask[r_start:r_end, :]
        _, col_prof = compute_projection_profiles(row_mask)
        col_bands = _find_content_bands(col_prof, min_gap)
        if not col_bands:
            col_bands = [(0, w)]
        for (c_start, c_end) in col_bands:
            bw = c_end - c_start
            bh = r_end - r_start
            if bw * bh >= MIN_BLOCK_AREA:
                blocks.append((c_start, r_start, bw, bh))

    return blocks


class RepackMap:
    """Per-block coordinate mapping for repack mode.

    Stores a list of (original_box, pruned_box) entries where each entry
    maps a source rectangle in the original image to its destination
    rectangle in the repacked collage.
    """

    def __init__(self, entries: list[dict]):
        # Each entry: {"orig_x", "orig_y", "orig_w", "orig_h",
        #              "pruned_x", "pruned_y", "pruned_w", "pruned_h"}
        self.entries = entries

    def pruned_to_original(self, x: int, y: int) -> tuple[int, int]:
        """Map a point from the repacked image back to original coordinates."""
        for e in self.entries:
            px, py, pw, ph = e["pruned_x"], e["pruned_y"], e["pruned_w"], e["pruned_h"]
            if px <= x < px + pw and py <= y < py + ph:
                ox = e["orig_x"] + (x - px)
                oy = e["orig_y"] + (y - py)
                return (ox, oy)
        return (x, y)

    def original_to_pruned(self, x: int, y: int) -> tuple[int, int]:
        """Map a point from original coordinates to the repacked image."""
        for e in self.entries:
            ox, oy, ow, oh = e["orig_x"], e["orig_y"], e["orig_w"], e["orig_h"]
            if ox <= x < ox + ow and oy <= y < oy + oh:
                px = e["pruned_x"] + (x - ox)
                py = e["pruned_y"] + (y - oy)
                return (px, py)
        return (x, y)

    def to_dict(self) -> dict:
        """Serialize the repack map to a plain dict."""
        return {"entries": self.entries}


def _prune_repack(
    image: np.ndarray,
    content_mask: np.ndarray,
    min_gap: int = MIN_GAP,
    gap_keep: int = GAP_KEEP,
    margin: int = 4,
) -> tuple[np.ndarray, RepackMap]:
    """Repack content blocks into a vertical collage (Mode B).

    Extracts each detected content block and stacks them vertically
    with a small margin, producing the most compact possible image.

    Args:
        image: Original BGR image.
        content_mask: Binary content mask.
        min_gap: Minimum gap to cut blocks on.
        gap_keep: Not used in repack (kept for API symmetry).
        margin: Pixel gap between blocks in the collage.

    Returns:
        Tuple of (repacked_image, RepackMap).
    """
    blocks = _find_content_blocks(content_mask, min_gap)
    h_orig, w_orig = image.shape[:2]

    if not blocks:
        # Fall back to full image
        dummy_entry = {
            "orig_x": 0, "orig_y": 0, "orig_w": w_orig, "orig_h": h_orig,
            "pruned_x": 0, "pruned_y": 0, "pruned_w": w_orig, "pruned_h": h_orig,
        }
        return image.copy(), RepackMap([dummy_entry])

    # Sort blocks top-to-bottom, then left-to-right
    blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))

    # Determine collage width = max block width
    collage_w = max(bw for _, _, bw, _ in blocks_sorted)
    total_h = sum(bh for _, _, _, bh in blocks_sorted) + margin * (len(blocks_sorted) - 1)

    if len(image.shape) == 3:
        collage = np.zeros((total_h, collage_w, image.shape[2]), dtype=image.dtype)
    else:
        collage = np.zeros((total_h, collage_w), dtype=image.dtype)

    entries = []
    cur_y = 0
    for (bx, by, bw, bh) in blocks_sorted:
        src = image[by:by + bh, bx:bx + bw]
        collage[cur_y:cur_y + bh, 0:bw] = src
        entries.append({
            "orig_x": bx, "orig_y": by, "orig_w": bw, "orig_h": bh,
            "pruned_x": 0, "pruned_y": cur_y, "pruned_w": bw, "pruned_h": bh,
        })
        cur_y += bh + margin

    return collage, RepackMap(entries)


# ── Legacy Prune (old global row/column crop method) ──────────────────────

def _prune_legacy(
    image: np.ndarray,
    tolerance: int = 240,
) -> tuple[np.ndarray, None]:
    """Legacy pruning method: global grayscale threshold + row/column crop.

    This is the original method kept as a baseline. It crops blank rows and
    columns using a fixed grayscale threshold (assumes light background).
    Returns (pruned_image, None) — no coordinate map is produced.

    Args:
        image: BGR image.
        tolerance: Grayscale value above which a pixel is considered blank.

    Returns:
        Tuple of (pruned_bgr_array, None).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Rows and columns with any content pixel (< tolerance)
    content_rows = np.any(gray < tolerance, axis=1)
    content_cols = np.any(gray < tolerance, axis=0)

    rows = np.where(content_rows)[0]
    cols = np.where(content_cols)[0]

    if len(rows) == 0 or len(cols) == 0:
        return image.copy(), None

    r_start, r_end = rows[0], rows[-1] + 1
    c_start, c_end = cols[0], cols[-1] + 1
    return image[r_start:r_end, c_start:c_end].copy(), None


def prune(
    image_path: str,
    output_path=None,
    tolerance=BG_TOLERANCE,
    min_gap=MIN_GAP,
    gap_keep=GAP_KEEP,
    save_debug=False,
    debug_dir="debug_output",
    mode="compress",
    count_tokens_fn=None,
):
    """Backward-compatible wrapper. Delegates to prune_image().

    Args:
        image_path: Path to the input screenshot PNG.
        output_path: Where to save the pruned image.
        tolerance: Abs-diff threshold for content vs background.
        min_gap: Minimum empty-run length to collapse.
        gap_keep: Collapsed gap kept at this many pixels.
        save_debug: If True, save intermediate images.
        debug_dir: Directory for debug images.
        mode: One of "compress" (default), "repack", "legacy".
        count_tokens_fn: Optional callable(pil_image) -> int for token counting.

    Returns:
        Same dict as prune_image().
    """
    return prune_image(
        image_path=image_path,
        output_path=output_path,
        mode=mode,
        tolerance=tolerance,
        min_gap=min_gap,
        gap_keep=gap_keep,
        save_debug=save_debug,
        debug_dir=debug_dir,
        count_tokens_fn=count_tokens_fn,
    )


def prune_image(
    image_path: str,
    output_path=None,
    mode="compress",
    tolerance=BG_TOLERANCE,
    min_gap=MIN_GAP,
    gap_keep=GAP_KEEP,
    save_debug=False,
    debug_dir="debug_output",
    count_tokens_fn=None,
) -> dict:
    """Multi-mode pruning: compress, repack, or legacy.

    Modes:
      - "compress": gap-collapse preserving spatial layout. Returns CoordinateMap.
      - "repack": tight vertical collage of content blocks. Returns RepackMap.
      - "legacy": old global row/column crop (fixed grayscale). No coord map.

    Args:
        image_path: Path to the input screenshot PNG.
        output_path: Where to save the pruned image.
        mode: One of "compress", "repack", "legacy".
        tolerance: Abs-diff threshold for content vs background.
        min_gap: Minimum empty-run length to collapse.
        gap_keep: Collapsed gap size in compress mode.
        save_debug: If True, save intermediate images.
        debug_dir: Directory for debug images.
        count_tokens_fn: Optional callable(pil_image) -> int. When provided,
            raw and pruned images are counted and token metrics are added.

    Returns:
        {
            "pruned_image": PIL Image,
            "output_path": str,
            "coord_map": CoordinateMap | RepackMap | None,
            "mode": str,
            "metrics": dict,
        }
    """
    valid_modes = ("compress", "repack", "legacy")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h_orig, w_orig = image.shape[:2]
    original_file_kb = os.path.getsize(image_path) / 1024

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        suffix = "_pruned" if mode == "compress" else f"_pruned_{mode}"
        output_path = f"{base}{suffix}{ext}"

    coord_map = None
    content_mask = None

    if mode == "legacy":
        pruned_bgr, coord_map = _prune_legacy(image)
    else:
        bg_color = estimate_background_color(image)
        content_mask = build_content_mask(image, bg_color, tolerance)
        if mode == "compress":
            row_profile, col_profile = compute_projection_profiles(content_mask)
            row_segments = _collapse_axis(row_profile, min_gap, gap_keep)
            col_segments = _collapse_axis(col_profile, min_gap, gap_keep)
            pruned_bgr = _apply_segments_to_image(image, row_segments, col_segments)
            hp, wp = pruned_bgr.shape[:2]
            coord_map = CoordinateMap(
                x_segments=col_segments,
                y_segments=row_segments,
                original_width=w_orig,
                original_height=h_orig,
                pruned_width=wp,
                pruned_height=hp,
            )
        else:
            pruned_bgr, coord_map = _prune_repack(image, content_mask, min_gap, gap_keep)

    # Ink retention metric
    original_ink_count = 0
    pruned_ink_count = 0
    if mode != "legacy" and content_mask is not None:
        original_ink_count = int(np.sum(content_mask > 0))
        if mode == "compress" and isinstance(coord_map, CoordinateMap):
            pm = _apply_segments_to_image(content_mask, coord_map.y_segments, coord_map.x_segments)
            pruned_ink_count = int(np.sum(pm > 0))
        else:
            pruned_ink_count = original_ink_count

    pruned_rgb = cv2.cvtColor(pruned_bgr, cv2.COLOR_BGR2RGB)
    pruned_image = Image.fromarray(pruned_rgb)
    pruned_image.save(output_path)
    pruned_file_kb = os.path.getsize(output_path) / 1024
    h_pruned, w_pruned = pruned_bgr.shape[:2]

    original_pixels = h_orig * w_orig
    pruned_pixels = w_pruned * h_pruned
    pixel_reduction = (1 - pruned_pixels / original_pixels) * 100 if original_pixels > 0 else 0
    file_reduction = (1 - pruned_file_kb / original_file_kb) * 100 if original_file_kb > 0 else 0
    ink_retention = pruned_ink_count / original_ink_count * 100 if original_ink_count > 0 else 100.0

    num_content_blocks = 0
    if mode == "repack" and hasattr(coord_map, "blocks"):
        num_content_blocks = len(coord_map.blocks)
    elif mode == "compress" and hasattr(coord_map, "y_segments"):
        num_content_blocks = len(coord_map.y_segments)
    elif mode == "legacy":
        num_content_blocks = 1

    metrics: dict = {
        "mode": mode,
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
        "num_content_blocks": num_content_blocks,
    }

    if count_tokens_fn is not None:
        try:
            raw_tokens = count_tokens_fn(Image.open(image_path).convert("RGB"))
            pruned_tokens = count_tokens_fn(pruned_image)
            token_reduction = (1 - pruned_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0.0
            metrics["raw_tokens"] = raw_tokens
            metrics["pruned_tokens"] = pruned_tokens
            metrics["token_reduction_pct"] = round(token_reduction, 2)
        except Exception as e:
            metrics["token_count_error"] = str(e)

    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if mode != "legacy" and content_mask is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_{mode}_grayscale.png"), gray)
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_{mode}_content_mask.png"), content_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_{mode}_pruned_debug.png"), pruned_bgr)

    return {
        "pruned_image": pruned_image,
        "output_path": output_path,
        "coord_map": coord_map,
        "mode": mode,
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
