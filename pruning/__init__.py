"""
pruning — OpenCV based screenshot pruning engine.

v2: Uses projection-profile whitespace collapsing instead of contour-based
block stitching. Detects background color from image borders (works for
light and dark themes). Preserves spatial layout and all content.

v2R (plan-6): Adds repack mode (tight collage), legacy mode (baseline),
token counting support, and multi-mode prune_image() function.
"""

from pruning.core import (
    prune,
    prune_image,
    CoordinateMap,
    RepackMap,
    Segment,
    MIN_BLOCK_AREA,
)

__all__ = [
    "prune",
    "prune_image",
    "CoordinateMap",
    "RepackMap",
    "Segment",
    "MIN_BLOCK_AREA",
]
