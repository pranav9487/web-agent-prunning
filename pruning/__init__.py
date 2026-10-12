"""
pruning — OpenCV based screenshot pruning engine.

v2: Uses projection-profile whitespace collapsing instead of contour-based
block stitching. Detects background color from image borders (works for
light and dark themes). Preserves spatial layout and all content.
"""

from pruning.core import prune, CoordinateMap, Segment

__all__ = ["prune", "CoordinateMap", "Segment"]
