"""
pruning — OpenCV based screenshot pruning engine.

Detects content regions in web screenshots using grayscale thresholding
and contour detection, then crops and stitches them into a dense,
reduced size image suitable for VLM consumption.
"""

from pruning.core import prune

__all__ = ["prune"]
