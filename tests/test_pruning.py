"""
tests/test_pruning.py — Unit tests for the pruning engine.

Covers:
    - Grayscale conversion
    - Binary thresholding
    - Contour detection
    - Box merging
    - Full prune() pipeline
    - Edge cases: near blank page, text dense page
    - Coordinate mapping integrity
"""

import os
import sys
import pytest
import numpy as np
import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.core import (
    to_grayscale,
    threshold_image,
    find_content_boxes,
    merge_nearby_boxes,
    draw_debug_boxes,
    crop_and_stitch,
    prune,
    THRESHOLD_VALUE,
    MIN_CONTOUR_AREA,
)


# ── Test fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def white_image():
    """A near blank white image (should produce minimal content)."""
    return np.ones((800, 1280, 3), dtype=np.uint8) * 255


@pytest.fixture
def text_dense_image():
    """A text dense image with content everywhere."""
    img = np.ones((800, 1280, 3), dtype=np.uint8) * 255
    # Fill with dark rectangles simulating text blocks
    for y in range(0, 800, 30):
        cv2.rectangle(img, (20, y), (1260, y + 20), (30, 30, 30), -1)
    return img


@pytest.fixture
def mixed_image():
    """An image with clear content and background regions."""
    img = np.ones((800, 1280, 3), dtype=np.uint8) * 255
    # Header block
    cv2.rectangle(img, (0, 0), (1280, 100), (50, 50, 50), -1)
    # Content block in the middle
    cv2.rectangle(img, (100, 200), (1100, 500), (80, 80, 80), -1)
    # Footer block
    cv2.rectangle(img, (0, 700), (1280, 800), (50, 50, 50), -1)
    return img


@pytest.fixture
def temp_image_path(tmp_path, mixed_image):
    """Save a mixed image to a temp file and return the path."""
    path = str(tmp_path / "test_input.png")
    cv2.imwrite(path, mixed_image)
    return path


# ── Grayscale tests ─────────────────────────────────────────────────────────

class TestGrayscale:
    def test_bgr_to_gray(self, mixed_image):
        gray = to_grayscale(mixed_image)
        assert len(gray.shape) == 2
        assert gray.shape == (800, 1280)

    def test_already_gray(self):
        gray_input = np.zeros((100, 100), dtype=np.uint8)
        result = to_grayscale(gray_input)
        assert result is gray_input  # should return same object


# ── Threshold tests ─────────────────────────────────────────────────────────

class TestThreshold:
    def test_white_becomes_black(self, white_image):
        gray = to_grayscale(white_image)
        binary = threshold_image(gray)
        # White image should produce almost all black in the mask (no content)
        white_pixels = np.sum(binary == 255)
        assert white_pixels / binary.size < 0.05  # less than 5% content

    def test_dark_becomes_white(self, text_dense_image):
        gray = to_grayscale(text_dense_image)
        binary = threshold_image(gray)
        # Dense text should produce lots of white in the mask (lots of content)
        white_pixels = np.sum(binary == 255)
        assert white_pixels / binary.size > 0.3  # more than 30% content

    def test_output_is_binary(self, mixed_image):
        gray = to_grayscale(mixed_image)
        binary = threshold_image(gray)
        unique_values = np.unique(binary)
        assert set(unique_values).issubset({0, 255})


# ── Contour detection tests ────────────────────────────────────────────────

class TestContourDetection:
    def test_finds_boxes(self, mixed_image):
        gray = to_grayscale(mixed_image)
        binary = threshold_image(gray)
        boxes = find_content_boxes(binary)
        assert len(boxes) > 0

    def test_filters_small_contours(self):
        # Create image with one big and many tiny spots
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (190, 190), 255, -1)  # big
        for i in range(0, 200, 50):
            img[i, i] = 255  # tiny 1px dots
        boxes = find_content_boxes(img, min_area=100)
        assert len(boxes) == 1  # only the big rectangle

    def test_sorted_by_y(self, mixed_image):
        gray = to_grayscale(mixed_image)
        binary = threshold_image(gray)
        boxes = find_content_boxes(binary)
        y_values = [b[1] for b in boxes]
        assert y_values == sorted(y_values)

    def test_near_blank_page(self, white_image):
        gray = to_grayscale(white_image)
        binary = threshold_image(gray)
        boxes = find_content_boxes(binary)
        # White page should produce very few or no boxes
        assert len(boxes) <= 2


# ── Box merging tests ──────────────────────────────────────────────────────

class TestMergeBoxes:
    def test_empty_input(self):
        assert merge_nearby_boxes([]) == []

    def test_no_merge_needed(self):
        boxes = [(0, 0, 100, 50), (0, 200, 100, 50)]  # 150px gap
        merged = merge_nearby_boxes(boxes, merge_distance=10)
        assert len(merged) == 2

    def test_merges_close_boxes(self):
        boxes = [(0, 0, 100, 50), (0, 55, 100, 50)]  # 5px gap
        merged = merge_nearby_boxes(boxes, merge_distance=10)
        assert len(merged) == 1

    def test_merge_preserves_extent(self):
        boxes = [(10, 0, 100, 50), (0, 55, 120, 50)]
        merged = merge_nearby_boxes(boxes, merge_distance=10)
        assert len(merged) == 1
        x, y, w, h = merged[0]
        assert x == 0  # min x
        assert y == 0  # min y
        assert x + w == 120  # max right edge
        assert y + h == 105  # max bottom edge


# ── Crop and stitch tests ──────────────────────────────────────────────────

class TestCropStitch:
    def test_produces_image(self, mixed_image):
        gray = to_grayscale(mixed_image)
        binary = threshold_image(gray)
        boxes = find_content_boxes(binary)
        boxes = merge_nearby_boxes(boxes)
        stitched, coord_map = crop_and_stitch(mixed_image, boxes)
        assert isinstance(stitched, Image.Image)
        assert stitched.width > 0
        assert stitched.height > 0

    def test_coord_map_has_entries(self, mixed_image):
        gray = to_grayscale(mixed_image)
        binary = threshold_image(gray)
        boxes = find_content_boxes(binary)
        boxes = merge_nearby_boxes(boxes)
        _, coord_map = crop_and_stitch(mixed_image, boxes)
        assert len(coord_map["blocks"]) == len(boxes)

    def test_coord_map_y_contiguous(self, mixed_image):
        gray = to_grayscale(mixed_image)
        binary = threshold_image(gray)
        boxes = find_content_boxes(binary)
        boxes = merge_nearby_boxes(boxes)
        _, coord_map = crop_and_stitch(mixed_image, boxes)
        for i, block in enumerate(coord_map["blocks"]):
            if i > 0:
                prev = coord_map["blocks"][i - 1]
                assert block["pruned_y_start"] == prev["pruned_y_end"]

    def test_empty_boxes(self, mixed_image):
        stitched, coord_map = crop_and_stitch(mixed_image, [])
        assert isinstance(stitched, Image.Image)
        assert len(coord_map["blocks"]) == 0


# ── Full pipeline tests ────────────────────────────────────────────────────

class TestPrune:
    def test_full_pipeline(self, temp_image_path):
        result = prune(temp_image_path)
        assert "pruned_image" in result
        assert "coord_map" in result
        assert "metrics" in result
        assert os.path.exists(result["output_path"])

    def test_metrics_present(self, temp_image_path):
        result = prune(temp_image_path)
        m = result["metrics"]
        assert "pixel_reduction_pct" in m
        assert "file_reduction_pct" in m
        assert "num_content_blocks" in m
        assert m["original_pixels"] > 0
        assert m["pruned_pixels"] > 0

    def test_mixed_image_has_reduction(self, temp_image_path):
        result = prune(temp_image_path)
        # Mixed image has clear whitespace gaps, should see some reduction
        assert result["metrics"]["pixel_reduction_pct"] > 0

    def test_custom_output_path(self, temp_image_path, tmp_path):
        out = str(tmp_path / "custom_output.png")
        result = prune(temp_image_path, output_path=out)
        assert result["output_path"] == out
        assert os.path.exists(out)

    def test_debug_saves(self, temp_image_path, tmp_path):
        debug_dir = str(tmp_path / "debug")
        prune(temp_image_path, save_debug=True, debug_dir=debug_dir)
        assert os.path.isdir(debug_dir)
        debug_files = os.listdir(debug_dir)
        assert any("grayscale" in f for f in debug_files)
        assert any("threshold" in f for f in debug_files)
        assert any("boxes" in f for f in debug_files)

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            prune("nonexistent_image.png")

    def test_near_blank_page(self, tmp_path, white_image):
        path = str(tmp_path / "blank.png")
        cv2.imwrite(path, white_image)
        result = prune(path)
        # Should not crash, should return something
        assert result["pruned_image"] is not None

    def test_text_dense_page(self, tmp_path, text_dense_image):
        path = str(tmp_path / "dense.png")
        cv2.imwrite(path, text_dense_image)
        result = prune(path)
        # Dense page should have minimal reduction (content everywhere)
        assert result["metrics"]["pixel_reduction_pct"] < 50
