"""
tests/test_prune.py — Comprehensive tests for the v2 pruning engine.

Covers:
    - Background detection (light and dark themes)
    - Content mask (abs-diff from estimated background)
    - Projection profiles
    - Whitespace collapse
    - Full prune() pipeline
    - Ink retention >= 98%
    - Coordinate round-trip error <= 1px
    - Optional OCR recall >= 95% (with pytesseract)
    - Sample image results table
"""

import os
import sys
import pytest
import numpy as np
import cv2
from PIL import Image
import random
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.core import (
    estimate_background_color,
    build_content_mask,
    compute_projection_profiles,
    _collapse_axis,
    _apply_segments_to_image,
    prune,
    CoordinateMap,
    Segment,
    BG_TOLERANCE,
    MIN_GAP,
    GAP_KEEP,
)


# ── Test fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def white_image():
    """A near blank white image (should produce minimal content)."""
    return np.ones((800, 1280, 3), dtype=np.uint8) * 255


@pytest.fixture
def dark_image():
    """A dark-mode image (dark bg, light content)."""
    img = np.ones((800, 1280, 3), dtype=np.uint8) * 25
    # Light content blocks on dark bg
    cv2.rectangle(img, (50, 50), (1230, 100), (200, 200, 200), -1)   # header
    cv2.rectangle(img, (50, 200), (900, 600), (180, 180, 180), -1)   # body
    cv2.rectangle(img, (50, 700), (1230, 780), (200, 200, 200), -1)  # footer
    return img


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


@pytest.fixture
def dark_image_path(tmp_path, dark_image):
    """Save a dark-mode image to a temp file and return the path."""
    path = str(tmp_path / "dark_input.png")
    cv2.imwrite(path, dark_image)
    return path


# ── F1: Background Detection ───────────────────────────────────────────────

class TestBackgroundDetection:
    def test_white_background(self, white_image):
        bg = estimate_background_color(white_image)
        # Should detect near-white
        assert np.all(bg > 200), f"White bg detected as {bg}"

    def test_dark_background(self, dark_image):
        bg = estimate_background_color(dark_image)
        # Should detect near-black (the border is dark)
        assert np.all(bg < 60), f"Dark bg detected as {bg}"

    def test_grayscale_input(self):
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        bg = estimate_background_color(gray)
        assert abs(int(bg) - 128) < 5

    def test_mixed_borders(self):
        """Image with different bg on top vs bottom still picks mode."""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Bottom half border is darker but top has more border pixels
        img[199, :] = [200, 200, 200]
        bg = estimate_background_color(img)
        assert np.all(bg > 230), "Mode should still be near-white"


# ── F2: Content Mask ───────────────────────────────────────────────────────

class TestContentMask:
    def test_light_theme_content(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[30:60, 30:60] = [50, 50, 50]  # dark content
        bg = estimate_background_color(img)
        mask = build_content_mask(img, bg)
        assert np.all(mask[30:60, 30:60] == 255), "Content should be detected"
        assert np.all(mask[0:10, 0:10] == 0), "Background should be 0"

    def test_dark_theme_content(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 25
        img[30:60, 30:60] = [220, 220, 220]  # light content on dark bg
        bg = estimate_background_color(img)
        mask = build_content_mask(img, bg)
        assert np.all(mask[30:60, 30:60] == 255), "Light content on dark bg should be detected"
        assert np.all(mask[0:10, 0:10] == 0), "Dark background should be 0"

    def test_grayscale_content(self):
        gray = np.ones((100, 100), dtype=np.uint8) * 240
        gray[40:60, 40:60] = 50
        bg = estimate_background_color(gray)
        mask = build_content_mask(gray, bg)
        assert np.all(mask[40:60, 40:60] == 255)

    def test_tolerance_effect(self):
        """Higher tolerance means fewer pixels count as content."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[40:60, 40:60] = [240, 240, 240]  # very subtle content
        bg = estimate_background_color(img)

        mask_low = build_content_mask(img, bg, tolerance=5)
        mask_high = build_content_mask(img, bg, tolerance=20)

        content_low = np.sum(mask_low > 0)
        content_high = np.sum(mask_high > 0)
        assert content_low >= content_high, "Lower tolerance should detect more"


# ── F3: Projection Profiles ───────────────────────────────────────────────

class TestProjectionProfiles:
    def test_row_profile(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 10:90] = 255
        row_prof, _ = compute_projection_profiles(mask)
        assert row_prof[0] == 0
        assert row_prof[25] == 80
        assert row_prof[50] == 0

    def test_col_profile(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 10:90] = 255
        _, col_prof = compute_projection_profiles(mask)
        assert col_prof[0] == 0
        assert col_prof[50] == 20
        assert col_prof[95] == 0

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        row_prof, col_prof = compute_projection_profiles(mask)
        assert np.all(row_prof == 0)
        assert np.all(col_prof == 0)

    def test_full_mask(self):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        row_prof, col_prof = compute_projection_profiles(mask)
        assert np.all(row_prof == 100)
        assert np.all(col_prof == 100)


# ── F4: Whitespace Collapse ───────────────────────────────────────────────

class TestWhitespaceCollapse:
    def test_large_gap_collapses(self):
        profile = np.zeros(200, dtype=np.int32)
        profile[10:30] = 100   # content
        profile[80:120] = 100  # content (50px gap between)
        segments = _collapse_axis(profile, min_gap=24, gap_keep=8)
        total_pruned = sum(s.length for s in segments)
        assert total_pruned < 200, "Should be smaller than original"

    def test_small_gap_preserved(self):
        profile = np.zeros(100, dtype=np.int32)
        profile[10:30] = 100   # content
        profile[40:60] = 100   # content (10px gap)
        segments = _collapse_axis(profile, min_gap=24, gap_keep=8)
        total_pruned = sum(s.length for s in segments)
        # 10px leading + 20 content + 10 gap (kept) + 20 content + 40 trailing
        # The 40px trailing gap should be collapsed to 8
        assert total_pruned < 100

    def test_no_content_at_all(self):
        profile = np.zeros(100, dtype=np.int32)
        segments = _collapse_axis(profile, min_gap=24, gap_keep=8)
        total_pruned = sum(s.length for s in segments)
        # One big 100px gap, collapsed to 8
        assert total_pruned == 8

    def test_all_content(self):
        profile = np.ones(100, dtype=np.int32) * 50
        segments = _collapse_axis(profile, min_gap=24, gap_keep=8)
        total_pruned = sum(s.length for s in segments)
        assert total_pruned == 100, "All-content should not change"

    def test_content_preserved_exactly(self):
        """All content pixels must survive the collapse."""
        profile = np.zeros(300, dtype=np.int32)
        profile[20:50] = 100    # 30px content
        profile[100:180] = 200  # 80px content
        profile[250:280] = 50   # 30px content
        segments = _collapse_axis(profile, min_gap=24, gap_keep=8)

        total_content = 0
        for s in segments:
            if profile[s.original_start] > 0:
                total_content += s.length
        assert total_content == 30 + 80 + 30, "All content must be preserved"

    def test_segments_are_contiguous(self):
        """Segments in pruned space must be contiguous (no gaps or overlaps)."""
        profile = np.zeros(200, dtype=np.int32)
        profile[10:30] = 100
        profile[80:120] = 100
        segments = _collapse_axis(profile, min_gap=24, gap_keep=8)

        for i in range(1, len(segments)):
            expected_start = segments[i-1].pruned_start + segments[i-1].length
            assert segments[i].pruned_start == expected_start, \
                f"Gap in pruned coords at segment {i}: {expected_start} != {segments[i].pruned_start}"


# ── F5: Coordinate Mapping ────────────────────────────────────────────────

class TestCoordinateMapping:
    def test_round_trip_content_pixels(self, temp_image_path):
        """Pick 20 random content pixels, map original->pruned->original, error <=1px."""
        result = prune(temp_image_path)
        coord_map = result["coord_map"]
        image = cv2.imread(temp_image_path)
        bg = estimate_background_color(image)
        mask = build_content_mask(image, bg)

        # Find content pixel locations
        content_ys, content_xs = np.where(mask > 0)
        if len(content_xs) < 20:
            pytest.skip("Not enough content pixels")

        random.seed(42)
        indices = random.sample(range(len(content_xs)), 20)

        max_error = 0
        for idx in indices:
            ox, oy = int(content_xs[idx]), int(content_ys[idx])
            px, py = coord_map.original_to_pruned(ox, oy)
            rx, ry = coord_map.pruned_to_original(px, py)
            error = abs(rx - ox) + abs(ry - oy)
            max_error = max(max_error, error)
            assert error <= 1, f"Round-trip error {error}px for ({ox},{oy})->({px},{py})->({rx},{ry})"

    def test_round_trip_dark_mode(self, dark_image_path):
        """Coordinate round-trip on dark-mode image."""
        result = prune(dark_image_path)
        coord_map = result["coord_map"]
        image = cv2.imread(dark_image_path)
        bg = estimate_background_color(image)
        mask = build_content_mask(image, bg)

        content_ys, content_xs = np.where(mask > 0)
        if len(content_xs) < 20:
            pytest.skip("Not enough content pixels")

        random.seed(42)
        indices = random.sample(range(len(content_xs)), 20)

        for idx in indices:
            ox, oy = int(content_xs[idx]), int(content_ys[idx])
            px, py = coord_map.original_to_pruned(ox, oy)
            rx, ry = coord_map.pruned_to_original(px, py)
            error = abs(rx - ox) + abs(ry - oy)
            assert error <= 1, f"Dark mode round-trip error {error}px"

    def test_pruned_to_original_basic(self):
        segs = [Segment(0, 10, 20), Segment(20, 50, 30)]
        cmap = CoordinateMap(segs, segs, 100, 100, 50, 50)

        ox, oy = cmap.pruned_to_original(5, 5)
        assert ox == 15, f"Expected 15, got {ox}"
        assert oy == 15, f"Expected 15, got {oy}"

    def test_original_to_pruned_basic(self):
        segs = [Segment(0, 10, 20), Segment(20, 50, 30)]
        cmap = CoordinateMap(segs, segs, 100, 100, 50, 50)

        px, py = cmap.original_to_pruned(15, 15)
        assert px == 5, f"Expected 5, got {px}"
        assert py == 5, f"Expected 5, got {py}"


# ── Ink Retention Tests ────────────────────────────────────────────────────

class TestInkRetention:
    def test_ink_retention_light(self, temp_image_path):
        """Ink retention >= 98% on light-theme image."""
        result = prune(temp_image_path)
        ink_ret = result["metrics"]["ink_retention_pct"]
        assert ink_ret >= 98.0, f"Ink retention {ink_ret}% < 98%"

    def test_ink_retention_dark(self, dark_image_path):
        """Ink retention >= 98% on dark-mode image."""
        result = prune(dark_image_path)
        ink_ret = result["metrics"]["ink_retention_pct"]
        assert ink_ret >= 98.0, f"Ink retention {ink_ret}% < 98%"

    def test_ink_retention_dense(self, tmp_path, text_dense_image):
        """Ink retention >= 98% on dense text image."""
        path = str(tmp_path / "dense.png")
        cv2.imwrite(path, text_dense_image)
        result = prune(path)
        ink_ret = result["metrics"]["ink_retention_pct"]
        assert ink_ret >= 98.0, f"Ink retention {ink_ret}% < 98%"


# ── Full Pipeline Tests ───────────────────────────────────────────────────

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
        assert "ink_retention_pct" in m
        assert "num_content_blocks" in m
        assert isinstance(m["num_content_blocks"], int)
        assert m["original_pixels"] > 0
        assert m["pruned_pixels"] > 0

    def test_coord_map_is_coordinate_map(self, temp_image_path):
        result = prune(temp_image_path)
        assert isinstance(result["coord_map"], CoordinateMap)
        assert hasattr(result["coord_map"], "pruned_to_original")
        assert hasattr(result["coord_map"], "original_to_pruned")

    def test_mixed_image_has_reduction(self, temp_image_path):
        result = prune(temp_image_path)
        assert result["metrics"]["pixel_reduction_pct"] > 0

    def test_dark_mode_has_reduction(self, dark_image_path):
        """Dark mode images should also get some pruning (unlike v1)."""
        result = prune(dark_image_path)
        assert result["metrics"]["pixel_reduction_pct"] > 0, \
            "Dark mode should also be pruned (was a known v1 failure)"

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
        assert any("content_mask" in f for f in debug_files)

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            prune("nonexistent_image.png")

    def test_near_blank_page(self, tmp_path, white_image):
        path = str(tmp_path / "blank.png")
        cv2.imwrite(path, white_image)
        result = prune(path)
        assert result["pruned_image"] is not None

    def test_text_dense_page(self, tmp_path, text_dense_image):
        path = str(tmp_path / "dense.png")
        cv2.imwrite(path, text_dense_image)
        result = prune(path)
        # Dense page should have minimal reduction (content everywhere)
        assert result["metrics"]["pixel_reduction_pct"] < 50


# ── Sample Image Results Table ──────────────────────────────────────────────

class TestSampleImages:
    """Run pruning on sample images and print a results table.
    
    These tests use synthetic samples if real captures aren't available.
    """

    @pytest.fixture(autouse=True)
    def setup_samples(self, tmp_path):
        """Create synthetic sample images if samples/ directory doesn't exist."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.samples_dir = os.path.join(project_root, "samples")

        if not os.path.isdir(self.samples_dir) or not os.listdir(self.samples_dir):
            self.samples_dir = str(tmp_path / "samples")
            os.makedirs(self.samples_dir, exist_ok=True)
            self._create_synthetic_samples(self.samples_dir)

    def _create_synthetic_samples(self, out_dir):
        """Create minimal synthetic test images."""
        # Wikipedia light
        img = np.ones((800, 1280, 3), dtype=np.uint8) * 252
        cv2.rectangle(img, (0, 0), (1280, 45), (230, 230, 230), -1)
        cv2.rectangle(img, (0, 50), (200, 700), (245, 245, 245), -1)
        for y in range(70, 680, 25):
            cv2.putText(img, "Link", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 200), 1)
        for y in range(130, 680, 18):
            cv2.putText(img, "Body text content for wiki article.", (220, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
        cv2.imwrite(os.path.join(out_dir, "wikipedia_light.png"), img)

        # Amazon page
        img = np.ones((800, 1280, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (0, 0), (1280, 55), (30, 42, 19), -1)
        for row in range(3):
            for col in range(3):
                x, y = 240 + col*340, 70 + row*240
                cv2.rectangle(img, (x, y), (x+320, y+220), (245, 245, 245), -1)
                cv2.putText(img, "Product", (x+10, y+175), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)
        cv2.imwrite(os.path.join(out_dir, "amazon_page.png"), img)

        # Dark mode
        img = np.ones((800, 1280, 3), dtype=np.uint8) * 22
        cv2.rectangle(img, (0, 0), (1280, 50), (35, 35, 35), -1)
        for row in range(2):
            for col in range(2):
                x, y = 50 + col*600, 80 + row*300
                cv2.rectangle(img, (x, y), (x+550, y+260), (38, 38, 42), -1)
                for line in range(6):
                    cv2.putText(img, "Code on dark bg.", (x+20, y+60+line*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
        cv2.imwrite(os.path.join(out_dir, "dark_mode_page.png"), img)

        # Dense text
        img = np.ones((800, 1280, 3), dtype=np.uint8) * 252
        cv2.rectangle(img, (0, 0), (1280, 40), (230, 230, 230), -1)
        for y in range(55, 790, 14):
            cv2.putText(img, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 40, 40), 1)
        cv2.imwrite(os.path.join(out_dir, "dense_text.png"), img)

    def test_all_samples_and_print_table(self):
        """Run pruning on all samples and print a results table."""
        samples = sorted([
            f for f in os.listdir(self.samples_dir) if f.endswith(".png")
        ])
        assert len(samples) >= 4, f"Need at least 4 samples, found {len(samples)}"

        results = []
        for filename in samples:
            path = os.path.join(self.samples_dir, filename)
            result = prune(path)
            m = result["metrics"]

            # Check ink retention
            ink_ret = m["ink_retention_pct"]
            assert ink_ret >= 98.0, f"{filename}: ink retention {ink_ret}% < 98%"

            results.append({
                "file": filename,
                "area_reduction": m["pixel_reduction_pct"],
                "size_reduction": m["file_reduction_pct"],
                "ink_retention": ink_ret,
            })

            # Clean up output
            if os.path.exists(result["output_path"]):
                os.unlink(result["output_path"])

        # Print table
        print("\n" + "=" * 80)
        print("PRUNING V2 RESULTS TABLE")
        print("=" * 80)
        print(f"{'File':<25} {'Area Red %':>12} {'Size Red %':>12} {'Ink Ret %':>12}")
        print("-" * 65)
        for r in results:
            print(f"{r['file']:<25} {r['area_reduction']:>11.1f}% {r['size_reduction']:>11.1f}% {r['ink_retention']:>11.1f}%")
        print("=" * 80)


# ── Optional OCR Test ──────────────────────────────────────────────────────

class TestOCRRecall:
    """Optional OCR recall test — requires pytesseract."""

    @pytest.fixture(autouse=True)
    def check_tesseract(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.has_tesseract = True
        except Exception:
            self.has_tesseract = False

    def test_ocr_word_recall(self, temp_image_path):
        """OCR word recall of pruned vs original >= 95%."""
        if not self.has_tesseract:
            pytest.skip("pytesseract not available")

        import pytesseract

        result = prune(temp_image_path)

        original_img = Image.open(temp_image_path)
        pruned_img = result["pruned_image"]

        orig_text = pytesseract.image_to_string(original_img)
        pruned_text = pytesseract.image_to_string(pruned_img)

        orig_words = set(orig_text.lower().split())
        pruned_words = set(pruned_text.lower().split())

        if len(orig_words) == 0:
            pytest.skip("No words detected in original")

        recall = len(orig_words & pruned_words) / len(orig_words) * 100
        assert recall >= 95.0, f"OCR recall {recall:.1f}% < 95%"
