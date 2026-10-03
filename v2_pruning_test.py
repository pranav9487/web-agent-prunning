"""
v2_pruning_test.py — Verify the pruning engine end to end.

Runs the full prune() pipeline on test_screenshot.png (and any sample
screenshots), saves debug images for visual inspection, and reports
reduction metrics. Used as the build-verify loop for V2 features.
"""

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning.core import (
    prune,
    to_grayscale,
    threshold_image,
    find_content_boxes,
    merge_nearby_boxes,
    draw_debug_boxes,
    log_metrics_csv,
)
import cv2
import numpy as np


def verify_imports():
    """Phase 1: Confirm all imports work."""
    print("=" * 60)
    print("PHASE 1: Context Checks")
    print("=" * 60)

    import cv2 as _cv2
    import PIL as _pil
    import numpy as _np

    print(f"  [OK] OpenCV  : {_cv2.__version__}")
    print(f"  [OK] Pillow  : {_pil.__version__}")
    print(f"  [OK] NumPy   : {_np.__version__}")
    return True


def verify_screenshot(path="test_screenshot.png"):
    """Phase 1: Confirm test screenshot exists and is valid."""
    if not os.path.exists(path):
        print(f"  [FAIL] {path} does not exist")
        return False

    img = cv2.imread(path)
    if img is None:
        print(f"  [FAIL] {path} could not be loaded (corrupted?)")
        return False

    h, w = img.shape[:2]
    size_kb = os.path.getsize(path) / 1024
    print(f"  [OK] {path}: {w}x{h}, {size_kb:.1f} KB")

    if size_kb < 10:
        print(f"  [WARN] File is very small, may be blank")
        return False

    return True


def verify_grayscale_threshold(path="test_screenshot.png"):
    """Phase 2: Test grayscale conversion and thresholding."""
    print("\n" + "=" * 60)
    print("PHASE 2: Grayscale & Thresholding")
    print("=" * 60)

    img = cv2.imread(path)
    gray = to_grayscale(img)

    if len(gray.shape) != 2:
        print(f"  [FAIL] Grayscale output has {len(gray.shape)} channels, expected 1")
        return False
    print(f"  [OK] Grayscale: shape={gray.shape}, dtype={gray.dtype}")

    binary = threshold_image(gray)
    white_pct = np.sum(binary == 255) / binary.size * 100
    print(f"  [OK] Threshold mask: {white_pct:.1f}% content (white pixels)")

    # Save for visual inspection
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite("debug_output/grayscale.png", gray)
    cv2.imwrite("debug_output/threshold_mask.png", binary)
    print(f"  [OK] Saved debug_output/grayscale.png")
    print(f"  [OK] Saved debug_output/threshold_mask.png")

    if white_pct < 1:
        print(f"  [WARN] Very little content detected ({white_pct:.1f}%), threshold may need tuning")
    elif white_pct > 95:
        print(f"  [WARN] Almost everything detected as content ({white_pct:.1f}%), threshold may be too low")

    return True


def verify_contour_detection(path="test_screenshot.png"):
    """Phase 3: Test contour detection and bounding boxes."""
    print("\n" + "=" * 60)
    print("PHASE 3: Contour Detection")
    print("=" * 60)

    img = cv2.imread(path)
    gray = to_grayscale(img)
    binary = threshold_image(gray)

    boxes_raw = find_content_boxes(binary)
    print(f"  [OK] Raw contours found: {len(boxes_raw)}")

    boxes_merged = merge_nearby_boxes(boxes_raw)
    print(f"  [OK] After merging: {len(boxes_merged)} content blocks")

    if len(boxes_merged) == 0:
        print(f"  [FAIL] No content blocks detected!")
        return False

    # Draw and save debug overlay
    debug_img = draw_debug_boxes(img, boxes_merged)
    cv2.imwrite("debug_output/bounding_boxes.png", debug_img)
    print(f"  [OK] Saved debug_output/bounding_boxes.png")

    # Print box details
    for i, (x, y, w, h) in enumerate(boxes_merged):
        print(f"    Box {i}: x={x}, y={y}, w={w}, h={h}, area={w*h}")

    return True


def verify_crop_stitch(path="test_screenshot.png"):
    """Phase 4: Test full prune pipeline with metrics."""
    print("\n" + "=" * 60)
    print("PHASE 4: Crop, Stitch & Measure")
    print("=" * 60)

    result = prune(
        image_path=path,
        save_debug=True,
        debug_dir="debug_output",
    )

    metrics = result["metrics"]
    print(f"  [OK] Original : {metrics['original_dimensions']} ({metrics['original_file_kb']:.1f} KB)")
    print(f"  [OK] Pruned   : {metrics['pruned_dimensions']} ({metrics['pruned_file_kb']:.1f} KB)")
    print(f"  [OK] Pixel reduction : {metrics['pixel_reduction_pct']:.1f}%")
    print(f"  [OK] File reduction  : {metrics['file_reduction_pct']:.1f}%")
    print(f"  [OK] Content blocks  : {metrics['num_content_blocks']}")
    print(f"  [OK] Saved: {result['output_path']}")

    # Check coordinate mapping
    coord_map = result["coord_map"]
    num_blocks = len(coord_map["blocks"])
    print(f"  [OK] Coordinate map has {num_blocks} block entries")

    if num_blocks > 0:
        b = coord_map["blocks"][0]
        print(f"    First block: original={b['original_box']}, pruned_y={b['pruned_y_start']}..{b['pruned_y_end']}")

    return result


def verify_reduction_target(metrics, target_min=20, target_max=80):
    """Phase 5: Check if reduction falls in acceptable range."""
    pct = metrics["pixel_reduction_pct"]
    if target_min <= pct <= target_max:
        print(f"  [OK] Pixel reduction {pct:.1f}% is within target range ({target_min}%-{target_max}%)")
        return True
    else:
        print(f"  [WARN] Pixel reduction {pct:.1f}% is outside target range ({target_min}%-{target_max}%)")
        return False


def run_all_tests():
    """Run the full verification suite."""
    print("v2 Pruning Engine — Verification Suite")
    print("=" * 60)

    # Phase 1
    if not verify_imports():
        print("\n[ABORT] Imports failed")
        return False

    if not verify_screenshot():
        print("\n[ABORT] Screenshot check failed")
        return False

    print("  [OK] Phase 1 PASSED\n")

    # Phase 2
    if not verify_grayscale_threshold():
        print("\n[ABORT] Grayscale/threshold failed")
        return False
    print("  [OK] Phase 2 PASSED\n")

    # Phase 3
    if not verify_contour_detection():
        print("\n[ABORT] Contour detection failed")
        return False
    print("  [OK] Phase 3 PASSED\n")

    # Phase 4
    result = verify_crop_stitch()
    if result is None:
        print("\n[ABORT] Crop/stitch failed")
        return False
    print("  [OK] Phase 4 PASSED\n")

    # Phase 5: Validation
    print("\n" + "=" * 60)
    print("PHASE 5: Validation")
    print("=" * 60)
    metrics = result["metrics"]
    verify_reduction_target(metrics)

    # Log metrics to CSV
    metrics["image_name"] = "test_screenshot.png"
    log_metrics_csv([metrics])
    print(f"  [OK] Metrics logged to pruning_metrics.csv")

    # Run on any additional sample screenshots
    sample_dir = "sample_screenshots"
    if os.path.isdir(sample_dir):
        all_metrics = [metrics]
        samples = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
        print(f"\n  Running on {len(samples)} additional samples...")
        for fname in samples:
            fpath = os.path.join(sample_dir, fname)
            try:
                r = prune(fpath, save_debug=True, debug_dir="debug_output")
                m = r["metrics"]
                m["image_name"] = fname
                all_metrics.append(m)
                print(f"    {fname}: pixel={m['pixel_reduction_pct']:.1f}%, file={m['file_reduction_pct']:.1f}%")
            except Exception as e:
                print(f"    {fname}: [ERROR] {e}")
        log_metrics_csv(all_metrics)
        print(f"  [OK] All metrics logged to pruning_metrics.csv")

    print("\n" + "=" * 60)
    print("ALL PHASES PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
